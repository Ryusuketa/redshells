import sys
from collections import Counter
from copy import deepcopy
from logging import getLogger
from typing import Optional, Dict, Any, Tuple, List, Set

import numpy as np
import scipy.sparse as sp
import sklearn

logger = getLogger(__name__)


class GcmcDataset(object):
    def __init__(self,
                 user_ids: np.ndarray,
                 item_ids: np.ndarray,
                 ratings: np.ndarray,
                 user_features: Optional[List[Dict[Any, np.ndarray]]] = None,
                 item_features: Optional[List[Dict[Any, np.ndarray]]] = None) -> None:
        self.size = len(user_ids)
        assert len(item_ids) == self.size
        assert len(ratings) == self.size
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.user_features = user_features
        self.item_features = item_features


class GcmcIdMap(object):
    def __init__(self, ids: np.ndarray, features: Optional[List[Dict[Any, np.ndarray]]] = None, min_count=0, max_count=sys.maxsize,
                 use_default: bool = True) -> None:
        self._ids = ids
        id_count = dict(Counter(ids))
        in_ids = sorted([i for i, c in id_count.items() if min_count <= c <= max_count])
        out_ids = sorted(list(set(id_count.keys()) - set(in_ids)))

        if use_default:
            self._default_index = 0
            start = 1
        else:
            self._default_index = None
            start = 0

        self._id2index = self._make_map(in_ids, start=start)
        self._id2feature_index = self._make_map(in_ids + out_ids, start=start)
        self._indices = self.to_indices(self.ids)
        self._features = features
        self._feature_indices = self.to_feature_indices(ids)
        self._feature_matrix = self._sort_features(features=features, order_map=self._id2feature_index)

    def add(self, ids: np.ndarray, features: Optional[List[Dict[Any, np.ndarray]]] = None) -> None:
        new_ids = set(ids) - set(self._ids)
        self._ids = np.concatenate([self._ids, ids])
        self._id2index = self._update_map(self._id2index, new_ids)
        self._id2feature_index = self._update_map(self._id2feature_index, new_ids)
        self._indices = np.concatenate([self._indices, self.to_indices(ids)])
        if features:

            def _update(x, y):
                x.update(y)
                return x

            self._features = [_update(original, new) for original, new in zip(self._features, features)]
        self._feature_indices = np.concatenate([self._feature_indices, self.to_feature_indices(ids)])
        self._feature_matrix = self._sort_features(self._features, self._id2feature_index)

    @staticmethod
    def _update_map(id_map: Dict, new_ids: Set) -> Dict:
        max_index = max(id_map.values())
        id_map.update(dict(zip(new_ids, range(max_index + 1, max_index + 1 + len(new_ids)))))
        return id_map

    @staticmethod
    def _make_map(xs: List, start: int = 0) -> Dict:
        return dict(zip(xs, range(start, start + len(xs))))

    def to_indices(self, ids: Any) -> np.ndarray:
        return np.array([self._id2index.get(i, self._default_index) for i in ids])

    def to_feature_indices(self, ids: Any) -> np.ndarray:
        return np.array([self._id2feature_index.get(i, self._default_index) for i in ids])

    def sorted_unique_ids(self) -> np.ndarray:
        return np.array(sorted(self._id2index.keys()))

    @property
    def index_count(self) -> int:
        return max(self._id2index.values()) + 1

    @property
    def ids(self) -> np.ndarray:
        return self._ids

    @property
    def indices(self) -> np.ndarray:
        return self._indices

    @property
    def features(self) -> List[Dict[Any, np.ndarray]]:
        return self._features

    @property
    def feature_matrix(self) -> List[np.ndarray]:
        return self._feature_matrix

    @property
    def feature_indices(self) -> np.ndarray:
        return self._feature_indices

    @staticmethod
    def _sort_features_impl(features: Dict[Any, np.ndarray], order_map: Dict) -> np.ndarray:
        def _get_feature_size(values):
            for v in (v for v in values if v is not None):
                return len(v)
            return 0

        feature_size = _get_feature_size(features.values())
        new_order, _ = zip(*list(sorted(order_map.items(), key=lambda x: x[1])))
        sorted_features = np.array(list(map(lambda x: features.get(x, np.zeros(feature_size)), new_order)))
        sorted_features = np.vstack([np.zeros(feature_size), sorted_features])
        return sorted_features.astype(np.float32)

    @classmethod
    def _sort_features(cls, features: List[Dict[Any, np.ndarray]], order_map: Dict) -> List[np.ndarray]:
        if features is None:
            return []
        return [cls._sort_features_impl(feature, order_map) for feature in features]


class GcmcGraphDataset(object):
    def __init__(self, dataset: GcmcDataset, test_size: float, min_user_click_count: int = 0, max_user_click_count: int = sys.maxsize) -> None:
        self._user = GcmcIdMap(dataset.user_ids, features=dataset.user_features, min_count=min_user_click_count, max_count=max_user_click_count)
        self._item = GcmcIdMap(dataset.item_ids, features=dataset.item_features)
        self._rating = GcmcIdMap(dataset.ratings, use_default=False)
        self._train_indices = np.random.uniform(0., 1., size=len(self._user.ids)) > test_size

    def _train_adjacency_matrix(self) -> sp.csr_matrix:
        m = sp.csr_matrix((self._user.index_count, self._item.index_count), dtype=np.float32)
        idx = self._train_indices
        # add 1 to rating_indices, because rating_indices starts with 0 and 0 is ignored in scr_matrix
        # `lil_matrix` is too slow
        m[self._user.indices[idx], self._item.indices[idx]] = self._rating.indices[idx] + 1.
        return m

    def train_rating_adjacency_matrix(self) -> List[sp.csr_matrix]:
        adjacency_matrix = self._train_adjacency_matrix()
        return [sp.csr_matrix(adjacency_matrix == r + 1., dtype=np.float32) for r in range(self._rating.index_count)]

    def add_dataset(self, additional_dataset: GcmcDataset) -> 'GcmcGraphDataset':
        dataset = deepcopy(self)
        dataset._user.add(additional_dataset.user_ids, additional_dataset.user_features)
        dataset._item.add(additional_dataset.item_ids, additional_dataset.item_features)
        dataset._rating.add(additional_dataset.ratings)
        dataset._train_indices = np.concatenate([dataset._train_indices, np.array([True] * additional_dataset.size)])
        return dataset

    def train_data(self):
        idx = self._train_indices
        shuffle_idx = sklearn.utils.shuffle(list(range(int(np.sum(idx)))))
        data = self._get_data(idx=idx)
        data = {k: v[shuffle_idx] for k, v in data.items()}
        return data

    def test_data(self):
        return self._get_data(idx=~self._train_indices)

    def _get_data(self, idx):
        data = dict()
        data['user'] = self._user.indices[idx]
        data['item'] = self._item.indices[idx]
        data['label'] = self._to_one_hot(self._rating.indices[idx])
        data['rating'] = self._rating.ids[idx]
        data['user_feature_indices'] = self._user.feature_indices[idx]
        data['item_feature_indices'] = self._item.feature_indices[idx]
        return data

    def to_indices(self, user_ids: List, item_ids: List) -> Tuple[np.ndarray, np.ndarray]:
        return self._user.to_indices(user_ids), self._item.to_indices(item_ids)

    def to_feature_indices(self, user_ids: List, item_ids: List) -> Tuple[np.ndarray, np.ndarray]:
        return self._user.to_feature_indices(user_ids), self._item.to_feature_indices(item_ids)

    def rating(self) -> np.ndarray:
        return self._rating.sorted_unique_ids()

    def _to_one_hot(self, ratings):
        return np.eye(self._rating.index_count)[ratings]

    @property
    def n_rating(self) -> int:
        return self._rating.index_count

    @property
    def n_user(self) -> int:
        return self._user.index_count

    @property
    def n_item(self) -> int:
        return self._item.index_count

    @property
    def user_features(self) -> List[np.ndarray]:
        return self._user.feature_matrix

    @property
    def item_features(self) -> List[np.ndarray]:
        return self._item.feature_matrix

    @property
    def user_ids(self) -> List:
        return list(set(self._user.ids))
