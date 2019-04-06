import copy
from collections import Counter
from copy import deepcopy
from logging import getLogger
from typing import Optional, Dict, Any, Tuple, List, NamedTuple

import numpy as np
import scipy.sparse as sp
import sklearn
import sys

logger = getLogger(__name__)


class GcmcRatingData(NamedTuple):
    user_ids: np.ndarray
    item_ids: np.ndarray
    ratings: np.ndarray


class _IdMap(object):
    def __init__(self, ids: np.ndarray, min_count=0, max_count=sys.maxsize, use_default: bool = True) -> None:
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
        self._id2information_index = self._make_map(in_ids + out_ids, start=start)
        self._indices = self.to_indices(self.ids)

    @staticmethod
    def _make_map(xs: List, start: int = 0) -> Dict:
        return dict(zip(xs, range(start, start + len(xs))))

    def to_indices(self, ids: Any) -> np.ndarray:
        return np.array([self._id2index.get(i, self._default_index) for i in ids])

    def to_information_indices(self, ids: Any) -> np.ndarray:
        return np.array([self._id2information_index.get(i, self._default_index) for i in ids])

    def sorted_unique_ids(self) -> np.ndarray:
        return np.array(sorted(self._id2index.keys()))

    def unique_id_size(self) -> int:
        return len(self._id2index)

    @property
    def index_count(self) -> int:
        return max(self._id2index.values()) + 1

    @property
    def ids(self) -> np.ndarray:
        return self._ids

    @property
    def indices(self) -> np.ndarray:
        return self._indices


class _IdMapWithFeature(_IdMap):
    def __init__(self, ids: np.ndarray, features: Optional[List[Dict[Any, np.ndarray]]] = None, min_count=0, max_count=sys.maxsize,
                 use_default: bool = True) -> None:
        super(_IdMapWithFeature, self).__init__(ids=ids, min_count=min_count, max_count=max_count, use_default=use_default)
        self._features = self._sort_features(features=features, order_map=self._id2information_index)
        self._features_indices = self.to_information_indices(ids)
        self._original_features = features

    @property
    def features(self) -> List[np.ndarray]:
        return self._features

    @property
    def feature_indices(self) -> np.ndarray:
        return self._features_indices

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


class GcmcDataset(object):
    def __init__(self,
                 rating_data: GcmcRatingData,
                 test_size: float,
                 user_information: Optional[List[Dict[Any, np.ndarray]]] = None,
                 item_information: Optional[List[Dict[Any, np.ndarray]]] = None,
                 min_user_click_count: int = 0,
                 max_user_click_count: int = sys.maxsize) -> None:
        self._user = _IdMapWithFeature(rating_data.user_ids, features=user_information, min_count=min_user_click_count, max_count=max_user_click_count)
        self._item = _IdMapWithFeature(rating_data.item_ids, features=item_information)
        self._rating = _IdMap(rating_data.ratings, use_default=False)
        self._train_indices = np.random.uniform(0., 1., size=self._user.ids) > test_size

    def _train_adjacency_matrix(self):
        m = sp.csr_matrix((self._user.index_count, self._item.index_count), dtype=np.float32)
        idx = self._train_indices
        # add 1 to rating_indices, because rating_indices starts with 0 and 0 is ignored in scr_matrix
        m[self._user.indices[idx], self._item.indices[idx]] = self._rating.indices[idx] + 1.
        return m

    def train_rating_adjacency_matrix(self) -> List[sp.csr_matrix]:
        adjacency_matrix = self._train_adjacency_matrix()
        return [sp.csr_matrix(adjacency_matrix == r + 1., dtype=np.float32) for r in range(self._rating.index_count)]

    def add_ratings(self, additional_rating_data: GcmcRatingData, additional_item_features: Optional[List[Dict[Any, np.ndarray]]] = None) -> 'GcmcDataset':
        dataset = deepcopy(self)
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
        data['user_information'] = self._user.feature_indices[idx]
        data['item_information'] = self._item.feature_indices[idx]
        return data

    def to_indices(self, user_ids: List, item_ids: List) -> Tuple[np.ndarray, np.ndarray]:
        return self._user.to_indices(user_ids), self._item.to_indices(item_ids)

    def to_information_indices(self, user_ids: List, item_ids: List) -> Tuple[np.ndarray, np.ndarray]:
        return self._user.to_information_indices(user_ids), self._item.to_information_indices(item_ids)

    def rating(self) -> np.ndarray:
        return self._rating.sorted_unique_ids()

    def _to_one_hot(self, ratings):
        return np.eye(self._rating.index_count)[ratings]

    @property
    def n_rating(self) -> int:
        return self._rating.unique_id_size()

    @property
    def n_user(self) -> int:
        return self._user.unique_id_size()

    @property
    def n_item(self) -> int:
        return self._item.unique_id_size()

    @property
    def user_information(self) -> List[np.ndarray]:
        return self._user.features

    @property
    def item_information(self) -> List[np.ndarray]:
        return self._item.features

    @property
    def user_ids(self) -> List:
        return list(set(self._user.ids))
