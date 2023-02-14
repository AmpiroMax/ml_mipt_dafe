""" Implementation of LambdaMART """

import random
from typing import List, Tuple

import logging
import pickle
import numpy as np
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

plt.style.use("ggplot")

logging.basicConfig(level="INFO")
logger = logging.getLogger("SOLUTION")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


class Solution:

    def __init__(
        self,
        n_estimators: int = 100,
        lr: float = 0.5,
        ndcg_top_k: int = 10,
        subsample: float = 0.6,
        colsample_bytree: float = 0.9,
        max_depth: int = 5,
        min_samples_leaf: int = 8
    ):
        logger.debug("Initializing class...")
        self.x_train = np.array(0)
        self.ys_train = np.array(0)
        self.x_test = np.array(0)
        self.ys_test = np.array(0)
        self._prepare_data()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.trees: List[DecisionTreeRegressor] = []
        self.features_indices = []

        self.all_ndcg_train: List[float] = []
        self.best_train_ndcg = float(0.0)
        self.all_ndcg_test: List[float] = []
        self.best_test_ndcg = float(0.0)
        logger.debug("Class was initialized")

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        x_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        x_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        data = [x_train, y_train, query_ids_train,
                x_test, y_test, query_ids_test]
        return data

    def _prepare_data(self) -> None:
        logger.info("Data preporation ...")
        x_train, y_train, self.query_ids_train, \
            x_test, y_test, self.query_ids_test = self._get_data()

        self.x_train = self._scale_features_in_query_groups(
            x_train, self.query_ids_train
        )
        self.ys_train = y_train[..., np.newaxis]

        self.x_test = self._scale_features_in_query_groups(
            x_test, self.query_ids_test
        )
        self.ys_test = y_test[..., np.newaxis]
        logger.info("Data was prepared!")

    def _scale_features_in_query_groups(
            self,
            inp_feat_array: np.ndarray,
            inp_query_ids: np.ndarray
    ) -> np.ndarray:

        for idx in np.unique(inp_query_ids):
            mask = inp_query_ids == idx
            scaler = StandardScaler()
            inp_feat_array[mask] = scaler.fit_transform(
                inp_feat_array[mask]
            )

        return inp_feat_array

    def _train_one_tree(
        self,
        cur_tree_idx: int,
        train_preds: np.ndarray
    ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        """
        Метод для тренировки одного дерева.

        @cur_tree_idx: номер текущего дерева, который предлагается
                       использовать в качестве random_seed для того,
                       чтобы алгоритм был детерминирован.
        @train_preds: суммарные предсказания всех предыдущих деревьев
                      (для расчёта лямбд).
        @return: это само дерево и индексы признаков,
                 на которых обучалось дерево
        """
        logger.debug("Training Tree...")
        curr_tree = DecisionTreeRegressor(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=cur_tree_idx
        )

        uniques = np.unique(self.query_ids_train)
        train_query_indeces = np.random.choice(
            uniques,
            size=int(self.subsample * uniques.shape[0]),
            replace=False
        )
        feature_indices = np.random.choice(
            self.x_train.shape[1],
            size=int(self.colsample_bytree * self.x_train.shape[1]),
            replace=False
        )

        train_data = None
        mask = None
        lambdas = np.zeros_like(self.query_ids_train, dtype=np.float32)

        for curr_idx in train_query_indeces:
            curr_mask = self.query_ids_train == curr_idx
            curr_lambdas = self._compute_lambdas(
                self.ys_train[curr_mask],
                train_preds[curr_mask]
            ).squeeze()

            if mask is None:
                mask = curr_mask
            else:
                mask |= curr_mask
            lambdas[curr_mask] = curr_lambdas

        train_data = self.x_train[mask][:, feature_indices]
        train_lambdas = lambdas[mask]

        curr_tree.fit(
            train_data,
            train_lambdas
        )

        logger.debug("Tree was fitted!")
        return (curr_tree, feature_indices)

    def _calc_data_ndcg(
        self,
        queries_list: np.ndarray,
        true_labels:  np.ndarray,
        preds:  np.ndarray
    ) -> float:
        """ Расчёт метрики по набору данных """
        mean_ndcgs = 0.0
        unique_queries = np.unique(queries_list)

        for curr_idx in unique_queries:
            mask = queries_list == curr_idx
            mean_ndcgs += self._ndcg_k(
                true_labels[mask],
                preds[mask]
            )

        return mean_ndcgs / np.unique(queries_list).size

    def fit(self) -> None:
        """
        генеральный метод обучения K деревьев, каждое из которых тренируется
        с использованием метода _train_one_tree
        """
        logger.info("Fitting the model...")
        set_seed(69)
        train_prediction = np.zeros(
            self.ys_train.shape[0], dtype=np.float32
        )
        test_prediction = np.zeros(
            self.ys_test.shape[0], dtype=np.float32
        )

        idx_of_best_ndcg = 0
        for iteration in tqdm(range(self.n_estimators)):
            new_tree, features_indices = self._train_one_tree(
                iteration,
                train_prediction
            )

            train_curr_preds = new_tree.predict(
                self.x_train[:, features_indices]
            )
            train_prediction -= self.lr * train_curr_preds
            test_curr_preds = new_tree.predict(
                self.x_test[:, features_indices]
            )
            test_prediction -= self.lr * test_curr_preds

            curr_train_ndcg = self._calc_data_ndcg(
                self.query_ids_train,
                self.ys_train,
                train_prediction
            )
            curr_test_ndcg = self._calc_data_ndcg(
                self.query_ids_test,
                self.ys_test,
                test_prediction
            )

            self.trees += [new_tree]
            self.all_ndcg_train += [curr_train_ndcg]
            self.all_ndcg_test += [curr_test_ndcg]
            self.features_indices += [features_indices]

            if self.all_ndcg_train[-1] > self.best_train_ndcg:
                self.best_train_ndcg = self.all_ndcg_train[-1]
            if self.all_ndcg_test[-1] > self.best_test_ndcg:
                self.best_test_ndcg = self.all_ndcg_test[-1]
                idx_of_best_ndcg = iteration

        self.trees = self.trees[:idx_of_best_ndcg]
        self.features_indices = self.features_indices[:idx_of_best_ndcg]

        logger.info("Model was fitted!")
        logger.info(f"Best train NDSG {self.best_train_ndcg}")
        logger.info(f"Best test NDSG  {self.best_test_ndcg}")

    def predict(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        """ Making prediction """
        predictions = np.zeros(data.shape[0])

        for i, tree in enumerate(self.trees):
            features = self.features_indices[i]
            predictions -= self.lr * tree.predict(data[:, features])

        return predictions

    def plot_ndcgs(self):
        """ Plotting learning history """
        plt.plot(self.all_ndcg_train, label="train")
        plt.plot(self.all_ndcg_test, label="test")
        plt.legend()
        plt.show()

    def _compute_labels_in_batch(
        self,
        y_true:  np.ndarray
    ) -> np.ndarray:
        rel_diff = y_true - y_true.T
        pos_pairs = (rel_diff > 0).astype(np.float32)
        neg_pairs = (rel_diff < 0).astype(np.float32)
        s_ij = pos_pairs - neg_pairs
        return s_ij

    def _compute_lambdas(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        logger.debug("Computing lambdas...")

        ideal_dcg = self._dcg(y_true, y_true, -1)
        if ideal_dcg <= 0:
            return np.zeros_like(y_true, dtype=np.float32)
        scaling_factor = 1 / ideal_dcg

        log_rank_order = np.log2(np.argsort(
            -y_true, axis=0) + 2
        )

        pos_pairs_score_diff = 1.0 + np.exp((y_pred[..., np.newaxis] - y_pred))
        s_ij = self._compute_labels_in_batch(y_true)

        gain_diff = 2.0 ** y_true - 2.0 ** y_true.T

        decay_diff = (1.0 / log_rank_order) - (1.0 / log_rank_order.T)

        delta_ndcg = np.abs(scaling_factor * gain_diff * decay_diff)

        lambda_update = (0.5 * (1 - s_ij) - 1 /
                         pos_pairs_score_diff) * delta_ndcg

        lambda_update = np.sum(
            lambda_update,
            axis=1,
            keepdims=True
        )
        logger.debug("Lambdas computed!")
        return lambda_update

    def _dcg(
        self,
        ys_true: np.ndarray,
        ys_pred: np.ndarray,
        top_k: int
    ) -> float:

        ys_true = np.squeeze(ys_true)
        ys_pred = np.squeeze(ys_pred)

        if top_k == -1:
            top_k = ys_true.shape[0]
        top_k = min(ys_true.shape[0], top_k)

        idx = np.argsort(-ys_pred, axis=0)
        gains = ys_true[idx][:top_k]
        gains = 2 ** gains - 1

        steps = np.arange(2, 2 + top_k, dtype=np.float32)
        steps = np.log2(steps)

        return float(np.sum(gains / steps))

    def _ndcg_k(
        self,
        ys_true: np.ndarray,
        ys_pred: np.ndarray
    ) -> float:
        pred_dcg = self._dcg(ys_true, ys_pred, self.ndcg_top_k)
        ideal_dcg = self._dcg(ys_true, ys_true, self.ndcg_top_k)

        if ideal_dcg == 0:
            return float(0)

        return float(pred_dcg / ideal_dcg)

    def save_model(self, path: str) -> None:
        """ Saving model """
        field_to_drop = [
            "x_train", "x_test", "ys_train", "ys_test",
            "query_ids_train", "query_ids_test"
        ]

        state_dict = {
            key: value for key, value in self.__dict__.items()
            if not (key in field_to_drop)
        }
        with open(path, "wb") as file:
            pickle.dump(state_dict, file)

    def load_model(self, path: str) -> None:
        """ Loading model """
        with open(path, 'rb') as file:
            state_dict = pickle.load(file)

            if isinstance(state_dict, dict):
                for key, value in state_dict.items():
                    self.__dict__[key] = value
                return
            raise TypeError("Expected type Dict in pkl file")


if __name__ == "__main__":
    sol = Solution()
    sol.fit()
    sol.plot_ndcgs()
