import numpy as np
import os
import logging
from timeit import default_timer as timer
from bodmas.config import config
from .classifier_md import GBDTForMD3


class MarginDetector(object):
    def __init__(self, saved_model_path, seed=1, num_subspaces=0, feature_subset_ratio=0.5, lower_bound=0.2,
                 upper_bound=0.8, setting='md3'):
        self.models = []
        self.saved_model_path = saved_model_path
        self.num_subspaces = num_subspaces  # Number of feature subspaces (or models) to generate
        self.feature_subset_ratio = feature_subset_ratio
        self.feature_subspaces = []
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.seed = seed
        self.setting = setting

    def train(self, X_train, y_train, task, families_cnt, retrain, params={}):

        logging.info('Training LightGBM model for MD3 classifier...')

        subspaces_file = os.path.join(self.saved_model_path,
                                      f"feature_subspaces_{self.num_subspaces}_seed_{self.seed}.npy")

        if not os.path.exists(subspaces_file) or retrain:
            self.feature_subspaces = self.generate_feature_subspaces(X_train)
            self.save_subspaces(self.feature_subspaces, subspaces_file)
        else:
            logging.info(f'Loading subspaces from {subspaces_file}')
            self.load_subspaces(subspaces_file)

        for i, feature_indices in enumerate(self.feature_subspaces):
            model_name = os.path.join(self.saved_model_path,
                                      f'gbdt_{self.setting}_ensemble_{i}_of_{self.num_subspaces}_ratio{self.feature_subset_ratio}_seed{self.seed}.joblib')

            logging.info(f'Starting working on model {i}: {model_name}')

            begin = timer()

            X_sub = X_train[:, feature_indices]
            y_sub = y_train

            md3_gbdt = GBDTForMD3(saved_model_path=model_name)
            clf = md3_gbdt.train(X_sub, y_sub, task, families_cnt, retrain, config['md3_params'])
            self.models.append(clf)

            logging.info(f'Preparation {i} finished, time {timer() - begin:.1f} seconds.')

    def generate_feature_subspaces(self, X_train):
        """
        Generates random subsets of feature indices for each model in the ensemble.

        Parameters:
        - X_train: The full training feature matrix.

        Returns:
        - A list of arrays, each containing indices for a feature subset.
        """
        np.random.seed(self.seed)
        logging.info(f'X_train shape {X_train.shape}, picking {X_train.shape[1]}')
        total_features = X_train.shape[1]
        subset_size = int(np.floor(total_features * self.feature_subset_ratio))

        feature_subspaces = []

        for _ in range(self.num_subspaces):
            # Randomly select subset_size features without replacement
            feature_indices = np.random.choice(total_features, size=subset_size, replace=False)
            feature_subspaces.append(feature_indices)

        return feature_subspaces

    def calculate_margin_density(self, X_new):
        """
        Calculate the margin density for new data across all trained classifiers.

        Parameters:
        - X_new: New feature matrix.

        Returns:
        - Overall margin density across all classifiers.
        """

        margin_densities = []

        for clf, feature_indices in zip(self.models, self.feature_subspaces):
            X_subset = X_new[:, feature_indices]
            probabilities = clf.predict(X_subset)

            # Assuming binary classification
            is_in_margin = (probabilities > self.lower_bound) & (probabilities < self.upper_bound)
            margin_densities.append(np.mean(is_in_margin))

        # Calculate overall margin density
        overall_margin_density = np.mean(margin_densities)
        std_deviation = np.std(margin_densities)

        return overall_margin_density, std_deviation

    def load_subspaces(self, file_path="multiple_models/md3/feature_subspaces.npy"):
        feature_subspaces_array = np.load(file_path, allow_pickle=True)
        self.feature_subspaces = [np.array(subspace, dtype=np.int64) for subspace in feature_subspaces_array]

    def save_subspaces(self, feature_subspaces, file_path="multiple_models/md3/feature_subspaces.npy"):
        feature_subspaces_array = np.array(feature_subspaces, dtype=object)
        np.save(file_path, feature_subspaces_array)
