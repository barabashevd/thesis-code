import os

import numpy as np
import logging
from bodmas.config import config
from sklearn.model_selection import train_test_split
from .margin_detector import MarginDetector
from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix, accuracy_score


class MD3:
    def __init__(self, sensitivity=2, K=10, bound=0.25):
        """
        MD3 implementation for LightGBM.

        Parameters:
        - sensitivity: Sensitivity for detecting drift based on changes in margin density.
        - k: Number of models to train for ensemble validation
        """
        self.sensitivity = sensitivity
        self.K = K
        self.classifiers = None
        self.reference_md = None
        self.reference_acc = None
        self.reference_std = None
        self.reference_std = None
        self.drifting = False
        self.bound = bound
        self.current_md = 0
        self.lambda_ = 0

    def set_reference(self, X, y, task, families_cnt, retrain, folder_name, setting='md3'):
        """
        Set the reference margin density and its standard deviation using k-fold cross-validation.

        Parameters:
        - X: Feature matrix of the initial dataset.
        - y: Target vector of the initial dataset.
        """

        # model_path = 'multiple_models/md3'
        model_path = folder_name

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self.classifiers = MarginDetector(saved_model_path=model_path, num_subspaces=self.K, feature_subset_ratio=0.5,
                                          lower_bound=self.bound, upper_bound=(1 - self.bound), setting=setting)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)
        self.lambda_ = (X_train.shape[1] - 1) / X_val.shape[1]

        begin = timer()
        logging.info(f'training {self.K} models for with {self.K} subspaces for MD3')
        self.classifiers.train(X_train, y_train, task, families_cnt, retrain, config['md3_params'])
        logging.info(f'MD3 loading finished in {timer() - begin:.1f} seconds.')

        logging.info(f'Evaluating models.')

        # Store the evaluation metrics for each model
        model_metrics = []

        for i, (clf, feature_indices) in enumerate(zip(self.classifiers.models, self.classifiers.feature_subspaces)):
            # Subset the validation data to the features used by this model
            feature_indices = np.array(feature_indices, dtype=int)
            X_val_subspace = X_val[:, feature_indices]

            # Make predictions
            y_pred = clf.predict(X_val_subspace)

            # Calculate the confusion matrix
            y_pred = (y_pred >= 0.5).astype(int)
            cm = confusion_matrix(y_val, y_pred)
            TN, FP, FN, TP = cm.ravel()

            # Calculate FPR and TPR from the confusion matrix
            FPR = FP / (FP + TN)
            TPR = TP / (TP + FN)  # Also known as recall

            # Calculate accuracy
            acc = accuracy_score(y_val, y_pred)

            # Store metrics for this model
            model_metrics.append({
                "model_index": i,
                "confusion_matrix": cm,
                "FPR": FPR,
                "TPR": TPR,
                "accuracy": acc
            })

            # Optionally, print out metrics for each model
            logging.info(f'Model {i}: FPR: {FPR:.4f}, TPR: {TPR:.4f}, Accuracy: {acc:.4f}')
            logging.info(f'Confusion Matrix: \n{cm}')

        sum_fpr = 0
        sum_tpr = 0
        sum_accuracy = 0
        num_models = len(model_metrics)

        # Sum up all metrics
        for metrics in model_metrics:
            sum_fpr += metrics['FPR']
            sum_tpr += metrics['TPR']
            sum_accuracy += metrics['accuracy']

        # Calculate averages
        avg_fpr = sum_fpr / num_models
        avg_tpr = sum_tpr / num_models
        avg_accuracy = sum_accuracy / num_models

        # Return a dictionary of average metrics
        average_metrics = {
            "average_FPR": avg_fpr,
            "average_TPR": avg_tpr,
            "average_accuracy": avg_accuracy
        }

        logging.info(f'Average metrics: {average_metrics}')

        self.reference_md, self.reference_std = self.classifiers.calculate_margin_density(X_val)
        self.current_md = self.reference_md
        logging.info(f'Reference MD from training data: {self.reference_md} std: {self.reference_std}')

    def suspect_drift(self, X):
        new_md, new_std = self.classifiers.calculate_margin_density(X)

        self.lambda_ = (X.shape[0] - 1) / X.shape[0]

        self.current_md = (self.lambda_ * self.current_md + (1 - self.lambda_) * new_md)

        # Check if the change in margin density exceeds the threshold
        if abs(self.current_md - self.reference_md) > (self.sensitivity * self.reference_std):
            self.drifting = True

        logging.info(f'current_md: {self.current_md}, reference_md: {self.reference_md}')
        logging.info(f'MD calculation: {self.current_md - self.reference_md} > {self.sensitivity * self.reference_std}')

        return self.drifting

    def confirm_drift(self, X, y, model, drop=0.01):
        # If currently drifting and enough labeled samples are collected
        if self.drifting:
            # Make decision based on labeled samples
            y_pred = model.predict(X)
            y_pred = (y_pred >= 0.5).astype(int)
            new_accuracy = accuracy_score(y, y_pred)
            logging.critical(f'Reference accuracy {self.reference_acc} and new accuracy is {new_accuracy}')
            logging.info(
                f'Calculation {self.reference_acc} - {new_accuracy} = {self.reference_acc - new_accuracy} > 0.005')
            if self.reference_acc - new_accuracy > drop:
                return True
        return False

    def set_reference_acc(self, model, X, y):
        """
        Evaluates the given model on the provided data (X, y) to compute the accuracy,
        and stores this accuracy as the reference accuracy for the model.

        Parameters:
        - model: The model to evaluate.
        - X: The feature matrix for evaluation.
        - y: The ground truth labels.
        """
        # Predict the labels for the dataset
        y_pred = model.predict(X)
        y_pred = (y_pred >= 0.5).astype(int)
        accuracy = accuracy_score(y, y_pred)
        logging.info(f'set the reference accuracy to {accuracy}')
        self.reference_acc = accuracy
