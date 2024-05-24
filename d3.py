import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as AUC
from sklearn.preprocessing import MinMaxScaler
from skmultiflow.data.data_stream import DataStream
import bodmas.multiple_data as multiple_data
import bodmas.classifier as bodmas_classifier
import itertools
import os
import logging
from bodmas.config import config
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
import bodmas.utils as utils
from bodmas.logger import init_log
import warnings
from md3.md3_eval import MD3Evaluator
import lightgbm as lgb


def drift_detector(S, T, threshold=0.75):
    T = pd.DataFrame(T)
    S = pd.DataFrame(S)
    T['in_target'] = 0  # in target set
    S['in_target'] = 1  # in source set
    ST = pd.concat([T, S], ignore_index=True, axis=0)
    labels = ST['in_target'].values
    ST = ST.drop('in_target', axis=1).values
    clf = lgb.LGBMClassifier(**config['d3_params'])
    predictions = np.zeros(labels.shape)
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    for train_idx, test_idx in skf.split(ST, labels):
        X_train, X_test = ST[train_idx], ST[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        predictions[test_idx] = probs
    auc_score = AUC(labels, predictions)
    if auc_score > threshold:
        return True
    else:
        return False


class D3:
    def __init__(self, w, rho, dim, auc):
        self.size = int(w * (1 + rho))
        self.win_data = np.zeros((self.size, dim))
        self.win_label = np.zeros(self.size)
        self.w = w
        self.rho = rho
        self.dim = dim
        self.auc = auc
        self.drift_count = 0
        self.window_index = 0

    def addInstance(self, X, y):
        if (self.isEmpty()):
            # print(self.win_data.shape)
            self.win_data[self.window_index] = X
            self.win_label[self.window_index] = y
            self.window_index = self.window_index + 1
        else:
            print("Error: Buffer is full!")

    def isEmpty(self):
        return self.window_index < self.size

    def driftCheck(self):
        # logging.info(self.win_data.shape)
        if drift_detector(self.win_data[:self.w], self.win_data[self.w:self.size], self.auc):
            self.window_index = int(self.w * self.rho)
            self.win_data = np.roll(self.win_data, -1 * self.w, axis=0)
            self.win_label = np.roll(self.win_label, -1 * self.w, axis=0)
            self.drift_count = self.drift_count + 1
            return True
        else:
            self.window_index = self.w
            self.win_data = np.roll(self.win_data, -1 * (int(self.w * self.rho)), axis=0)
            self.win_label = np.roll(self.win_label, -1 * (int(self.w * self.rho)), axis=0)
            return False

    def getCurrentData(self):
        return self.win_data[:self.window_index]

    def getCurrentLabels(self):
        return self.win_label[:self.window_index]


def select_data(x):
    df = pd.read_csv(x)
    scaler = MinMaxScaler()
    df.iloc[:, 0:df.shape[1] - 1] = scaler.fit_transform(df.iloc[:, 0:df.shape[1] - 1])
    return df


def check_true(y, y_hat):
    if (y == y_hat):
        return 1
    else:
        return 0


def load_npz_data(file_name, folder_path):
    data = np.load(f"{folder_path}/{file_name}.npz")
    return data['X'], data['y']


def main():
    SEED = 1
    log_path = 'logs/d3'
    init_log(log_path, level=logging.DEBUG)

    X_npz, y_npz = load_npz_data('bodmas', "multiple_data")

    print(np.any(np.isnan(X_npz)), np.any(np.isnan(y_npz)))

    df = pd.DataFrame(X_npz, columns=[f'feature_{i}' for i in range(X_npz.shape[1])])
    df['target'] = y_npz

    metadata = pd.read_csv("multiple_data/bodmas_metadata.csv")
    df = pd.concat([df, metadata.reset_index(drop=True)], axis=1)

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, format='mixed')

    # Extract the month and year from the timestamp
    df['month_year'] = df['timestamp'].dt.to_period('M')

    start_date = pd.to_datetime('2019-09-01', utc=True)
    df_train = df[df['timestamp'] <= start_date]
    df = df[df['timestamp'] >= start_date]

    # set up classifier
    task = 'binary'
    test_begin_time = '2019-09'
    test_end_time = '2020-09'
    families_cnt = 2
    GENERAL_DATA_FOLDER = 'multiple_data'
    DATA_FOLDER = 'multiple_data/d3'
    SAVED_DATA_PATH = os.path.join(DATA_FOLDER, f'X_and_y_d3_r{SEED}.h5')

    X_train_origin, y_train_origin, X_test_list, y_test_list = \
        multiple_data.load_bluehex_data(task, test_begin_time, test_end_time, families_cnt,
                                        is_normalize=True,
                                        general_data_folder=GENERAL_DATA_FOLDER,
                                        setting_data_folder=DATA_FOLDER,
                                        saved_data_path=SAVED_DATA_PATH)

    X_train, X_val, y_train, y_val = train_test_split(X_train_origin, y_train_origin, test_size=0.2, random_state=SEED,
                                                      shuffle=True)

    SAVED_MODEL_PATH = 'multiple_models/pretrain_ember/d3_gbdt_model_seed1.txt'
    gbdt_clf = bodmas_classifier.GBDTClassifierOld(saved_model_path=SAVED_MODEL_PATH)
    logging.debug(f'model path: {SAVED_MODEL_PATH}')
    base_model = gbdt_clf.train(X_train, y_train, task, families_cnt, True, config['gbdt_params'])

    REPORT_FOLDER = 'multiple_reports/d3'
    timer_postfix = timer()
    evaluator = MD3Evaluator(f'multiple_reports/d3/gbdt_d3_full_r{SEED}_{timer_postfix}', REPORT_FOLDER,
                             test_begin_time, test_end_time, SEED)

    metadata_new = df[['timestamp']]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        metadata_new['month_year'] = metadata_new['timestamp'].dt.to_period('M')

    df = df.drop(columns=['sha', 'family', 'timestamp', 'month_year'])

    features = df.drop(columns=['target'])  # Adjust according to your dataframe structure

    scaler = MinMaxScaler()
    df[features.columns] = scaler.fit_transform(features)

    stream = DataStream(df)
    stream.prepare_for_use()

    args = utils.parse_multiple_dataset_args()

    ws = args.w
    rhos = args.rho
    aucs = args.auc

    for combinations in itertools.product(ws, rhos, aucs):
        print(combinations)
        w, rho, auc = combinations  # values for D3

        D3_win = D3(w, rho, stream.n_features, auc)
        i = 0  # counter

        X_new, y_new = [], []  # new incoming samples

        # data that was already used for training
        X_all = X_train
        y_all = y_train

        MONTH_LIST = multiple_data.get_testing_month_list(test_begin_time, test_end_time)
        iteration = list(zip(range(1, len(MONTH_LIST)), MONTH_LIST[:-1], MONTH_LIST[1:]))

        logging.info(f'start to load BODMAS npz file...')
        X, y = multiple_data.load_npz_data('bodmas', GENERAL_DATA_FOLDER)  # unnormalized data
        logging.info(f'start to load BODMAS npz file finished')

        samples_sum = 0

        for idx, sample_month, test_month in iteration:
            begin_month = timer()
            logging.critical(f'{idx}, {sample_month}, {test_month}')

            if type(sample_month) is list:
                sample_month_str = '_'.join(sample_month)
                test_month_str = '_'.join(test_month)
            else:
                sample_month_str = sample_month
                test_month_str = test_month

            POSTFIX = f'ember_test_{test_month_str}_ratio_random_{SEED}'
            SAVED_DATA_PATH = os.path.join(DATA_FOLDER, f'X_and_y_{POSTFIX}.h5')
            logging.info(f'idx-{idx} start to extract sampling month, test month data... ')

            # sample data are for validation, test data are the next month and are used for prediction
            X_sample_full, y_sample_full, X_test, y_test = \
                multiple_data.load_ember_drift_data(X, y, sample_month, test_month, GENERAL_DATA_FOLDER,
                                                    SAVED_DATA_PATH)

            logging.info(f'idx-{idx} sampling, testing set prepared')

            X_sample = X_sample_full
            y_sample = y_sample_full

            logging.info(f'idx-{idx} gbdt sampling: {X_sample.shape}, {y_sample.shape}')

            for i, (one_X, one_y) in enumerate(zip(X_sample, y_sample)):
                X_new.append(one_X)
                y_new.append(one_y)

                if D3_win.isEmpty():
                    D3_win.addInstance(one_X, one_y)
                elif D3_win.driftCheck():  # drift found
                    logging.critical(f'RECALCULATING DRIFT')
                    logging.info(f'evaluating model before retraining')
                    eval_name = f'{sample_month}_chunk_{i}_on_{test_month_str}'
                    evaluator.evaluate_classifier(base_model, X_sample_full, X_test, y_sample_full, y_test, eval_name,
                                                  POSTFIX)

                    # use 10 % of new data
                    X_new_skip, X_new_train, y_new_skip, y_new_train = train_test_split(X_new, y_new, test_size=0.10,
                                                                                        random_state=2, shuffle=True)

                    # use 90 % of base
                    X_all_base, X_all_test, y_all_base, y_all_test = train_test_split(X_all, y_all, test_size=0.10,
                                                                                      random_state=2, shuffle=True)

                    samples_sum += len(X_new_train)
                    X_train = np.vstack((X_new_train, X_all_base))
                    y_train = np.hstack((y_new_train, y_all_base))

                    max_sample_size = 10000

                    if X_train.shape[0] > max_sample_size:
                        logging.info(f'pruning pretrain dataset to {max_sample_size} size')
                        selection_size = max_sample_size
                        selected_indices = np.random.choice(X_train.shape[0], size=selection_size, replace=False)

                        X_train = X_train[selected_indices]
                        y_train = y_train[selected_indices]

                    # retrain base model
                    postfix = f'{idx}-{sample_month_str}-chunk-{i}'
                    logging.critical(f'{X_train.shape[0]} used to retrain at {postfix}.')
                    SAVED_MODEL_PATH = os.path.join('multiple_models/d3', f'd3-{postfix}.txt')
                    gbdt_clf = bodmas_classifier.GBDTClassifierOld(saved_model_path=SAVED_MODEL_PATH)
                    logging.debug(f'model path: {SAVED_MODEL_PATH}')
                    base_model = gbdt_clf.train(X_train, y_train, task, families_cnt, True, config['gbdt_params'])

                    # update learning dataset
                    X_all = np.vstack((X_all, X_new))
                    y_all = np.hstack((y_all, y_new))

                    X_new = []
                    y_new = []

                    D3_win.addInstance(one_X, one_y)

            # evaluate classifier at the end of every month
            evaluator.evaluate_classifier(base_model, X_sample_full, X_test, y_sample_full, y_test, test_month, POSTFIX)

            logging.info(f'month {sample_month_str} finished in {timer() - begin_month:.1f} seconds')

        logging.info(f'samples sum {samples_sum}')


if __name__ == '__main__':
    main()
