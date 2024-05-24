import os
from numpy.random import seed
import random
import sys
import logging
import numpy as np
from collections import Counter
from pprint import pformat
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
import bodmas.multiple_data as multiple_data
import bodmas.utils as utils
import bodmas.classifier as bodmas_classifier
from bodmas.config import config
from bodmas.logger import init_log
from md3.md3 import MD3
from md3.md3_eval import MD3Evaluator
from d3 import D3

random.seed(1)
seed(1)
os.environ['PYTHONHASHSEED'] = '0'

def main():
    # ----------------------------------------------- #
    # 0. Init log path and parse args                 #
    # ----------------------------------------------- #

    args = utils.parse_drift_args()

    # the log file would be "./logs/concept_drift.log" and "./logs/concept_drift.log.wf" if no redirect
    log_path = './logs/concept_improved'
    if args.quiet:
        init_log(log_path, level=logging.INFO)
    else:
        init_log(log_path, level=logging.DEBUG)
    logging.warning('Running with configuration:\n' + pformat(vars(args)))
    logging.getLogger('matplotlib.font_manager').disabled = True

    task = args.task
    test_begin_time, test_end_time = args.testing_time.split(',')
    families_cnt = args.families
    interval = args.month_interval
    retrain = args.retrain

    diversity = 'no'

    sample_ratio = args.sample_ratio
    ember_ratio = args.ember_ratio  # NOTE: use 1.0 for proba and random, use 0.3 for Transcend
    SEED = args.seed  # only for drift_random, pick up 1% samples using np.random.choice with this seed.

    # ----------------------------------------------- #
    # 1. global setting and folders                   #
    # ----------------------------------------------- #

    setting = args.setting_name  # use 'ember_drift_random', 'ember_drift_transcend', 'ember_drift_proba'
    GENERAL_DATA_FOLDER = f'multiple_data'
    DATA_FOLDER = f'multiple_data/{setting}'
    MODELS_FOLDER = f'multiple_models/{setting}'
    REPORT_FOLDER = f'multiple_reports/{setting}'
    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    os.makedirs(REPORT_FOLDER, exist_ok=True)

    MONTH_LIST = multiple_data.get_testing_month_list(test_begin_time, test_end_time)
    logging.info(f'MONTH_LIST: {MONTH_LIST}')

    # -------------------------------------------------------------------- #
    # 2. load ember dataset and the baseline model pre-trained on ember    #
    # -------------------------------------------------------------------- #

    # we will fix the follwing model as the base model (randomly picked),
    # you may also change to multiple random modesl
    fixed_model_path = 'multiple_models/pretrain_ember/ember_gbdt_model_seed1.txt'
    fpr_threshold = 0.9922799999981189  # TODO: hard code for FPR 0.1%, taken from the validation set threshold in logs/pretrain_ember/seed_2_gbdt_xxx.log
    if not os.path.exists(fixed_model_path):
        logging.error(
            f'{fixed_model_path} not exist, need to execute "run_contro.sh" with ember as the training set first')
        sys.exit(-1)

    logging.info(f'load part/full of Ember data as initial training and validation set...')
    X_ember, y_ember = multiple_data.load_npz_data('ember', GENERAL_DATA_FOLDER)  # full ember dataset
    logging.info(f'load part of Ember data as initial training and validation set finished')

    # final feature vectors and labels for relabeling and testing set
    logging.info(f'start to load BODMAS npz file...')
    X, y = multiple_data.load_npz_data('bodmas', GENERAL_DATA_FOLDER)  # unnormalized data
    bodmas_metadata = multiple_data.load_meta('bodmas', GENERAL_DATA_FOLDER)  # loads metadata for bodmas dataset
    logging.info(f'start to load BODMAS npz file finished')

    # Split by months
    if interval == 1:
        iteration = list(zip(range(1, len(MONTH_LIST)), MONTH_LIST[:-1], MONTH_LIST[1:]))
    else:
        sampling_list = evenly_split_list(MONTH_LIST[:-1], interval)
        test_list = evenly_split_list(MONTH_LIST[interval:], interval)
        iteration = list(zip(range(1, len(MONTH_LIST) // interval), sampling_list, test_list))

    # Set up init classifier
    folder_name = 'multiple_models/improved'  # path to saved models

    if ember_ratio < 1.0:
        os.path.join(folder_name, 'small')

    i, first_month, second_month = iteration[0]
    logging.info(f'i: {i}, first_month: {first_month}, second_month: {second_month}')

    POSTFIX = f'ember_{ember_ratio}_sample_{first_month}_test_{second_month}_ratio_{sample_ratio}_random_{SEED}'

    X_pretrain = X_ember
    y_pretrain = y_ember

    if ember_ratio < 1.0:
        # prune the dataset
        logging.info(f'pruning pretrain dataset to {ember_ratio} size')
        selection_size = int(ember_ratio * X_pretrain.shape[0])
        selected_indices = np.random.choice(X_pretrain.shape[0], size=selection_size, replace=False)

        X_pretrain = X_pretrain[selected_indices]
        y_pretrain = y_pretrain[selected_indices]

    # split for training and validation 80:20
    X_train, X_val, y_train, y_val = train_test_split(X_pretrain, y_pretrain, test_size=0.2, random_state=2,
                                                      shuffle=True)

    logging.info(f'Mixed training set: {X_train.shape}, mixed validation set: {X_val.shape}')
    logging.info(f'training label: {Counter(y_train)}')
    logging.info(f'validation label: {Counter(y_val)}')

    clf = args.classifier

    SAVED_MODEL_PATH = os.path.join(folder_name, f'improved_base_initial_setup_{SEED}.txt')
    gbdt_clf = bodmas_classifier.GBDTClassifierOld(saved_model_path=SAVED_MODEL_PATH)
    logging.debug(f'model path: {SAVED_MODEL_PATH}')
    base_model = gbdt_clf.train(X_train, y_train, task, families_cnt, retrain, config['gbdt_params'])

    # -------------------- #
    # MD3 set up           #
    # -------------------- #

    logging.info(f'initialising MD3')

    # load MD3 class and set reference
    md3_lgb = MD3(K=5, sensitivity=0.02, bound=0.2)  # K is number of gbdt clf in the MD3,
    md3_lgb.set_reference(X_train, y_train, task, families_cnt, retrain, folder_name, setting='improved')
    md3_lgb.set_reference_acc(base_model, X_val, y_val)

    # try to detect drift in the
    drift = md3_lgb.suspect_drift(X_val)

    if drift:
        raise Exception('Drift confirmed on validation, this should not happen.')

    logging.info(f'MD3 initialisation finished, testing drift in validation data {drift}')
    logging.info(f'Evaluator setup')
    report_path = os.path.join(REPORT_FOLDER, f'improved_{sample_ratio}',
                               f'improved_{task}_report_{POSTFIX}_{timer()}.csv')
    utils.create_parent_folder(report_path)
    evaluator = MD3Evaluator(report_path, REPORT_FOLDER, test_begin_time, test_end_time, SEED)
    logging.info(f'Evaluating base')
    evaluator.evaluate_classifier(base_model, X_val, X_val, y_val, y_val, first_month, POSTFIX)

    # D3 setup

    ws = [1000]
    rhos = [0.2]
    aucs = [0.85]
    n_features = X_val[0].shape[0]

    D3_win = D3(ws[0], rhos[0], n_features, aucs[0])

    # contains all data that was used to train the models
    X_all = X_train
    y_all = y_train

    X_new = None
    y_new = None

    sample_count = 0
    labelled_untrained = 0

    logging.info(f'started iterating through samples')
    for idx, sample_month, test_month in iteration:
        begin_month = timer()
        logging.critical(f'{idx}, {sample_month}, {test_month}')

        if type(sample_month) is list:
            sample_month_str = '_'.join(sample_month)
            test_month_str = '_'.join(test_month)
        else:
            sample_month_str = sample_month
            test_month_str = test_month

        POSTFIX = f'ember_{ember_ratio}_sample_{sample_month_str}_test_{test_month_str}_ratio_{sample_ratio}_random_{SEED}'
        SAVED_DATA_PATH = os.path.join(DATA_FOLDER, f'X_and_y_{POSTFIX}.h5')

        logging.info(f'idx-{idx} start to extract sampling month, test month data... ')

        X_sample_full, y_sample_full, X_test, y_test = \
            multiple_data.load_ember_drift_data(X, y, sample_month, test_month, GENERAL_DATA_FOLDER, SAVED_DATA_PATH)

        logging.info(f'idx-{idx} sampling, testing set prepared')

        X_sample = X_sample_full
        y_sample = y_sample_full

        logging.info(f'idx-{idx} {setting} sampling: {X_sample.shape}, {y_sample.shape}')

        # split samples into num_chunks parts
        num_chunks = 15

        X_sample_chunks = np.array_split(X_sample, num_chunks)
        y_sample_chunks = np.array_split(y_sample, num_chunks)

        # test chunks for drift
        drift_d3 = False
        drift_suspected = False

        for i, (X_chunk, y_chunk) in enumerate(zip(X_sample_chunks, y_sample_chunks)):
            for X_d3_input, y_d3_input in zip(X_chunk, y_chunk):
                if D3_win.isEmpty():
                    D3_win.addInstance(X_d3_input, y_d3_input)
                else:
                    if D3_win.driftCheck():
                        logging.info(f'D3 drift detected')
                        drift_d3 = True
                    D3_win.addInstance(X_d3_input, y_d3_input)

            if drift_d3:
                drift_suspected = md3_lgb.suspect_drift(X_chunk)
                drift_d3 = False

            logging.info(f'chunk i: {i} {sample_month_str}, drift suspected: {drift_suspected}')

            if X_new is None:
                X_new = X_chunk
                y_new = y_chunk
            else:
                X_new = np.vstack((X_new, X_chunk))
                y_new = np.hstack((y_new, y_chunk))

            if drift_suspected:
                drift_suspected = False
                logging.info(f'testing id- {i} chunk of {sample_month_str} for confirming drift')
                drift = md3_lgb.confirm_drift(X_chunk, y_chunk, base_model, drop=0.005)
                logging.critical(f'drift confirmed: {drift}')

                if drift:
                    month_retrain = True
                    logging.critical(f'RECALCULATING DRIFT')
                    logging.info(f'evaluating model before retraining')
                    eval_name = f'{sample_month}_chunk_{i}_on_{test_month_str}'
                    evaluator.evaluate_classifier(base_model, X_sample_full, X_test, y_sample_full, y_test, eval_name,
                                                  POSTFIX)

                    # use 5 % of new data
                    X_new_skip, X_new_train, y_new_skip, y_new_train = train_test_split(X_new, y_new, test_size=0.05,
                                                                                        random_state=2, shuffle=True)

                    # use 95 % of base
                    X_all_base, X_all_test, y_all_base, y_all_test = train_test_split(X_all, y_all, test_size=0.95,
                                                                                      random_state=2, shuffle=True)

                    X_train = np.vstack((X_new_train, X_all_base))
                    y_train = np.hstack((y_new_train, y_all_base))

                    if X_new_train.shape[0] > labelled_untrained:
                        sample_count += X_new_train.shape[0]
                    else:
                        sample_count += labelled_untrained

                    labelled_untrained = 0

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
                    SAVED_MODEL_PATH = os.path.join(MODELS_FOLDER, f'improved-{postfix}.txt')
                    gbdt_clf = bodmas_classifier.GBDTClassifierOld(saved_model_path=SAVED_MODEL_PATH)
                    logging.debug(f'model path: {SAVED_MODEL_PATH}')
                    base_model = gbdt_clf.train(X_train, y_train, task, families_cnt, retrain, config['gbdt_params'])

                    # calibrate md3
                    new_folder = os.path.join(folder_name, f'month-{sample_month_str}-chunk-{i}')
                    md3_lgb.set_reference(X_all_test, y_all_test, task, families_cnt, retrain, new_folder,
                                          setting='improved')
                    md3_lgb.set_reference_acc(base_model, X_all_test, y_all_test)

                    # update learning dataset
                    X_all = np.vstack((X_all, X_new))
                    y_all = np.hstack((y_all, y_new))

                    X_new = None
                    y_new = None

                else:
                    # update reference accuracy
                    logging.info('resetting accuracy of for the MD3')
                    # sample_count += X_chunk.shape[0]
                    labelled_untrained += X_chunk.shape[0]

                    md3_lgb.set_reference_acc(base_model, X_new, y_new)

        # evaluate classifier at the end of every month
        evaluator.evaluate_classifier(base_model, X_sample_full, X_test, y_sample_full, y_test, test_month, POSTFIX)

        logging.info(f'month {sample_month_str} finished in {timer() - begin_month:.1f} seconds')
        logging.info(f'sample count {sample_count}')

    sample_count += labelled_untrained
    logging.info(f'total used samples {sample_count}')


def evenly_split_list(list1, interval):
    # remove the last element if cannot evenly split
    list2 = [list1[i:i + interval] for i in range(0, len(list1), interval)]
    if len(list2[-1]) != interval:
        list2 = list2[:-1]
    return list2


if __name__ == "__main__":
    start = timer()
    main()
    end = timer()
    logging.info(f'time elapsed: {end - start:.2f}')
