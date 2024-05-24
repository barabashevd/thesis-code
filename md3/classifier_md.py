import os
import logging
from timeit import default_timer as timer
import lightgbm as lgb


class GBDTForMD3(object):
    def __init__(self, saved_model_path):
        self.saved_model_path = saved_model_path

    def train(self, X_train, y_train, task, families_cnt, retrain, params={}):
        lgbm_model = None

        if not os.path.exists(self.saved_model_path) or retrain:
            begin = timer()
            logging.info('Training LightGBM model...')
            # params.update({'application': 'binary'}) # not needed since 'application' is an alias to 'objective'
            if task != 'binary':
                params.update({'objective': 'multiclass',
                               'metric': 'multi_logloss', # alias for softmax
                               'num_class': families_cnt})

            # Train, it would use all 24 CPUs
            lgbm_dataset = lgb.Dataset(X_train, y_train)
            lgbm_model = lgb.train(params, lgbm_dataset)
            lgbm_model.save_model(self.saved_model_path)
            end = timer()
            logging.info(f'Training LightGBM finished, time: {end - begin:.1f} seconds')
        else:
            logging.info("Loading pre-trained LightGBM model...")
            lgbm_model = lgb.Booster(model_file=self.saved_model_path)

        return lgbm_model
