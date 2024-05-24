config = {
    'md3_params': {
        "boosting": "gbdt",
        "objective": "binary",
        "num_iterations": 1000,
        "learning_rate": 0.01,
        "num_leaves": 1024,
        "max_depth": 10,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.5,
        "verbosity": -1,  # 1 means INFO, > 1 means DEBUG, 0 means Error(WARNING), <0 means Fatal
        "device_type": "gpu"
    },
    'gbdt_params': {
        "boosting": "gbdt",
        "objective": "binary",
        "num_iterations": 1000,
        "learning_rate": 0.05,
        "num_leaves": 2048,
        "max_depth": 15,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.5,
        "verbosity": -1,  # 1 means INFO, > 1 means DEBUG, 0 means Error(WARNING), <0 means Fatal
        "device_type": "gpu"
    },
    'd3_params': {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': 15,
        'max_depth': 5,
        'min_data_in_leaf': 20,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbosity': -1
    }
}
