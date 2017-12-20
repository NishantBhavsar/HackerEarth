import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# read train and test data from files
train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')

# features required for training this model
features_name = [i for i in train.columns.values if i not in ['connection_id', 'target']]
# target data
target = train['target']


X_train, X_valid, y_train, y_valid = train_test_split(train, target, train_size=0.8, stratify=target, random_state=2017)


# a function to train the model and check its accuracy score, features importance
def modelfit(model, train_data, train_label, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=100):

    if useTrainCV:
        xgb_param = model.get_xgb_params()
        params = {
            'objective': xgb_param['objective'],
            'num_class': 3,
            'base_score': xgb_param['base_score'],
            'colsample_bylevel': xgb_param['colsample_bylevel'],
            'colsample_bytree': xgb_param['colsample_bytree'],
            'gamma': xgb_param['gamma'],
            'eta': xgb_param['learning_rate'],
            'max_delta_step': xgb_param['max_delta_step'],
            'max_depth': xgb_param['max_depth'],
            'min_child_weight': xgb_param['min_child_weight'],
            'alpha': xgb_param['reg_alpha'],
            'lambda': xgb_param['reg_lambda'],
            'scale_pos_weight': xgb_param['scale_pos_weight'],
            'subsample': xgb_param['subsample']
        }

        dtrain = xgb.DMatrix(data=train_data[predictors], label=train_label)
        cvresult = xgb.cv(params, dtrain, num_boost_round=model.get_params()['n_estimators'], stratified=True, nfold=cv_folds, metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)
        model.set_params(n_estimators=cvresult.shape[0])
        # print(cvresult)
        print("########### n_estimators = %f" % cvresult.shape[0])

    # Fit the algorithm on the data
    model.fit(train_data[predictors], train_label, eval_metric='auc')

    # Predict training set:
    train_predictions = model.predict(train_data[predictors])

    # Predict X_valid set:
    valid_predictions = model.predict(X_valid[predictors])

    # Print model report:
    print("\nModel Report")
    print("Accuracy (Train): %.5g" % accuracy_score(train_label, train_predictions))
    print("Accuracy (Validation): %.5g" % accuracy_score(y_valid, valid_predictions))

    feat_imp = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
    return model


# a XGBoost classifier with booster parameters
clf = xgb.XGBClassifier(objective='multi:softmax', learning_rate=0.1, max_depth=4, n_estimators=190, subsample=0.8, colsample_bytree=0.6, gamma=0.1, min_child_weight=1, reg_alpha=0.001)

# train the XGBoost classifier using modelfit function
clf = modelfit(clf, train, target, features_name, False)

# predict test data set
pred = clf.predict(test[features_name])

# make submission
sub = pd.read_csv('sample_submission.csv')
sub['target'] = pred
sub['target'] = sub['target'].astype(int)
sub.to_csv('sub_xgb.csv', index=False)
