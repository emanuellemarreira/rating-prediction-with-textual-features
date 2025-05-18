from sklearnex import patch_sklearn
patch_sklearn()
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
import scipy.stats as stats
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import make_scorer

dir_train = '../../../dataset/train.csv'
dir_test = '../../../dataset/test.csv'
save_path = '../../../code/RQ3/results'

best_params = {'subsample': 0.8, 'reg_lambda': 1, 'reg_alpha': 0, 'n_estimators': 500, 'min_child_weight': 1, 'max_depth': 10, 'learning_rate': 0.05, 'gamma': 0.0, 'colsample_bytree': 0.6}

estimator = XGBClassifier(**best_params)

df_train = pd.read_csv(dir_train)
df_test = pd.read_csv(dir_test)
df = pd.concat([df_train, df_test])
df['categoria_rating'] = df['categoria'].astype(str) + "_" + df['rating'].astype(str)
X = df.drop(columns=['categoria','text','rating','categoria_rating'])
y = df['rating'] - 1

kfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
splits = list(kfold.split(X, df['categoria_rating']))

features = X.columns
min_features = 1
max_features = len(features)

def define_auc(y_true, y_pred):
    y_true_bin = np.array([1 if y in [3, 4] else 0 for y in y_true])
    y_pred_bin = np.array([1 if y in [3, 4] else 0 for y in y_pred])
    return roc_auc_score(y_true_bin, y_pred_bin)

metrics = {
    'mae': make_scorer(mean_absolute_error),
    'rmse': make_scorer(mean_squared_error),
    'auc': make_scorer(define_auc)
}

first_iter = True

while max_features >= min_features:
    selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
    X_selected = selector.fit_transform(X, y)
    selected_features = [features[i] for i, selected in enumerate(selector.get_support()) if selected]

    scores = cross_validate(estimator, X_selected, y, cv=splits, scoring=metrics, verbose = 5)

    metrics_results = {metric: scores[f'test_{metric}'].mean() for metric in metrics}
    metrics_results_std = {metric: scores[f'test_{metric}'].std() for metric in metrics}

    rmses = [np.sqrt(mse) for mse in scores['test_rmse']]
    metrics_results['rmse'] = np.mean(rmses)
    metrics_results_std['rmse'] = np.std(rmses)


    row = pd.DataFrame([{
        'iter': max_features,
        'features': selected_features,
        'mae': metrics_results['mae'],
        'mae_std': metrics_results_std['mae'],
        'rmse': metrics_results['rmse'],
        'rmse_std': metrics_results_std['rmse'],
        'auc': metrics_results['auc'],
        'auc_std': metrics_results_std['auc']
    }])

    row.to_csv(f'{save_path}/{max_features}_results.csv', mode='a', header=first_iter, index=False)
    first_iter = False

    max_features -= 1