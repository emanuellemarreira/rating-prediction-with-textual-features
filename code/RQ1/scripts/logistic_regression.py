import pandas as pd
import numpy as np
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
import scipy.stats as stats


from sklearnex import patch_sklearn
patch_sklearn()


clf =  LogisticRegression()

params = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "class_weight": [None, "balanced"],
    "random_state": [None],
    "max_iter": [10000],
}


i = 1
for param in params.keys():
  i*=len(params[param])
iter = i
n_iter = int(10/100*iter)
print(f'Número de combinações possíveis: {iter}')
print(f'Número de combinações a serem testadas: {n_iter}')


save_path = '../../../code/RQ1/results'
dir_train = '../../../dataset/train.csv'
dir_test = '../../../dataset/test.csv'
df_train = pd.read_csv(dir_train)
df_test = pd.read_csv(dir_test)
df = pd.concat([df_train, df_test])
df['categoria_rating'] = df['categoria'].astype(str) + "_" + df['rating'].astype(str)


X = df.drop(columns=['categoria','text','rating','categoria_rating'])
y = df['rating']


kfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
splits = list(kfold.split(X, df['categoria_rating']))


random_search = RandomizedSearchCV(
    clf,
    param_distributions = params,
    n_iter = 300,
    scoring = 'neg_root_mean_squared_error',
    cv = splits,
    random_state = 42,
    verbose = 3)


random_search.fit(X, y)


best_params = random_search.best_params_
print(f'Melhores parâmetros: {best_params}')


def calculate_metrics(y_pred, y_test):
      report = metrics.classification_report(y_test, y_pred, target_names=list(map(str, [1,2,3,4,5])), output_dict=True)
      cm = confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5])
      f1_macro = report['macro avg']['f1-score']

      mae = mean_absolute_error(y_test, y_pred)
      rmse = np.sqrt(mean_squared_error(y_test, y_pred))

      y_true_bin = np.array([1 if y in [4,5] else 0 for y in y_test])
      y_pred_bin = np.array([1 if y in [4,5] else 0 for y in y_pred])
      auc = roc_auc_score(y_true_bin, y_pred_bin)

      return report, round(f1_macro,4), cm, round(mae,4), round(rmse,4), round(auc,4)


all_metrics = []


for i, (train_idx, test_idx) in enumerate(splits):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = clf
    model.set_params(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n***FOLD {i + 1}***")
    all_metrics.append((calculate_metrics(y_pred, y_test)))


for i, (report, f1_macro, cm, mae, rmse, auc) in enumerate(all_metrics):
    print(f"\n***FOLD {i + 1}***")
    print(f"\nClassification Report:\n{report}")
    print(f"\nF1 Macro: {f1_macro}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nMAE: {mae}")
    print(f"\nRMSE: {rmse}")
    print(f"\nAUC: {auc}")


results = pd.DataFrame(columns = ['MAE','STD_MAE','RMSE','STD_RMSE','AUC','STD_AUC','CM'])
for i, (report, f1_macro, cm, mae, rmse, auc) in enumerate(all_metrics):
    results.loc[i] = [mae, 0, rmse, 0, auc, 0,cm]


mean_mae = results['MAE'].mean()
mean_rmse = results['RMSE'].mean()
mean_auc = results['AUC'].mean()
std_mae = results['MAE'].std()
std_rmse = results['RMSE'].std()
std_auc = results['AUC'].std()
mean = {'MAE': round(mean_mae,4), 'STD_MAE': round(std_mae,4), 'RMSE': round(mean_rmse,4), 'STD_RMSE': round(std_rmse,4), 'AUC': round(mean_auc,4), 'STD_AUC': round(std_auc,4),'CM':"--"}
results = pd.concat([results, pd.DataFrame([mean])], ignore_index=True)


std_err_mae = stats.sem(results['MAE'])
std_err_rmse = stats.sem(results['RMSE'])
std_err_auc = stats.sem(results['AUC'])

conf_int_mae = stats.t.interval(0.95, len(results['MAE'])-1, loc=mean_mae, scale=std_err_mae)
conf_int_mae = [round(float(conf_int_mae[0]),4), round(float(conf_int_mae[1]),4)]
conf_int_rmse = stats.t.interval(0.95, len(results['RMSE'])-1, loc=mean_rmse, scale=std_err_rmse)
conf_int_rmse = [round(float(conf_int_rmse[0]),4), round(float(conf_int_rmse[1]),4)]
conf_int_auc = stats.t.interval(0.95, len(results['AUC'])-1, loc=mean_auc, scale=std_err_auc)
conf_int_auc = [round(float(conf_int_auc[0]),4), round(float(conf_int_auc[1]),4)]

print(f'MAE: {mean_mae:.4f} ± {std_mae:.4f} 95% IC: [{conf_int_mae[0]:.4f}, {conf_int_mae[1]:.4f}]')
print(f'RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} 95% IC: [{conf_int_rmse[0]:.4f}, {conf_int_rmse[1]:.4f}]')
print(f'AUC: {mean_auc:.4f} ± {std_auc:.4f} 95% IC: [{conf_int_auc[0]:.4f}, {conf_int_auc[1]:.4f}]')


conf_int = {'MAE': conf_int_mae, 'STD_MAE': round(std_err_mae,4), 'RMSE': conf_int_rmse, 'STD_RMSE': round(std_err_rmse,4), 'AUC': conf_int_auc, 'STD_AUC': round(std_err_auc,4), 'CM': '--'}
results = pd.concat([results, pd.DataFrame([conf_int])], ignore_index=True)


results.to_csv(f'{save_path}/{clf.__class__.__name__}_results.csv')


