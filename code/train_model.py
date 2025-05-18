from sklearnex import patch_sklearn
patch_sklearn()
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
import scipy.stats as stats
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
DIR_TRAIN = os.path.join(DATASET_DIR, 'train.csv')
DIR_TEST = os.path.join(DATASET_DIR, 'test.csv')
SAVE_PATH = os.path.join(PROJECT_ROOT, 'code') 

class TrainModel:
    def __init__(self, model, params, rq, 
                 group = 'all',
                 category = 'all', 
                 classes = [1,2,3,4,5], 
                 n_iter = 300, 
                 n_splits = 5,
                 dir_train = DIR_TRAIN,
                 dir_test = DIR_TEST,
                 save_path = SAVE_PATH,
                 #dir_train = '../../../dataset/train.csv',
                 #dir_test = '../../../dataset/test.csv',
                 #save_path = '../../../code',
                 ablation = False,
                 using_tfidf = False,
                 ):
        self.clf = model
        self.params = params
        self.classes = classes
        self.dir_train = dir_train
        self.dir_test = dir_test
        self.save_path = f'{save_path}/{rq}/results'
        self.rq = rq
        self.group = group
        self.n_iter = n_iter
        self.n_splits = n_splits
        self.ablation = ablation
        self.category = category
        self.using_tfidf = using_tfidf
    
    def train_test_split(self):
        df_train = pd.read_csv(self.dir_train)
        df_test = pd.read_csv(self.dir_test)
        df = pd.concat([df_train, df_test])
        df['categoria_rating'] = df['categoria'].astype(str) + "_" + df['rating'].astype(str)
        if self.classes == [0,1,2,3,4]:
            df['rating'] = df['rating'] - 1
            df_train['rating'] = df_train['rating'] - 1
            df_test['rating'] = df_test['rating'] - 1
        if self.group != 'all' and not self.ablation:    
            drop = [column for column in df.columns if not column.startswith(self.group)]
            drop.extend(['categoria','text','rating','categoria_rating'])
            X = df.drop(columns=drop)
            print(f"{self.group}: {X.shape}")
        elif self.group != 'all' and self.ablation:
            drop = [column for column in df.columns if column.startswith(self.group)]
            drop.extend(['categoria','text','rating','categoria_rating'])
            X = df.drop(columns=drop)
            print(f"all - {self.group}: {X.shape}")
        elif self.category != 'all':
            df_train.drop(df_train[df_train["categoria"] != self.category].index, inplace=True)
            df_test.drop(df_test[df_test["categoria"] != self.category].index, inplace=True)
            df = pd.concat([df_train, df_test])
            df['categoria_rating'] = df['categoria'].astype(str) + "_" + df['rating'].astype(str)
            drop = ['categoria','text','rating','categoria_rating']
            X = df.drop(columns=drop)
            print(f"{self.category}: {X.shape}")
        elif self.using_tfidf:
            tfidf = TfidfVectorizer(
                        min_df=5,
                        encoding='utf-8',
                        ngram_range=(1, 2),
                        lowercase=True,
                        stop_words=nltk.corpus.stopwords.words('portuguese')
                    )
            X = tfidf.fit_transform(df.text.values.astype('U')).toarray()
        else:
            X = df.drop(columns=['categoria','text','rating','categoria_rating'])
        y = df['rating']
        return X, y, df
    
    def do_kfold(self):
        X, y, df = self.train_test_split()
        kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=False, random_state=None)
        splits = list(kfold.split(X, df['categoria_rating']))
        return X, y, splits

    def perform_random_search(self):
        X, y, splits = self.do_kfold()

        random_search = RandomizedSearchCV(
            self.clf,
            param_distributions = self.params,
            n_iter = self.n_iter,
            scoring = 'neg_root_mean_squared_error',
            cv = splits,
            random_state = 42,
            verbose = 2,
            n_jobs = -1
        )

        random_search.fit(X, y)
        return random_search, X, y, splits
    
    def train(self):
        random_search, X, y, splits = self.perform_random_search()

        best_params = random_search.best_params_

        all_metrics = []
        for i, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = self.clf
            model.set_params(**best_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            print(f"\nFOLD {i + 1}...")
            all_metrics.append((self.calculate_metrics(y_pred, y_test)))

        print("finished.")

        self.save_results(best_params, all_metrics)
    
    def calculate_metrics(self, y_pred, y_test):
        report = metrics.classification_report(y_test, y_pred, target_names=list(map(str, self.classes)), output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=self.classes)
        f1_macro = report['macro avg']['f1-score']

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        y_true_bin = np.array([1 if y in [self.classes[-1], self.classes[-2]] else 0 for y in y_test])
        y_pred_bin = np.array([1 if y in [self.classes[-1], self.classes[-2]] else 0 for y in y_pred])
        auc = roc_auc_score(y_true_bin, y_pred_bin)

        return report, round(f1_macro,4), cm, round(mae,4), round(rmse,4), round(auc,4)
    
    def save_results(self, best_params, all_metrics):
        print(f'Melhores parâmetros: {best_params}')

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

        if self.group != 'all' and not self.ablation:
            with open(f"{self.save_path}/{self.clf.__class__.__name__}_{self.group}_best_params.txt", 'w') as f:
                f.write(str(best_params))
        elif self.group != 'all' and self.ablation:
            with open(f"{self.save_path}/ablation_{self.clf.__class__.__name__}_all_except_{self.group}_best_params.txt", 'w') as f:
                f.write(str(best_params))
        elif self.category != 'all':
            with open(f"{self.save_path}/categoria_{self.category}_{self.clf.__class__.__name__}_best_params.txt", 'w') as f:
                f.write(str(best_params))
        elif self.using_tfidf:
            with open(f"{self.save_path}/{self.clf.__class__.__name__}_tfidf_best_params.txt", 'w') as f:
                f.write(str(best_params))
        else:
            with open(f"{self.save_path}/{self.clf.__class__.__name__}_best_params.txt", 'w') as f:
                f.write(str(best_params))

        if self.group != 'all' and not self.ablation:
            results.to_csv(f'{self.save_path}/{self.clf.__class__.__name__}_{self.group}_results.csv')
        elif self.group != 'all' and self.ablation:
            results.to_csv(f'{self.save_path}/ablation_{self.clf.__class__.__name__}_all_except_{self.group}_results.csv')
        elif self.category != 'all':
            results.to_csv(f'{self.save_path}/categoria_{self.category}_{self.clf.__class__.__name__}_results.csv')
        elif self.using_tfidf:
            results.to_csv(f'{self.save_path}/{self.clf.__class__.__name__}_tfidf_results.csv')
        else:
            results.to_csv(f'{self.save_path}/{self.clf.__class__.__name__}_results.csv')