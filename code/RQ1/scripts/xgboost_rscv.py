from train_model import TrainModel
from xgboost import XGBClassifier

clf = XGBClassifier()

params =  {
 "n_estimators": [100, 200, 300, 500],
 "learning_rate": [0.05, 0.10, 0.20],
 "max_depth": [6, 8, 10],
 "min_child_weight": [1, 3, 5],
 "gamma": [0.0, 0.1, 0.3],
 "subsample": [0.6, 0.8, 1.0],
 "colsample_bytree": [0.4, 0.6, 0.8],
 "reg_alpha": [0, 0.1, 1],
 "reg_lambda": [1, 1.5, 2]
}

model = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ1',
    classes = [0,1,2,3,4],
)

model.train()