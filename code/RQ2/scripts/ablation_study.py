from code.train_model import TrainModel
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

model_all_except_conc = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ2',
    group = 'conc',
    classes = [0,1,2,3,4], 
    ablation = True
)

model_all_except_conc.train()

model_all_except_lex = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ2',
    group = 'lex',
    classes = [0,1,2,3,4], 
    ablation = True
)

model_all_except_lex.train()

model_all_except_pos = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ2',
    group = 'pos',
    classes = [0,1,2,3,4], 
    ablation = True
)

model_all_except_pos.train()

model_all_except_sbj = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ2',
    group = 'sbj',
    classes = [0,1,2,3,4], 
    ablation = True
)

model_all_except_sbj.train()

model_all_except_str = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ2',
    group = 'str',
    classes = [0,1,2,3,4], 
    ablation = True
)

model_all_except_str.train()

model_all_except_synt = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ2',
    group = 'synt',
    classes = [0,1,2,3,4], 
    ablation = True
)

model_all_except_synt.train()

model_all_except_twt = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ2',
    group = 'twt',
    classes = [0,1,2,3,4], 
    ablation = True
)

model_all_except_twt.train()