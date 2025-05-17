from train_model import TrainModel
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier()

params = {
    "n_estimators": [100, 200, 300, 500, 1000],
    "learning_rate": [0.001, 0.01, 0.05, 0.1],
    "max_depth": [3, 5, 10],
    "min_samples_split": [5, 10, 15],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5, None],
    "random_state": [42]
}

groups = ['conc', 'lex', 'pos', 'sbj', 'str', 'synt','twt']

model_conc = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ2',
    group = 'conc'
)

model_conc.train()

model_lex = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ2',
    group = 'lex'
)

model_lex.train()

model_pos = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ2',
    group = 'pos'
)

model_pos.train()

model_sbj = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ2',
    group = 'sbj'
)

model_sbj.train()

model_str = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ2',
    group = 'str'
)

model_synt = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ2',
    group = 'synt'
)

model_synt.train()

model_twt = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ2',
    group = 'twt'
)

model_twt.train()