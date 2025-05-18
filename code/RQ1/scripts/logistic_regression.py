from train_model import TrainModel
from sklearn.linear_model import LogisticRegression

clf =  LogisticRegression()

params = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "class_weight": [None, "balanced"],
    "random_state": [None],
    "max_iter": [10000],
}

model = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ1',
)

model.train()