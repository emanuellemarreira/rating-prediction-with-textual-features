from code.train_model import TrainModel
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

model = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ1',
)

model.train()