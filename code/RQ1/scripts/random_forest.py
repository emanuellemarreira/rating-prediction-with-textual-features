from train_model import TrainModel
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

params = {
    "n_estimators": [100, 200, 300, 500, 1000],
    "max_depth": [5, 10, 30, None],
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5, None],
    "class_weight": [None, "balanced", "balanced_subsample"],
    "max_leaf_nodes": [None, 10, 20],
    "min_impurity_decrease": [0.0, 0.01, 0.1],
    "random_state":[None]
}

model = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ1'
)

model.train()