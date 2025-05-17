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


categories = ['auto', 'baby','celular','food','games','laptops','livros','moda','pets','toys']

model_auto = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    category='auto',
)

model_auto.train()

model_baby = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    category='baby',
)

model_celular = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    category='celular',
)

model_celular.train()

model_food = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    category='food',
)
model_food.train()

model_games = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    category='games',
)
model_games.train()

model_laptops = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    category='laptops',
)
model_laptops.train()

model_livros = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    category='livros',
)
model_livros.train()

model_moda = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    category='moda',
)
model_moda.train()

model_pets = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    category='pets',
)
model_pets.train()

model_toys = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    category='toys',
)
model_toys.train()