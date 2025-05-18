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


categories = ['auto', 'baby','celular','food','games','laptops','livros','moda','pets','toys']

model_auto = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    classes=[0,1,2,3,4],
    category='auto',
)

model_auto.train()

model_baby = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    classes=[0,1,2,3,4],
    category='baby',
)

model_celular = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    classes=[0,1,2,3,4],
    category='celular',
)

model_celular.train()

model_food = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    classes=[0,1,2,3,4],
    category='food',
)
model_food.train()

model_games = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    classes=[0,1,2,3,4],
    category='games',
)
model_games.train()

model_laptops = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    classes=[0,1,2,3,4],
    category='laptops',
)
model_laptops.train()

model_livros = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    classes=[0,1,2,3,4],
    category='livros',
)
model_livros.train()

model_moda = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    classes=[0,1,2,3,4],
    category='moda',
)
model_moda.train()

model_pets = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    classes=[0,1,2,3,4],
    category='pets',
)
model_pets.train()

model_toys = TrainModel(
    model=clf,
    params=params,
    rq='RQ4',
    classes=[0,1,2,3,4],
    category='toys',
)
model_toys.train()