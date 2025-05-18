from code.train_model import TrainModel
from sklearn.svm import SVC as SVM

clf = SVM()

params = { 
    "C": [0.01, 0.1, 1, 10, 100],
    "kernel": ["poly", "rbf"],
    "degree": [3, 4, 5, 6],
    "gamma": ["scale", "auto", 0.01, 0.1, 1],
}

model = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ1',
    using_tfidf=True
)

model.train()




