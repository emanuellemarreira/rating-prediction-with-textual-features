from code.train_model import TrainModel
from sklearn.svm import SVC as SVM

clf = SVM()

params = {
  'C': [0.01, 0.1],
  'kernel': ['linear', 'rbf'],
  'gamma': ['scale', 'auto', 0.01, 0.1]
}

model = TrainModel(
    model = clf,
    params = params,
    rq = 'RQ1',
    using_tfidf=True
)

model.train()




