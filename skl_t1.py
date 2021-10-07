# from sklearn import tree
#
#
# X = [[0, 0], [2, 2]]
# y = [0.5, 2.5]
# clf = tree.DecisionTreeRegressor()
# clf = clf.fit(X, y)
# arry = clf.predict([[1, 1]])
# print(arry)

from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

dataset = datasets.load_iris()

model = ExtraTreesClassifier()
model.fit(dataset.data, dataset.target)

#print(model.feature_importances_)
print(dataset)