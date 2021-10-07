from sklearn import linear_model
from sklearn import datasets
from sklearn.datasets.samples_generator import make_classification
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve
import pickle
from sklearn.externals import joblib

#########################################################################################################
# iris = datasets.load_iris() # 导入数据集
# X = iris.data # 获得其特征向量
# y = iris.target # 获得样本label
# print(X[:2, :])
# print(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# print(y_train)
# knn = KNeighborsClassifier()
# knn.fit(X_train,y_train)
# print(knn.predict(X_test))
# print(y_test)
# boston = datasets.load_boston()
# print(boston.data.shape)
# X = boston.data
# y = boston.target
# model = LinearRegression()
# model.fit(X, y)
# print(model.predict(X[:4, :]))
# print(y[:4])
# print(model.coef_)# y = 0.1x + 0.3 >>>0.1
# print(model.intercept_) #>>>>0.3
# print(model.get_params())
# print(model.score(X, y))# R^2 coefficient of determination
# M, n = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=100)
# plt.scatter(M,n)
# plt.show()
######################################################################################################################
# a = np.array([[10, 2.7, 3.6],[-100, 5, -2], [120, 20, 40]], dtype=np.float64)
# print(a)
# print(preprocessing.scale(a))
# X, y = make_classification(n_samples=300, n_features=2,
#                                     n_redundant=0, n_informative=2,
#                                     random_state=22, n_clusters_per_class=1,
#                                     scale=100)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# #plt.show()
# X = preprocessing.scale(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# clf = SVC()#支持向量机
# clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()
###################################################################################################################
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
# knn = KNeighborsClassifier(n_neighbors=5)
# scores = cross_val_score(knn, X, y, cv=5, scoring="accuracy")
# print(scores.mean())
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# print(knn.score(X_test, y_test))
# k_range = range(1, 31)
# k_scores = []
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X, y, cv=5, scoring="accuracy")
#     k_scores.append(scores.mean())
# # print(k_scores)
# plt.plot(k_range, k_scores)
# plt.xlabel("Value of K for KNN")
# plt.ylabel("Cross-Validated Accuracy")
# plt.show()
#####################################################################################################################
digits = load_digits()
X = digits.data
y = digits.target
train_sizes, train_loss, test_loss = learning_curve(
        SVC(gamma=0.01), X, y, cv=10, scoring=None,
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1])
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
             label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
             label="Cross-validation")

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
######################################################################################################################

digits = load_digits()
X = digits.data
y = digits.target
param_range = np.logspace(-6, -2.3, 5)
train_loss, test_loss = validation_curve(
        SVC(), X, y, param_name='gamma', param_range=param_range, cv=10,
        scoring='accuracy')
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(param_range, train_loss_mean, 'o-', color="r",
             label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color="g",
             label="Cross-validation")

plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
#################################################################################################################

# clf = SVC()
# iris = datasets.load_iris()
# X, y = iris.data, iris.target
# clf.fit(X, y)
#
# with open('save/clf.pickle', 'wb') as f:
#     pickle.dump(clf, f)
# restore
# with open('save/clf.pickle', 'rb') as f:
#     clf2 = pickle.load(f)
#     print(clf2.predict(X[0:1]))

# method 2: joblib
#
# Save
# joblib.dump(clf, 'save/clf.pkl')
# restore
# clf3 = joblib.load('save/clf.pkl')
# print(clf3.predict(X[0:1]))

##################################################################################################################
















