from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from numpy import *

# download the dataset
iris_dataset = datasets.load_iris()
iris_data = iris_dataset.data
iris_target = iris_dataset.target

# split data and target into training set and testing set
# 80% training, 20% testing
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size = 0.2)
# construct SVC by using rbf as kernel function
SVC_0 = SVC(kernel='rbf')
SVC_0.fit(x_train, y_train)

predict = SVC_0.predict(x_test)
right = sum(predict == y_test)
# accuracy rate
print("%f%%" % (right * 100.0 / predict.shape[0]))

