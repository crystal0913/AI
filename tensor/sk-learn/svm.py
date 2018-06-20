from sklearn import datasets, svm

iris = datasets.load_iris()
digits = datasets.load_digits()
clf = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(digits.data[:-1], digits.target[:-1])

print("prediction: ", clf.predict(digits.data[-1:]))
print("actual: ", digits.target[-1:])
