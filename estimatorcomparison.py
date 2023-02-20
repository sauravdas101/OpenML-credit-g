"""
We compare the performace of 4 different estimators. The metric we use to compare is ROC AUC. 
"""
import numpy as np
import openml
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn import datasets, metrics, model_selection


names = [
    "Linear SVM",
    "Random Forest",
    "AdaBoost",
    "Naive Bayes",
]

classifiers = [
    SVC(kernel="linear", C=0.025),
    RandomForestClassifier(n_estimators = 501),
    AdaBoostClassifier(n_estimators=80, random_state=0),
    GaussianNB(),
]

dataset = openml.datasets.get_dataset(31)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)
#Split the dataset into training and testing set with 20% in the testing pool
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 8-10 fetures works the best, from experimentation
sfm = SelectFromModel(RandomForestClassifier(n_estimators = 100), max_features=9, threshold=-np.inf)

for name, clf in zip(names, classifiers):
    ax = plt.subplot(2, 2, classifiers.index(clf) + 1)
    pipe = Pipeline([('reduce_dim', sfm), ('clf', clf)])
    clf1 = pipe.fit(X_train, y_train)
    metrics.RocCurveDisplay.from_estimator(clf1, X_test, y_test, ax=ax,
    name = name)
    
plt.suptitle("Estimator performance comparison for different estimators")
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
_ = plt.axis("tight")
plt.show()  