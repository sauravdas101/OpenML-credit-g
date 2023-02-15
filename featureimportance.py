import numpy as np
import openml
import matplotlib.pyplot as plt
from sklearn import ensemble, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pandas as pd
from sklearn.inspection import permutation_importance

#credit-g data set 
dataset = openml.datasets.get_dataset(31)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)
#Split the dataset into training and testing set with 20% in the testing pool
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#feature imporatance based on mean impunity decrease
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
forest_importances = pd.Series(importances, index=attribute_names).sort_values(ascending=False)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()

#feature imporatance based on feature permutaion
result = permutation_importance(
    forest, X_test, y_test, n_repeats=30, random_state=0, n_jobs=3)
forest_importances = pd.Series(result.importances_mean, index=attribute_names).sort_values(ascending=False)
fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()