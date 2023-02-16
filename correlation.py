"""
Plots the correlation between the features. 
"""
import numpy as np
import openml
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pylab import rcParams

#credit-g data set 
dataset = openml.datasets.get_dataset(31)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)

def correlation_plot(data):
    # init figure size
    rcParams['figure.figsize'] = 15, 10
    fig = plt.figure()
    sns.heatmap(data.corr(), annot=True, fmt=".2f")
    plt.show()
    fig.savefig('correlation.png')

#creates a data frame with attribute_names as the row index    
df=pd.DataFrame(X,columns=attribute_names)
correlation_plot(df)