import cPickle as pickle
import sklearn as sk
import numpy as np
import pylab as py
import pandas as pd

import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold, cross_val_score
from sklearn import linear_model, decomposition
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

# Load data into Pandas dataframe
def TrainData():
    fname = "../data/train.csv"
    data = pd.read_csv(fname, delimiter=',')
    data.columns = ['y'] + range(784)
    return np.array(data[range(784)]), np.array(data['y'])

def TestData():
    fname = "/home/sean/Dropbox/DataScience/Kaggle/DigitRecognizer/data/test.csv"
    data = pd.read_csv(fname, delimiter=',')
    data.columns = range(784)
    return pd.DataFrame(data)

# Helper function to Score an sklearn estimator (`model`) 
# using kfold cross validation
def ScoreModel(model, X, y):
    # Split the data into folds
    N_Data = len(y)
    folds = KFold(N_Data, 3)

    accuracy = []

    print "Starting cross validation"; i=0
    # Loop over folds and train/test the model
    for train, test in folds:
        print "{}/{}".format(i+1, 3); i+=1
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        print "Training..."
        model.fit(X_train, y_train)
        print "Done."
        y_pred = model.predict(X_test)

        accuracy.append(model.score(X_test, y_test))
        print sklearn.metrics.confusion_matrix(y_test, y_pred)

    return np.mean(accuracy), np.std(accuracy) / np.sqrt(len(folds)-1.0)
