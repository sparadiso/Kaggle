#!/usr/bin/python

# A dump of included packages and a few helper functions
from include import *
import sys

# Pull the pickled model filename (an sklearn Estimator)
model_fname = sys.argv[1]

# Unpickle the model for scoring (the model 
with open(model_fname, 'rb') as f:
    model = pickle.load(f)

# Load the training data from data/train.csv
X, y = TrainData()

# Print the result of kfold cross validation
print ScoreModel(model, X, y)
