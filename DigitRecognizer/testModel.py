#!/usr/bin/python

# A dump of included packages and a few helper functions
from lib.include import *
import sys

# Pull the pickled model filename (an sklearn Estimator)
model_fname = sys.argv[1]

# Unpickle the model for scoring (the model 
with open(model_fname, 'rb') as f:
    model = pickle.load(f)

# Load the training data from data/train.csv
X, y = TrainData("./data")

# Print the result of kfold cross validation
score, err = ScoreModel(model, X, y)

# Now compute the predicted result with the test data
model.fit(X, y)
X_test = TestData("./data")
y_test = model.predict(X_test)
df = pd.DataFrame(dict(ImageID=range(1,len(y_test)+1), Label=y_test))

# Save the predictions for submission
df.to_csv("{:.5f}_svm_rbf.csv".format(score), ',', index=False)
