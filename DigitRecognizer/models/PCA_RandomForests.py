import sys
sys.path.append("../")

from lib.include import *
from sklearn.ensemble import RandomForestClassifier

# Load the training data 
X, y = TrainData()

# Filename of the grid (set of models) to serialize for final submission/visualization
model_name = "model_RandomForests.pkl"

# Build the pipeline
rfs = RandomForestClassifier(n_jobs=2)
pca = decomposition.PCA(n_components=400)
pipe = Pipeline(steps=[('pca', pca), ('normalize', StandardScaler()), ('rfs', rfs)])

# Do a quick grid search for n_estimator convergence
for nest in [15, 50, 100, 200, 350]:
    rfs.n_estimators = nest
    print "NEstimators = {}, score = {}".format(nest, ScoreModel(pipe, X, y))

with open(model_name, 'wb') as f:
    pickle.dump(pipe, f)
