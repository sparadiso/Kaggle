import sys
sys.path.append("../")

from lib.include import *
from sklearn.ensemble import RandomForestClassifier

# Load the training data 
X, y = TrainData()

# Filename of the grid (set of models) to serialize for final submission/visualization
model_name = "model_RandomForests.pkl"

# Build the pipeline
rfs = RandomForestClassifier(n_jobs=1)
pca = decomposition.PCA(n_components=150)
pipe = Pipeline(steps=[('pca', pca), ('normalize', StandardScaler()), ('rfs', rfs)])

with open(model_name, 'wb') as f:
    pickle.dump(pipe, f)

# Set a few model parameters
params = dict(rfs__n_estimators=[10, 50, 150, 250])

grid = GridSearchCV(pipe, params, n_jobs=3, verbose=5, cv=3)
grid.fit(X, y)

best = grid.best_estimator_
print best.named_steps['rfs'].n_estimators

with open(model_name, 'wb') as f:
    pickle.dump(grid, f)

