import sys
sys.path.append("../")

from lib.include import *

# Load the training data 
X, y = TrainData()

# Filename of the grid (set of models) to serialize for final submission/visualization
model_name = "model_SVM_lin.pkl"

# Build the pipeline
svm = sk.svm.SVC(kernel='linear', C=10)
pca = decomposition.PCA(n_components=400)
pipe = Pipeline(steps=[('pca', pca), ('normalize', StandardScaler()), ('svm', svm)])

with open(model_name, 'wb') as f:
    pickle.dump(pipe, f)

# Set a few model parameters
Cs = [.1, 1, 10, 100]
Gammas = [.00001, .0001, .001, 0]# np.linspace(0.01, 3, 5)

grid = GridSearchCV(pipe, dict(
                svm__C=Cs,
                svm__gamma=Gammas), 
            n_jobs=2, verbose=5, cv=2)

# Only use half the data
grid.fit(X[:len(X)/2], y[:len(X)/2])

best = grid.best_estimator_
print best.named_steps['svm'].C
print best.named_steps['svm'].gamma

with open(model_name, 'wb') as f:
    pickle.dump(grid.best_estimator_, f)
