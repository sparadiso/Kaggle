import sys
sys.path.append("../")

from lib.include import *

# Load the training data
X, y = TrainData()

# Create a simple pipeline that identifies the principle components of the training data (ultimately the eigenvectors of the covariance matrix), then normalizes the bases
pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('normalize', StandardScaler())])

# Fit the pca model
pipe.fit(X, y)

# Compute the explained variance and corresponding cumulative sum of the variance retained with 'i' components.
x = pca.explained_variance_ratio_
cdf = [np.sum(x[:i]) for i in range(len(x))]

# Plot the CDF of the variance to identify reasonable cutoff
import pylab as py
py.figure(figsize=(4,3))

py.plot(cdf)
py.xlabel("PCA N_Components")
py.ylabel("Variance Retained")

py.savefig("pca_variance.png")
