from sklearn.datasets import make_classification
X, y = make_classification(n_samples=10000, n_features=20, n_informative=10, n_classes=2, n_redundant=10, random_state=34)
print(f"X shape: {X.shape}, y shape: {y.shape}")

import numpy as np
from sklearn.metrics import classification_report

class LogisticRegression:
  def __init__(self, n_features=1, random_state=34, confidence=0.5):
    self.rng = np.random.default_rng(seed=random_state)
    self.weights = self.rng.standard_normal(n_features+1)
    self.confidence = confidence

  def _f(self, x_w_one):
    return 1/(1+np.exp(np.sum(-x_w_one*self.weights, axis=1)))

  def inference(self, x):
    confs = self._f(np.insert(x, 0, 1., axis=1))
    return np.where(confs > self.confidence, 1, 0)

  def _loss(self, x, y):
    x_w_one = np.insert(x, 0, 1., axis=1)
    return -np.mean(y*np.log(self._f(x_w_one))+(1-y)*np.log(1-self._f(x_w_one)))

  def _gradient(self, x, y):
    x_w_one = np.insert(x, 0, 1., axis=1)
    return -np.mean((x_w_one * (y*self._f(-x_w_one)+(y-1)*self._f(x_w_one))[:, np.newaxis]), axis=0)
  
  def fit(self, x, y, learning_rate=0.01, momentum=0.7, epochs=20):
    self.weights = self.rng.standard_normal(self.weights.shape[0])
    for epoch in range(epochs):
      p = self.rng.permutation(x.shape[0])
      delta_w = np.zeros(self.weights.shape[0])
      for x_sample, y_sample in zip(x[p], y[p]):
        delta_w = momentum*delta_w-learning_rate*self._gradient(np.reshape(x_sample, (1, -1)), np.reshape(y_sample, (1, -1)))
        self.weights = self.weights + delta_w
      print(f"Epoch {epoch+1}: loss - {self._loss(x, y)}")
      
model = LogisticRegression(n_features = X.shape[1])

print(classification_report(y, model.inference(X)))
model.fit(X, y, learning_rate=0.0001, momentum=0.7)
print(classification_report(y, model.inference(X)))