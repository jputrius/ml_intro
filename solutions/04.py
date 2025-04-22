from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
X, y = make_classification(n_samples=10000, n_features=20, n_informative=10, n_classes=3, n_redundant=10, random_state=34)
y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
print(f"X shape: {X.shape}, y shape: {y.shape}")

import numpy as np
from sklearn.metrics import classification_report

class NeuralNet:
  def __init__(self, n_features=1, n_outputs=1, random_state=34):
    self.rng = np.random.default_rng(seed=random_state)
    self.n_features = n_features
    self.n_outputs = n_outputs
    self.W1 = self.rng.uniform(low=-1, high=1, size=(self.n_features, 2*self.n_features))
    self.b1 = self.rng.uniform(low=-1, high=1, size=2*self.n_features)
    self.W2 = self.rng.uniform(low=-1, high=1, size=(2*self.n_features, self.n_outputs))
    self.b2 = self.rng.uniform(low=-1, high=1, size=self.n_outputs)

  def _relu(self, x):
    return np.maximum(x, 0)

  def _drelu(self, x):
    return np.where(x >= 0, 1, 0)

  def _A(self, x, W, b):
    return x@W+b

  def _output(self, x):
    return self._relu(self._A(self._relu(self._A(x, self.W1, self.b1)), self.W2, self.b2))

  def inference(self, x):
    y = self._output(x)
    return np.argmax(y, axis=1)

  def _loss(self, x, y):
    out = self._softmax(self._output(x))
    return -1*np.mean(np.log(np.vecdot(y, out, axis=1)))

  def _softmax(self, x):
    e = np.exp(x)
    return (e.T/np.sum(e, axis=1)).T
  
  def _dloss(self, x, y):
    out = self._softmax(self._output(x))
    return np.mean(out-y, axis=0)

  def fit(self, x, y, learning_rate=0.01, momentum=0.7, epochs=20):
    self.W1 = self.rng.uniform(low=-1, high=1, size=(self.n_features, 2*self.n_features))
    self.b1 = self.rng.uniform(low=-1, high=1, size=2*self.n_features)
    self.W2 = self.rng.uniform(low=-1, high=1, size=(2*self.n_features, self.n_outputs))
    self.b2 = self.rng.uniform(low=-1, high=1, size=self.n_outputs)

    for epoch in range(epochs):
      delta_W1 = np.zeros(self.W1.shape)
      delta_W2 = np.zeros(self.W2.shape)
      delta_b1 = np.zeros(self.b1.shape)
      delta_b2 = np.zeros(self.b2.shape)

      p = self.rng.permutation(x.shape[0])
      for x_sample, y_sample in zip(x[p], y[p]):
        x1 = np.reshape(x_sample, shape=(1, -1))
        y1 = np.reshape(y_sample, shape=(1, -1))

        # Forward pass
        A1 = self._A(x1, self.W1, self.b1)
        f1 = self._relu(A1)
        A2 = self._A(f1, self.W2, self.b2)
        f2 = self._relu(A2)

        # Backwards pass
        df1 = self._drelu(A1)
        df2 = self._drelu(A2)
        dloss = self._dloss(x1, y1)

        delta_W2 = momentum*delta_W2 - learning_rate*(f1.reshape(-1, 1)@(df2*dloss).reshape(1, -1))
        delta_b2 = (momentum*delta_b2 - learning_rate*df2*dloss).flatten()
        delta_W1 = momentum*delta_W1 - learning_rate*(x1.T@((df2*self.W2*df1.T)@dloss).reshape(1, -1))
        delta_b1 = (momentum*delta_b1 - learning_rate*(df2*self.W2*df1.T)@dloss).flatten()

        self.W2 = self.W2 + delta_W2
        self.W1 = self.W1 + delta_W1
        self.b2 = self.b2 + delta_b2
        self.b1 = self.b1 + delta_b1

      print(f"Epoch {epoch+1}: loss - {self._loss(x, y)}")

model = NeuralNet(n_features=X.shape[1], n_outputs=y.shape[1])

print(classification_report(np.argmax(y, axis=1), model.inference(X)))
model.fit(X, y, learning_rate=0.002, momentum=0.3)
print(classification_report(np.argmax(y, axis=1), model.inference(X)))