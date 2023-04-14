import numpy as np

class LinearLayer:
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.w = np.random.normal(loc=0.0, scale=0.01, size=(n_out, n_in))
        self.b = np.random.normal(loc=0.0, scale=0.01, size=(n_out,))

    def forward(self, x):
        self.x = x
        self.y = self.w @ x + self.b
        return self.y

    def backward(self, dy):
        dx = self.w.T @ dy
        dw = dy.reshape(-1, 1) @ self.x.reshape(1, -1)
        db = dy
        self.w -= dw
        self.b -= db
        return dx

layer = LinearLayer(2, 3)
x = np.array([1, 2])
y = layer.forward(x)
print(y)
dy = np.array([0.1, 0.2, 0.3])
dx = layer.backward(dy)
print(dx)