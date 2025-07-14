import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class layer:
    def __init__(self,inp_dim , out_dim):
        self.inp_dim = inp_dim
        self.out_dim = out_dim 
        self.weights = np.zeros((out_dim, inp_dim))
        self.biases = np.zeros((out_dim,1))

        self.z = 0
        self.a = 0
        self.delta = np.zeros((out_dim,1))
        self.grad = np.zeros((out_dim, inp_dim))

    def random_param_Intilzation(self):
        self.weights = np.random.randn(self.out_dim, self.inp_dim)
        self.biases = np.random.randn(self.out_dim,1)

    
    def forward(self , x):
        self.z = self.weights @ x + self.biases
        self.a = sigmoid(self.z)

    
class sequential:
    def __init__(self,layers):
        self.layers = layers
    
    def forward(self,x):
        self.layers[0].forward(x)
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i-1].a)

    def backward(self,x,y):
        #Base Case
        lastLayer = self.layers[-1]
        lastLayer.delta = lastLayer.a - y 

        for i in reversed(range(len(self.layers)-1)):
            layer = self.layers[i] 
            nextLayer = self.layers[i+1]
            layer.delta = (nextLayer.weights.T @ nextLayer.delta) * sigmoid_derivative(layer.z)
            if i == 0:
                a_prev = x
            else:
                a_prev = self.layers[i-1].a
            layer.grad = layer.delta @ a_prev.T

    def step(self, lr=0.1):
        for layer in self.layers:
            layer.weights -= lr * layer.grad
            layer.biases -= lr * layer.delta

    def predict(self, x):
        self.forward(x)
        return self.layers[-1].a
X = [
    np.array([[0], [0]]),
    np.array([[0], [1]]),
    np.array([[1], [0]]),
    np.array([[1], [1]])
]

Y = [
    np.array([[0]]),
    np.array([[1]]),
    np.array([[1]]),
    np.array([[0]])
]

layer1 = layer(2, 1000)
layer2 = layer(1000, 1)
model = sequential([layer1, layer2])

for layer in model.layers:
    layer.random_param_Intilzation()

epochs = 3000
lr = 0.1

for epoch in range(epochs):
    total_loss = 0
    for x, y in zip(X, Y):
        model.forward(x)
        model.backward(x, y)
        model.step(lr)

        total_loss += np.square(model.layers[-1].a - y).sum()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")


print("\nXOR Predictions:")
for x, y in zip(X, Y):
    pred = model.predict(x)
    print(f"Input: {x.ravel()}, Predicted: {pred.item():.4f}, Target: {y.item()}")