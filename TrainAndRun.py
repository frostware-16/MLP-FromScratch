import numpy as np
import time

Epochs = 100000

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoidDot(x):
    exp = np.exp(-x)
    return exp/(1+exp)**2

def relu(x): 
    return np.maximum(0, x)
def relu_grad(x): 
    return (x > 0).astype(float)

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

        self.grad  = np.zeros(self.weights.size + 1)

        self.a     = None
        self.z     = None
        self.delta = 0

    def forward(self,a_prev,ActivationFunction):
        # z = W dot A_prev + bias
        # a = activationfunction(z)
        self.z = np.dot(self.weights , a_prev) + self.bias
        self.a = ActivationFunction(self.z)
    
    def backward(self , y , a_prev, neuronIndex, activationGrad, NextLayer):
        self.delta = 0
        if not NextLayer:
            self.delta = 2 * (self.a - y)

        else:
            for i in range(len(NextLayer.Neurons)):
                neuron = NextLayer.Neurons[i]
                self.delta += neuron.weights[neuronIndex] * activationGrad(neuron.z) * neuron.delta

        
        act_grad = activationGrad(self.z)
        self.grad = a_prev * act_grad * self.delta
        self.grad[-1] = act_grad * self.delta

class Layer:
    def __init__(self,inp_dim,out_dim,activationfunction,activationGrad):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.Neurons = []
        self.delta = None
        self.a_vec = np.zeros(out_dim)

        self.activationfunction = activationfunction
        self.activationGrad = activationGrad

    def Param_Initialization(self):
        for i in range(self.out_dim):
            #Initializating Weights
            #Initializating Biases
            self.Neurons.append( 
                Neuron(
                np.random.randn(self.inp_dim) * np.sqrt(2. / self.inp_dim), 
                np.random.rand(1)[0]) 
                ) 

    def forward(self , a_prev):
        for i in range(len(self.Neurons)):
            self.Neurons[i].forward(a_prev , self.activationfunction)
            self.a_vec[i] = self.Neurons[i].a

    def backward(self, y, a_prev , NextLayer):
        for i in range(self.out_dim):
            self.Neurons[i].backward(y, a_prev, i, self.activationGrad, NextLayer)


class Sequtinal:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, baseCase):
        self.layers[0].forward(baseCase)
        for i in range(1, len(self.layers)):
                self.layers[i].forward(self.layers[i-1].a_vec)

    def backward(self , x ,y):
        #Last layer
        self.layers[-1].backward(y , self.layers[-2].a_vec , None)

        #Middle ones
        for i in reversed(range(1,len(self.layers)-1)):
            self.layers[i].backward(y , self.layers[i-1].a_vec , self.layers[i+1])
        
        #First layer
        self.layers[0].backward(y , x , self.layers[1])

    def update_weights(self,lr):
        for layer in self.layers:
            for neuron in layer.Neurons:
                for i in range(len(neuron.weights)):
                    neuron.weights[i] -= neuron.grad[i]* lr
                neuron.bias -= neuron.grad[-1]* lr

    def clip_grad(self, lb, hb):
        for layer in self.layers:
            for neuron in layer.Neurons:
                np.clip(neuron.grad, lb, hb, out=neuron.grad)

datapoints = [

[np.array([1,1]),0],
[np.array([1,0]),1],
[np.array([0,1]),1],
[np.array([0,0]),0]
                
                ]

layer1 = Layer(2,4,sigmoid,sigmoidDot)
layer1.Param_Initialization()
layer2 = Layer(4,8,sigmoid,sigmoidDot)
layer2.Param_Initialization()
layer3 = Layer(8, 1, sigmoid, sigmoidDot)
layer3.Param_Initialization()

model = Sequtinal([layer1,layer2,layer3])

lr = 0.1
start_time = time.time()


for epoch in range(Epochs):
    for x, y in datapoints:
        model.forward(x)
        model.backward(x, y)
        model.clip_grad(-1,1)
        model.update_weights(lr)

    if epoch % 1000 == 0:
        now = time.time()
        elapsed = now - start_time
        loss = 0
        for x, y in datapoints:
            model.forward(x)
            pred = model.layers[-1].a_vec[0]
            loss += (pred - y) ** 2
        print(f"Epoch {epoch} / Loss: {loss:.4f} / Time Elapsed: {elapsed:.2f} seconds")
        start_time = now  # Reset timer

for x, y in datapoints:
    model.forward(x)
    print(model.layers[-1].a_vec)