import numpy as np
import matplotlib.pyplot as plt

# Generate input data ----------------------------------------
rng = np.random.default_rng()

N = 1000
x_min, xmax = -2, 2
y_min, ymax = -2, 2

x = rng.uniform(x_min, xmax, N)
y = rng.uniform(y_min, ymax, N)

# Create neurons ---------------------------------------------
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def step_function(self, local_field):
        return np.heaviside(local_field, 0)
    
    def forward(self, inputs):
        local_field = self.bias + np.dot(self.weights, inputs)
        return self.step_function(local_field)
    
# Layer 1 neurons
n11 = Neuron(weights=[1, -1], bias=1)
n12 = Neuron(weights=[-1, -1], bias=1)
n13 = Neuron(weights=[-1], bias=0)

# Layer 2 neuron
n2 = Neuron(weights=[1, 1, -1], bias=-1.5)

# Forward pass through the network --------------------------------

# Layer 1
out11 = n11.forward([x, y])
out12 = n12.forward([x, y])
out13 = n13.forward([x])

# Layer 2
z = n2.forward([out11, out12, out13])

# Plotting --------------------------------------------------------

# 0 = blue, 1 = red
plt.figure(figsize=(8, 8))
plt.scatter(x, y, c=z, cmap='bwr', alpha=0.5)
plt.show()