import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, rng, learning_rate=0.01):
        self.hidden_size = 24
        self.learning_rate = learning_rate

        # Initialize parameters
        self.hidden_weights = rng.standard_normal(self.hidden_size)
        self.hidden_biases = rng.standard_normal(self.hidden_size)
        self.output_weights = rng.standard_normal(self.hidden_size)
        self.output_bias = rng.standard_normal(1)

    def forward_all(self, X):
        """Forward pass for all samples."""
        self.hidden_field = X[:, np.newaxis] * self.hidden_weights + self.hidden_biases
        self.hidden_output = np.tanh(self.hidden_field)
        self.output = np.dot(self.hidden_output, self.output_weights) + self.output_bias
        return self.output.flatten()

    def forward_single(self, x):
        """Forward pass for a single sample."""
        self.hidden_field = x * self.hidden_weights + self.hidden_biases
        self.hidden_output = np.tanh(self.hidden_field)
        self.output = np.dot(self.hidden_output, self.output_weights) + self.output_bias
        return self.output

    def backprop(self, x, d):
        """One-sample online backpropagation update."""
        y = self.forward_single(x)
        error = d - y

        # Gradients
        tanh_derivative = 1 - self.hidden_output ** 2
        dE_dhidden_weights = error * self.output_weights.T * tanh_derivative * x
        dE_dhidden_biases = error * self.output_weights.T * tanh_derivative
        dE_doutput_weights = error * self.hidden_output
        dE_doutput_bias = error

        # Weight updates
        self.hidden_weights += self.learning_rate * dE_dhidden_weights
        self.hidden_biases += self.learning_rate * dE_dhidden_biases
        self.output_weights += self.learning_rate * dE_doutput_weights
        self.output_bias += self.learning_rate * dE_doutput_bias
    
        return float(error ** 2)

    def gradient_descent(self, X, D, mse_threshold=1e-6, epoch_limit=10000):
        """Train with online gradient descent."""
        mse_history = []
        
        mse = float('inf')
        epoch = 0
        while mse > mse_threshold and epoch < epoch_limit:
            mse = 0
            for x, d in zip(X, D):
                mse += self.backprop(np.array([x]), np.array([d]))
            mse /= len(X)

            # Adaptive learning rate
            if epoch > 0 and mse >= mse_history[-1]:
                    self.learning_rate *= 0.9
            
            # Logging    
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epoch_limit} - MSE: {mse:.6f} - LR: {self.learning_rate:.5f}")
            
            epoch += 1
            mse_history.append(mse)


        return mse_history

    
if __name__ == "__main__":
    # Set up rng
    rng = np.random.default_rng(1234)
    
    # Generate data
    n = 300
    x = rng.uniform(0, 1, n)
    nu = rng.uniform(-0.1, 0.1, n)
    d = np.sin(20 * x) + 3 * x + nu
    
    # Train network with backpropagation
    nn = NeuralNetwork(rng, learning_rate=0.01)
    mse_history = nn.gradient_descent(x, d, mse_threshold=1e-6, epoch_limit=10000)
    
    # Plot MSE vs epochs
    plt.figure()
    plt.plot(mse_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training MSE vs Epochs")
    plt.grid(True)
    plt.show()
    
    # Plot fit curve
    x_sorted = np.sort(x)
    y_fit = np.array([nn.forward_single(xi) for xi in x_sorted])
    
    plt.figure()
    plt.scatter(x, d, s=10, label="Data")
    plt.plot(x_sorted, y_fit, color='red', linewidth=2, label="Model")
    plt.xlabel("x")
    plt.ylabel("d(x)")
    plt.legend()
    plt.title(f"Model fit with lr={nn.learning_rate} after {len(mse_history)} epochs")
    plt.show()