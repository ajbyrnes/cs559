import numpy as np

class mnist_loader:    
    def __init__(self):
        pass
            
    def load_labels(self, filename):
        # MNIST labels are stored as a 1D array of bytes.
        
        with open(filename, 'rb') as f:
            print(f"Reading data from {filename}...")
            
            # First 4 bytes = "magic number", which we don't need
            # Next 4 bytes = Number of labels, which we also don't need, because the rest of the file is just 1-byte integers
            f.seek(8)
            
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            print(f"Read {len(labels)} labels.")
    
        return labels
    
    def load_images(self, filename):
        # MNIST images are stored as a 2D array of bytes, with each row representing an image.
        
        with open(filename, 'rb') as f:
            print(f"Reading data from {filename}...")
            
            # First 4 bytes = "magic number", which we can skip
            # Next 4 bytes = Number of images
            # Next 4 bytes = Number of rows
            # Next 4 bytes = Number of columns
            _, num_images, num_rows, num_cols = np.frombuffer(f.read(16), dtype='>i4')
            
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows * num_cols)
            print(f"Read {images.shape[0]} images with shape {num_rows}x{num_cols}.")
            
        return images
    
def load_training_data():
    loader = mnist_loader()
    train_labels = loader.load_labels('mnist/train-labels.idx1-ubyte')
    train_images = loader.load_images('mnist/train-images.idx3-ubyte')

    return train_labels, train_images

def load_test_data():
    loader = mnist_loader()
    test_labels = loader.load_labels('mnist/t10k-labels.idx1-ubyte')
    test_images = loader.load_images('mnist/t10k-images.idx3-ubyte')

    return test_labels, test_images

class NeuralNetwork:
    def __init__(self, weights_shape):
        self.weights = np.zeros(shape=weights_shape)
        self.step_function = lambda x: np.heaviside(x, 1)

    def multicategory_predict(self, inputs_X):
        local_fields = self.weights @ inputs_X
        outputs_Y = np.argmax(local_fields)
        return outputs_Y

    def multicategory_update(self, inputs_X, labels_d, learning_rate):
        for input_x, label_d in zip(inputs_X, labels_d):
            output_y = self.step_function(self.weights @ input_x)
            
            # label_d - output_y has shape (10,)
            # input_x has shape (784,)
            # Use np.newaxis to align dimensions for broadcasting
            self.weights += learning_rate * (label_d - output_y)[:, np.newaxis] * input_x[np.newaxis, :]
        
    def multicategory_pta(self, inputs_X, labels_d, learning_rate=0.01, epsilon=1e-5, seed=None):
        # Set up rng
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        # Randomly initialize weights
        self.weights = rng.uniform(-0.01, 0.01, size=self.weights.shape)
        
        # Report initial accuracy
        num_epochs = 0
        errors_per_epoch = []
        
        accuracy, errors = self.multicategory_accuracy(inputs_X, labels_d)
        errors_per_epoch.append(errors)
        print(f"Epoch {num_epochs}:\t{errors} errors on {len(labels_d)} samples, accuracy {accuracy * 100:.4f}%")

        # Convert scalar labels to one-hot encoded vectors
        label_vectors_d = np.zeros((len(labels_d), self.weights.shape[0]))
        label_vectors_d[np.arange(len(labels_d)), labels_d] = 1

        # Perform training until accuracy threshold is met
        while (errors / len(labels_d)) > epsilon:
            num_epochs += 1
            
            # Iterate over examples and update weights
            self.multicategory_update(inputs_X, label_vectors_d, learning_rate)
            
            # Report accuracy at end of epoch
            accuracy, errors = self.multicategory_accuracy(inputs_X, labels_d)
            errors_per_epoch.append(errors)
            print(f"Epoch {num_epochs}:\t{errors} errors on {len(labels_d)} samples, accuracy {accuracy * 100:.4f}%")
        
        return errors_per_epoch
    
    def multicategory_test(self, inputs_X, labels_d):
        accuracy, errors = self.multicategory_accuracy(inputs_X, labels_d)
        print(f"Test set:\t{errors} errors on {len(labels_d)} samples, accuracy {accuracy * 100:.4f}%")
        return accuracy, errors
        
    def multicategory_accuracy(self, inputs_X, labels_d):
        outputs_Y = [self.multicategory_predict(inputs) for inputs in inputs_X]
        errors = sum(output_y != label_d for output_y, label_d in zip(outputs_Y, labels_d))
        accuracy = (len(labels_d) - errors) / len(labels_d)
        
        return accuracy, errors
    
    
if __name__ == "__main__":
    seed = 1234

    # Load data
    train_labels, train_inputs = load_training_data()    
    test_labels, test_inputs = load_test_data()

    # Create neural network
    nn = NeuralNetwork((10, 784))
    
    def run_experiment(N, eta, epsilon, seed=None):
        print(f"\nRunning experiment with N={N}, eta={eta}, epsilon={epsilon}, seed={seed}")
        training_errors = nn.multicategory_pta(train_inputs[:N], train_labels[:N], learning_rate=eta, epsilon=epsilon, seed=seed)
        test_accuracy, test_errors = nn.multicategory_test(test_inputs, test_labels)
        return training_errors, test_accuracy, test_errors

    def plot_training_errors(training_errors, N, eta, epsilon):
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.plot(training_errors, marker='o')
        plt.title(f'Training Errors vs Epochs (N={N}, eta={eta}, epsilon={epsilon})')
        plt.xlabel('Epoch')
        plt.ylabel('Number of Errors')
        plt.grid(True)
        plt.show()
        
    # Test 1 (Step (e))
    training_errors, test_accuracy, test_errors = run_experiment(N=50, eta=1, epsilon=0, seed=seed)
    plot_training_errors(training_errors, N=50, eta=1, epsilon=0)
    
    # Test 2 (Step (f))
    training_errors, test_accuracy, test_errors = run_experiment(N=1000, eta=1, epsilon=0, seed=seed)
    plot_training_errors(training_errors, N=1000, eta=1, epsilon=0)
    
    # Test 3 (Step (g))
    # training_errors, test_accuracy, test_errors = run_experiment(N=60000, eta=1, epsilon=0, seed=seed)
    
    # Test 4 (Step (h))
    for seed in [1234, 5678, 91011]:
        training_errors, test_accuracy, test_errors = run_experiment(N=60000, eta=1, epsilon=0.15, seed=seed)
        plot_training_errors(training_errors, N=60000, eta=1, epsilon=0.15)