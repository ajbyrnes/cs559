import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_dataset():
    """Load and normalized the iris dataset."""
    iris = pd.read_csv("iris.csv")
    
    iris_mean = iris.mean(numeric_only=True)
    iris_std = iris.std(numeric_only=True)

    iris_normalized = (iris.select_dtypes(include=[np.number]) - iris_mean) / iris_std
    iris_normalized['variety'] = iris['variety']

    return iris_normalized

class CompetitiveNet():
    def __init__(self, X):
        # Select three random samples as initial weight vectors
        np.random.seed(12345)
        self.w = X[np.random.choice(X.shape[0], 3, replace=False)]
        
    def train(self, X, eta, n_epochs, decay_eta=False, decay_rate=0.99):
        for _ in range(n_epochs):
            for x in X:
                # Compute distances from sample to each weight vector
                distances = np.linalg.norm(self.w - x, axis=1)
                
                # Find index of winning neuron (closest weight vector)
                winner_idx = np.argmin(distances)

                # Update the winning weight vector
                self.w[winner_idx] += eta * (x - self.w[winner_idx])
                
                if decay_eta:
                    eta *= decay_rate

    def cluster(self, X):
        indices = []
        for x in X:
            distances = np.linalg.norm(self.w - x, axis=1)
            winner_idx = np.argmin(distances)
            indices.append(winner_idx)
        return indices


def kmeans(k, data, n_iters):
    # Randomly initialize centroids by selecting k random samples from data
    np.random.seed(12345)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    for _ in range(n_iters):
        # Assignment step
        cluster_indices = []
        for x in data:
            distances = np.linalg.norm(centroids - x, axis=1)
            winner_idx = np.argmin(distances)
            cluster_indices.append(winner_idx)
        
        cluster_indices = np.array(cluster_indices)
        
        # Update step
        new_centroids = np.array([data[cluster_indices == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, cluster_indices


def cluster_map(df, true_col, cluster_col, pred_col):
    cluster_mapping = df[true_col].groupby(df[cluster_col]).agg(lambda x: x.value_counts().index[0])
    df[pred_col] = df[cluster_col].map(cluster_mapping)
    return df


def compute_confusion_matrix(df, true_col, pred_col):
    return pd.crosstab(df[true_col], df[pred_col], rownames=['Actual'], colnames=['Predicted'])


def compute_accuracy(df, true_col, pred_col):
    return np.sum(df[true_col] == df[pred_col]) / len(df)


def do_competitive_learning(iris_normalized, X, eta, n_epochs):
    print(f"Competitive Learning: eta={eta}, n_epochs={n_epochs}")
    
    # Train network
    nn = CompetitiveNet(X)
    nn.train(X, eta=eta, n_epochs=n_epochs)
    
    # Assign clusters
    cluster_indices = nn.cluster(X)
    iris_normalized['cluster_index_competitive'] = cluster_indices
    
    # Evaluate clustering
    iris_normalized = cluster_map(iris_normalized, 'variety', 'cluster_index_competitive', 'variety_pred_competitive')
    confusion_matrix_competitive = compute_confusion_matrix(iris_normalized, 'variety', 'variety_pred_competitive')
    accuracy_competitive = compute_accuracy(iris_normalized, 'variety', 'variety_pred_competitive')
    
    print("Confusion Matrix:")
    print(confusion_matrix_competitive)
    print("Accuracy:", accuracy_competitive)
    print("==============================\n")
    
    return accuracy_competitive
    

def do_kmeans(iris_normalized, X, n_iters=100):
    print(f"K-Means Clustering: n_iters={n_iters}")
    
    centroids, cluster_indices_kmeans = kmeans(k=3, data=X, n_iters=n_iters)
    iris_normalized['cluster_index_kmeans'] = cluster_indices_kmeans
    
    # Evaluate clustering
    iris_normalized = cluster_map(iris_normalized, 'variety', 'cluster_index_kmeans', 'variety_pred_kmeans')
    confusion_matrix_kmeans = compute_confusion_matrix(iris_normalized, 'variety', 'variety_pred_kmeans')
    accuracy_kmeans = compute_accuracy(iris_normalized, 'variety', 'variety_pred_kmeans')
    
    print("Confusion Matrix:")
    print(confusion_matrix_kmeans)
    print("Accuracy:", accuracy_kmeans)
    print("==============================\n")
    
    return accuracy_kmeans
    

def competitive_learning_eta_experiment(iris_normalized, X, etas):
    accuracies = []
    for eta in etas:
        accuracy = do_competitive_learning(iris_normalized, X, eta=eta, n_epochs=10)
        accuracies.append((eta, accuracy))

    return accuracies
    

def competitive_learning_epochs_experiment(iris_normalized, X, n_epochs_list, eta=0.1):
    accuracies = []
    for n_epochs in n_epochs_list:
        accuracy = do_competitive_learning(iris_normalized, X, eta=eta, n_epochs=n_epochs)
        accuracies.append((n_epochs, accuracy))

    return accuracies
        
        
def kmeans_iters_experiment(iris_normalized, X, n_iters_list):
    accuracies = []
    for n_iters in n_iters_list:
        accuracy = do_kmeans(iris_normalized, X, n_iters=n_iters)
        accuracies.append((n_iters, accuracy))

    return accuracies
        
        
if __name__ == "__main__":
    # Load data
    iris_normalized = load_dataset()

    # Extract features for clustering
    X = iris_normalized.drop('variety', axis=1).values
    
    # Competitive Learning
    comp_eta_accuracies = competitive_learning_eta_experiment(iris_normalized, X, etas=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    comp_epoch_accuracies = competitive_learning_epochs_experiment(iris_normalized, X, n_epochs_list=[1, 5, 10, 20, 50], eta=0.01)
    
    # K-means
    kmeans_iters_accuracies = kmeans_iters_experiment(iris_normalized, X, n_iters_list=[1, 5, 10, 20, 50, 100])
    
    # Plot eta vs accuracy for Competitive Learning
    plt.figure(figsize=(12, 4))
    etas, comp_accuracies = zip(*comp_eta_accuracies)
    plt.plot(etas, comp_accuracies, marker='o')
    # plt.xscale('log')
    plt.xlabel('Learning Rate (eta)')
    plt.ylabel('Accuracy')
    plt.title('Competitive Learning: Eta vs Accuracy')
    
    # Plot epochs vs accuracy and iters vs accuracy for K-means
    plt.figure(figsize=(12, 4))
    n_epochs, comp_epoch_accuracies_vals = zip(*comp_epoch_accuracies)
    plt.plot(n_epochs, comp_epoch_accuracies_vals, marker='o', label='Competitive Learning')
    n_iters, kmeans_accuracies_vals = zip(*kmeans_iters_accuracies)
    plt.plot(n_iters, kmeans_accuracies_vals, marker='o', label='K-Means')
    plt.xlabel('Number of Epochs/Iterations')
    plt.ylabel('Accuracy')
    plt.title('Number of Epochs/Iterations vs Accuracy')
    plt.legend()
    plt.show()