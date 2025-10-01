import numpy as np
import matplotlib.pyplot as plt

def generate_training_data(num_samples, seed=None):
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed=seed)
    
    # Generate random bias and weights
    w0 = rng.uniform(low=-0.25, high=0.25)
    w1, w2 = rng.uniform(low=-1, high=1, size=2)
    
    # Generate inputs
    S = rng.uniform(low=-1, high=1, size=(num_samples,2))
    
    # Generate labels
    # True: v >= 0, sample input is in S1
    # False: v < 0, sample input is in S0
    desired_output = (np.dot(np.insert(S, 0, 1, axis=1), [w0, w1, w2]) >= 0)
    
    return S, (w0, w1, w2), desired_output

def plot_data(S, weights, desired_output, title, save=True):
    S0 = S[desired_output]
    S1 = S[~desired_output]
    
    plt.figure()
    
    # Plot S0
    plt.scatter(
        S0[:, 0], S0[:, 1], 
        color='tab:red', marker='x',
        label='S0'
    )

    # Plot S1
    plt.scatter(
        S1[:, 0], S1[:, 1], 
        color='tab:blue', marker='P',
        label='S1'
    )

    # Plot line w0 + w1x1 + w2x2 = 0
    # x2 = (-w0 - w1x1) / w2
    x1 = np.linspace(-1, 1, len(S))
    x2 = (-weights[0] - weights[1] * x1) / weights[2]
    
    plt.plot(
        x1, x2,
        color='black',
        label='boundary'
    )

    plt.legend(loc='upper right')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(title)
        
    plt.show()
    # plt.savefig(f"{title}.png", dpi=300)
    
def pta(S, desired_output, eta, verbose=False, seed=None):
    num_samples = len(S)
    epoch = 0
        
    # Randomly initialize weights (and bias)
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed=seed)
    
    weights = rng.uniform(low=-1, high=1, size=3)
    
    # Calculate number of correct classifications
    misclassifications = []
    output = (np.dot(np.insert(S, 0, 1, axis=1), weights) >= 0)
    num_correct = np.sum(~(desired_output ^ output))

    # Record number of misclassifications for plotting
    misclassifications.append(num_samples - num_correct)

    if verbose:
        print(f"Epoch {epoch}, {eta=}")
        print(f"\t{weights=}")
        print(f"\t{num_correct} of {num_samples} correct") 
    
    while(num_correct != num_samples):
        # Start next epoch
        epoch += 1
        
        # Calculate new outputs
        output = (np.dot(np.insert(S, 0, 1, axis=1), weights) >= 0)
                
        # Update weights according to type of misprediction
        for i in range(num_samples):
            if desired_output[i] and not output[i]:
                weights = weights + eta * np.insert(S[i], 0, 1)
            elif not desired_output[i] and output[i]:
                weights = weights - eta * np.insert(S[i], 0, 1)
            else:
                pass

        # Calculate new outputs
        output = (np.dot(np.insert(S, 0, 1, axis=1), weights) >= 0)
        num_correct = np.sum(~(desired_output ^ output))
        misclassifications.append(num_samples - num_correct)
        
        # Perform updates for each incorrect 
        if verbose:
            print(f"Epoch {epoch}, {eta=}")
            print(f"\t{weights=}")
            print(f"\t{num_correct} of {num_samples} correct") 
        
    return misclassifications

if __name__ == "__main__":
    seed=12345

    # Initialize misclassifications dictionary
    misclassifications = {num_samples: {eta: 0 for eta in [0.1, 1, 10]} for num_samples in [100, 1000]}

    # Generate training data and train for different values of N and eta
    for num_samples in [100, 1000]:
        S, weights, desired_output = generate_training_data(num_samples=num_samples, seed=seed)
        
        if num_samples == 100:
            print(f"Original weights: {weights}")
        
        plot_data(S, weights, desired_output, title=f"Decision Boundary from Original Weights, N={num_samples}")
        
        for eta in [0.1, 1, 10]:
            verbose = (num_samples == 100) and (eta == 1)
            misclassifications[num_samples][eta] = pta(S, desired_output, eta=eta, verbose=verbose, seed=seed)
            
            
    # Plot misclassifications vs epochs
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    
    fig.suptitle("Misclassifications vs Epochs for Different N and Eta")

    for i, num_samples in enumerate([100, 1000]):
        for j, eta in enumerate([0.1, 1, 10]):
            axs[i, j].plot(
                np.arange(len(misclassifications[num_samples][eta])), 
                (np.array(misclassifications[num_samples][eta]) / num_samples) * 100,
                label=f"Epochs={len(misclassifications[num_samples][eta]) - 1}"
            )
            axs[i, j].legend(loc='upper right')
            axs[i, j].set_title(f"N={num_samples}, eta={eta}")
            axs[i, j].set_xlabel("Epoch")
            axs[i, j].set_ylabel("Misclassifications (%)")
            axs[i, j].grid()

    plt.show()
    # plt.savefig("epochs_vs_misclassifications.png", dpi=300)