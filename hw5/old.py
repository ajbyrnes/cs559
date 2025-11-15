import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def split_dataset(data_dir, shape_classes):
    # Get alphabetized list of filenames
    filenames = sorted(os.listdir(data_dir))

    # Split filenames by class
    # Each class has 10,000 samples
    # Use first 8000 samples of each class for training
    # Last 2000 samples for testing
    training_files = []
    test_files = []

    for i in range(len(shape_classes)):
        start_index = i * 10000
        end_index = (i + 1) * 10000

        class_filenames = filenames[start_index:end_index]
        training_files.extend(class_filenames[:8000])
        test_files.extend(class_filenames[8000:])

    # Count samples per class in training and test sets to validate
    train_counter = Counter()
    for filename in training_files:
        shape_class = filename.split("_")[0]
        train_counter[shape_class] += 1
        
    assert all(count == 8000 for count in train_counter.values()), "Training set class distribution is incorrect."

    test_counter = Counter()
    for filename in test_files:
        shape_class = filename.split("_")[0]
        test_counter[shape_class] += 1
        
    assert all(count == 2000 for count in test_counter.values()), "Test set class distribution is incorrect."

    return training_files, test_files

def make_imagefolder_dataset(data_dir, classes):
    

def DataLoader():
    pass


# Define tensor shapes
# num_train_samples = len(training_files)
# num_test_samples = len(test_files)
# image_height, image_width, image_channels = 200, 200, 3

# X_train_shape = (num_train_samples, image_height, image_width, image_channels)
# y_train_shape = (num_train_samples,)
# X_test_shape = (num_test_samples, image_height, image_width, image_channels)
# y_test_shape = (num_test_samples,)

# Memmap training data
# X_train = np.memmap("X_train.dat", dtype=np.uint8, mode="w+", shape=X_train_shape)
# y_train = np.memmap("y_train.dat", dtype=np.uint8, mode="w+", shape=y_train_shape)

# print(f"Memmapping {num_train_samples} training samples...")

# for i, filename in enumerate(training_files):
#     shape_class = filename.split("_")[0]
#     class_index = shape_classes.index(shape_class)
    
#     image_path = os.path.join(data_dir, filename)
#     image = plt.imread(image_path)
    
#     X_train[i] = image
#     y_train[i] = class_index
    
#     if (i + 1) % 1000 == 0:
#         print(f"  Wrote sample {i + 1}/{num_train_samples}")
    
# print("Finished writing training data.")

# del X_train, y_train

# # Memmap test data
# X_test = np.memmap("X_test.dat", dtype=np.uint8, mode="w+", shape=X_test_shape)
# y_test = np.memmap("y_test.dat", dtype=np.uint8, mode="w+", shape=y_test_shape)

# print(f"Memmapping {num_test_samples} test samples...")

# for i, filename in enumerate(test_files):
#     shape_class = filename.split("_")[0]
#     class_index = shape_classes.index(shape_class)

#     image_path = os.path.join(data_dir, filename)
#     image = plt.imread(image_path)

#     X_test[i] = image
#     y_test[i] = class_index
    
#     if (i + 1) % 1000 == 0:
#         print(f"  Processing sample {i + 1}/{num_test_samples}")

# print("Finished writing test data.")

# del X_test, y_test

if __name__ == "__main__":
    shape_classes = [
        "Circle", "Square", "Octagon", "Heptagon", "Nonagon",
        "Star", "Hexagon", "Pentagon", "Triangle"
    ]
    shape_classes = sorted(shape_classes)
        
    raw_data_dir = "geometry_dataset"
    if not os.path.exists(raw_data_dir):
        print(f"Error: Raw data directory '{raw_data_dir}' does not exist.")
    
    training_files, test_files = split_dataset(raw_data_dir, shape_classes)
    
    