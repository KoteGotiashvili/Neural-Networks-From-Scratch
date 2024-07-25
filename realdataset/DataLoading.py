import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class DataLoading:

    @staticmethod
    def get_labels(dir):
        labels = os.listdir(dir)
        print(labels)

    @staticmethod
    def visualize_data(img_dir):

        img_data = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
        plt.imshow(img_data)
        plt.show()

    # Loads a MNIST dataset

    # Loads a MNIST dataset
    def load_mnist_dataset(self, dataset, path):
        IMG_SIZE = (28, 28)  # Adjust this to the size you need

        # Scan all the directories and create a list of labels
        labels = os.listdir(os.path.join(path, dataset))

        # Create lists for samples and labels
        X = []
        y = []

        # For each label folder
        for label in labels:
            # And for each image in the given folder
            for file in os.listdir(os.path.join(path, dataset, label)):
                # Read the image
                image_path = os.path.join(path, dataset, label, file)
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

                # Check if image is loaded correctly
                if image is not None:
                    # Resize the image to ensure consistent shape
                    image_resized = cv2.resize(image, IMG_SIZE)

                    # Convert image to grayscale if necessary
                    if len(image_resized.shape) == 3:  # If image has multiple channels
                        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

                    # Append image and label to lists
                    X.append(image_resized)
                    y.append(int(label))  # Convert label to integer if it's a string

        # Convert lists to NumPy arrays and ensure uniform shape
        X = np.array(X)
        y = np.array(y)


        return X, y

    # MNIST dataset (train + test)
    def create_data_mnist(self, path):
        # Load both sets separately
        X, y = self.load_mnist_dataset('train', path)
        X_test, y_test = self.load_mnist_dataset('test', path)

        # And return all the data
        return X, y, X_test, y_test

    # Thanks to this function, we can load in our data by doing:
    # Create dataset
