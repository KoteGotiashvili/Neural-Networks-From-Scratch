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

