import os
import urllib
import urllib.request
from zipfile import ZipFile
UTL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

# Download dataset
def download_dataset():
    """
    Download the Fashion-MNIST dataset from the given URL.

    :return: None
    """
    if not os.path.exists(FILE):
        print(f'Downloading {FILE}...')
        urllib.request.urlretrieve(UTL, FILE)


