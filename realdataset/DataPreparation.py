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
def unzip_dataset():
    print("unzipping images...")
    with ZipFile(FILE, 'r') as zip_images:
        zip_images.extractall(FOLDER)

#download_dataset()
#unzip_dataset()
#print("done")

