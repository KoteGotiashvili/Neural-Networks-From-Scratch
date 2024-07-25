import os
import urllib
import urllib.request
from zipfile import ZipFile
class DataPreparation:
    URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
    FILE = 'fashion_mnist_images.zip'
    FOLDER = 'fashion_mnist_images'

    @staticmethod
    def download_dataset():
        """
        Download the Fashion-MNIST dataset from the given URL.

        :return: None
        """
        if not os.path.exists(DataPreparation.FILE):
            print(f'Downloading {DataPreparation.FILE}...')
            urllib.request.urlretrieve(DataPreparation.URL, DataPreparation.FILE)
        else:
            print(f'{DataPreparation.FILE} already exists.')

    @staticmethod
    def unzip_dataset():
        """
        Unzip the downloaded dataset.

        :return: None
        """
        if not os.path.exists(DataPreparation.FOLDER):
            print("Unzipping images...")
            with ZipFile(DataPreparation.FILE, 'r') as zip_images:
                zip_images.extractall(DataPreparation.FOLDER)
            print("Unzipping done.")
        else:
            print(f'{DataPreparation.FOLDER} already exists.')



DataPreparation.download_dataset()
DataPreparation.unzip_dataset()
print("Done")

