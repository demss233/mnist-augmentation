import numpy as np
from src.load import finalize_data
from src.utils import image_pixel_shift

def dummy_data():
    X_train, X_test, y_train, y_test = finalize_data()
    dummy_train_data = [img for img in X_train]
    dummy_train_labels = [label for label in y_train]
    return dummy_train_data, dummy_train_labels