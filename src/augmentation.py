import numpy as np
from middle import dummy_data
from src.utils import image_pixel_shift
from src.load import finalize_data

def augmented_data():
    X_train, X_test, y_train, y_test = finalize_data()
    augmented_train_data, augmented_train_labels = dummy_data()

    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for img, label in zip(X_train, y_train):
            augmented_train_data.append(image_pixel_shift(img, dx, dy))
            augmented_train_labels.append(label)
        
    augmented_train_data = np.array(augmented_train_data)
    augmented_train_labels = np.array(augmented_train_labels)
    return augmented_train_data, augmented_train_labels