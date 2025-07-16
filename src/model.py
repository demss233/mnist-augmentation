import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
from src.augmentation import augmented_data
from models.grid import hypertune_knn

print("Data has been loaded, augmentation started")
augmented_train_data, augmented_train_labels = augmented_data()

shuffle_index = np.random.permutation(len(augmented_train_data))
augmented_train_data = augmented_train_data[shuffle_index]
augmented_train_labels = augmented_train_labels[shuffle_index]
print("Augmentation ended (shuffled for randomness).")

model_path = 'models/knn_model.pkl'

if os.path.exists(model_path):
    print("Loading existing model")
    knn_classifier = joblib.load(model_path)
else:
    print("Training new model via grid search")
    knn_params = hypertune_knn()
    knn_classifier = KNeighborsClassifier(**knn_params)
    knn_classifier.fit(augmented_train_data, augmented_train_labels)
    joblib.dump(knn_classifier, model_path)
    