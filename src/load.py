import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

TEST_RATIO = 0.2
SEED = 42

mnist = fetch_openml("mnist_784", version = 1, as_frame = False)

X = mnist.data
y = mnist.target.astype("int")  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_RATIO, random_state = SEED)

def finalize_data():
    return X_train, X_test, y_train, y_test