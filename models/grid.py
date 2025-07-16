from src.load import finalize_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def hypertune_knn():
    X_train, X_test, y_train, y_test = finalize_data()
    param_grid = {
        'weights': ["uniform", "distance"],
        'n_neighbors': [3, 4, 5]
    }

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv = 5, verbose = 3)
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_