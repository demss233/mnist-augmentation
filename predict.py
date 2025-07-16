import joblib 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.load import finalize_data

model_path = 'models/knn_model.pkl'
knn_classifier = joblib.load(model_path)

X_train, X_test, y_train, y_test = finalize_data()
predictions = knn_classifier.predict(X_test)

print("Sample Predictions:", predictions[:10])
print("Accuracy on the test set:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

conf_mat = confusion_matrix(y_test, predictions)
plt.figure(figsize = (10, 8))
sns.heatmap(conf_mat, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = range(10), yticklabels = range(10))

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

X_test_images = X_test.reshape(-1, 28, 28)
plt.figure(figsize = (10, 10))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_test_images[i], cmap='gray')
    plt.title(f"T:{y_test[i]} | P:{predictions[i]}", fontsize = 8)
    plt.axis("off")

plt.tight_layout()
plt.savefig("sample_predictions_grid.png")
plt.show()

pred = pd.DataFrame({
    'Actual': y_test, 
    'Predicted': predictions,
})

pred.to_csv('predictions.csv', index = False)
print("predictions.csv saved")
