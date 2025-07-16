# Description
A machine learning pipeline to classify handwritten digits using the MNIST dataset. 
Includes data augmentation, hyperparameter tuning via GridSearchCV, model saving & loading, classification reports and prediction export.

All code modularized with clean structure for easy extension (CNNs, deployment, etc.).

## How to run?
### Create virtual environment
<pre>python -m venv venv
.\venv\Scripts\activate</pre>

### Install dependencies
<pre>pip install -r requirements.txt</pre>

### Train the model
<pre>python train.py</pre>

### Run predictions
<pre>python predict.py</pre>

### Sample Predictions

<img width="951" height="960" alt="Screenshot 2025-07-17 025124" src = "https://github.com/user-attachments/assets/bb1e3cd0-fd87-40d6-9eb7-b5a341917dc4" />
<code>Visualization of predicted digits on the test set. Model accuracy: 97.13%</code>

### Confustion Matrix

<img width="995" height="793" alt="Screenshot 2025-07-17 025140" src="https://github.com/user-attachments/assets/e02306df-0751-42c3-b56e-efbf7e339249" />
<code> Also prints a classification report with accuracy_score, precision & recall.</code>
