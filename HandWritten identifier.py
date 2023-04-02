import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Load the dataset of handwritten digits
digits = load_digits()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Create a neural network classifier
clf = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=1000)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Evaluate the classifier on the testing data
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Plot some example digits and their predicted labels
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Predicted: {clf.predict(digits.data[i].reshape(1, -1))[0]}")
plt.show()
