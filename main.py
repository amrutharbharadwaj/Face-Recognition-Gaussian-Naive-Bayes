
#### `src/face_recognition.py`
Here’s your complete script file. You can place your original code here, with some added structure for better organization:
```python
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_data():
    """Load the Olivetti Faces dataset."""
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = data.data
    y = data.target
    return X, y

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train Gaussian Naive Bayes and evaluate performance."""
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return gnb, y_pred

def plot_results(X_test, y_test, y_pred):
    """Visualize the predictions."""
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))
    for ax, image, label, prediction in zip(axes.ravel(), X_test, y_test, y_pred):
        ax.imshow(image.reshape(64, 64), cmap=plt.cm.gray)
        ax.set_title(f"True: {label}, Pred: {prediction}")
        ax.axis('off')

    plt.show()

def cross_validate(X, y):
    """Perform cross-validation."""
    gnb = GaussianNB()
    cross_val_accuracy = cross_val_score(gnb, X, y, cv=5, scoring='accuracy')
    print(f'\nCross-validation accuracy: {cross_val_accuracy.mean() * 100:.2f}%')

def main():
    """Main function to run the workflow."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    gnb, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test)
    cross_validate(X, y)
    plot_results(X_test, y_test, y_pred)

if __name__ == "__main__":
    main()
