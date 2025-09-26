import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

class MnistClassifier:

    def __init__(self, learning_rate=0.01, epochs=1000, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.weights = None  # Shape: (n_features, n_classes)
        self.n_classes = 10  # Digits 0-9
        self.n_features = 784  # 28x28 pixels
        self.trained = False
        self.scaler = StandardScaler()

    def LoadDataset(self,n_samples):
        print("loading MNIST dataset")
        mnist = fetch_openml('mnist_784',version=1,as_frame=False,parser='auto')
        X,y = mnist.data, mnist.target.astype(int)

        X= X[:n_samples]
        y= y[:n_samples]

        train_split = int(0.8 * len(X))
        X_train,X_test = X[:train_split],X[train_split:]
        print(X_test)

        y_train,y_test = y[:train_split], y[train_split:]
        print(y_test)

        # normalization
        X_train = X_train/255.0
        y_train = y_train/255.0
        print(X_train)

        # stdscalar
        X_train = self.scaler.fit(X_train)
        X_test = self.scaler.fit(X_test)
        print(X_test)

        return X_train, X_test, y_train, y_test

    def softmax (self,z):
        z_stable = z - np.max(z, axis=1, keepdims=True)  # For numerical stability
        exp_z = np.exp(z_stable)
        probabilities = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return probabilities
    
    def one_hot_encoding(self,y):
        n_samples = len(y)
        y_onehot = np.zeros((n_samples),self.n_classes)
        y_onehot[np.arange(n_samples), y] = 1
        return y_onehot
    
    def compute_cost(self, y_true_onehot, y_pred_proba):
        """
        Compute cross-entropy cost for multi-class classification
        """
        n_samples = y_true_onehot.shape[0]
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        cost = -np.sum(y_true_onehot * np.log(y_pred_proba)) / n_samples
        return cost
    
    def compute_gradients(self, X, y_true_onehot, y_pred_proba):
        """
        Compute gradients for weight updates in multi-class setting
        """
        n_samples = X.shape[0]
        error = y_pred_proba - y_true_onehot  # Shape: (n_samples, n_classes)
        gradients = np.dot(X.T, error) / n_samples  # Shape: (n_features, n_classes)
        return gradients
    
    def train(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        np.random.seed(self.random_state)
        self.weights = np.random.normal(0, 0.01, (n_features, self.n_classes))
        y_train_onehot = self.one_hot_encode(y_train)

        print(f"Training on {n_samples} samples with {n_features} features...")
        print(f"Weight matrix shape: {self.weights.shape}")

        costs = []
        for iteration in range(self.max_iterations):
            # Forward pass
            z = np.dot(X_train, self.weights)
            probabilities = self.softmax(z)

            # Compute cost
            cost = self.compute_cost(y_train_onehot, probabilities)

            # Compute gradients
            gradients = self.compute_gradients(X_train, y_train_onehot, probabilities)

            # Update weights
            self.weights -= self.learning_rate * gradients

            costs.append(cost)

            if iteration % 100 == 0:
                predictions = np.argmax(probabilities, axis=1)
                accuracy = np.mean(predictions == y_train) * 100
                print(f"Iteration {iteration}, Cost: {cost:.4f}, Accuracy: {accuracy:.2f}%")

        self.trained = True
        print("Training completed!")
        self.plot_cost_curve(costs)
        return costs
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples
        """
        if not self.trained:
            raise ValueError("Model must be trained first!")
        z = np.dot(X, self.weights)
        probabilities = self.softmax(z)
        return probabilities

    def predict(self, X):
        """
        Make class predictions
        """
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data
        """
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        accuracy = np.mean(predictions == y_test) * 100

        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Overall Accuracy: {accuracy:.2f}%")

        # Per-class accuracy
        for digit in range(self.n_classes):
            digit_mask = (y_test == digit)
            if np.sum(digit_mask) > 0:
                digit_accuracy = np.mean(predictions[digit_mask] == digit) * 100
                print(f"Digit {digit} Accuracy: {digit_accuracy:.2f}% ({np.sum(digit_mask)} samples)")

        # Optional: Confusion matrix
        # from sklearn.metrics import confusion_matrix
        # cm = confusion_matrix(y_test, predictions)
        # print("Confusion Matrix:\n", cm)

        return accuracy