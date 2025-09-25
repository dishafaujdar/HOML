"""
Custom Classification Engine - Deep Understanding Implementation
Builds binary classifiers from mathematical foundations -> for spam or not spam email
"""

import numpy as np
import re
from collections import Counter
import math


class SpamClassifier:
    def __init__(self,lr=0.01,epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.vocab = {}
        self.trained = False

    def preprocess_text (self,text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        print("words",words)
        return words
    
    def build_vocabulary (self,texts):
        vocab_set = set()
        for text in texts:
            words = self.preprocess_text(text)
            vocab_set.update(words)
            print("vocab",vocab_set)
        self.vocab = {word: idx for idx, word in enumerate(sorted(vocab_set))}    

    # def text_to_features (self,texts,vocabulary= None):
    #     if vocabulary is None:
    #         vocabulary = self.vocab

    #     features = np.zeros((len(texts),len(vocabulary)))
    #     print("features",features)
    #     for i,text in enumerate(texts):
    #         words = self.preprocess_text(text)
    #         for word in words:
    #             if word in vocabulary:
    #                 idx = vocabulary[word]
    #                 features[i, idx] += 1  # Bag-of-Words: count occurrences
    #                 print("text to features",features)
    #     return features

    def text_to_features(self, texts,vocabulary= None):
            num_docs = len(texts)
            vocab_size = len(self.vocab)

            # Step 1: Bag-of-Words (count matrix)
            counts = np.zeros((num_docs, vocab_size))
            for i, text in enumerate(texts):
                words = self.preprocess_text(text)
                for word in words:
                    if word in self.vocab:
                        idx = self.vocab[word]
                        counts[i, idx] += 1

            # Step 2: Term Frequency (TF) -> normalize counts by total words in doc
            tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)

            # Step 3: Inverse Document Frequency (IDF)
            df = np.count_nonzero(counts > 0, axis=0)   # in how many docs each word appears
            idf = np.log((num_docs + 1) / (df + 1)) + 1  # smoothing

            # Step 4: TF-IDF = TF × IDF
            feature = tf * idf

            return feature
    
    def sigmoid (self,z):
        z = np.clip(z,-500,500) #prevent overflow from the expo fn since it can produce large number
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, h, y):
        m = y.shape[0]
        epsilon = 1e-15  # to avoid log(0)
        cost = -1/m * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        return cost
        
    def compute_gradients(self, X, h, y):
        """
        Compute gradients for weight updates
        
        Args:
            X: Feature matrix (n_samples, n_features)
            h: Predictions (n_samples,)
            y: True labels (n_samples,)
            
        Returns:
            gradients: Gradient values for each weight (n_features,)
        """
        m = X.shape[0]
        gradients = (1 / m) * np.dot(X.T, (h - y))
        return gradients

    def train(self, X_train, y_train):
        """
        Train the logistic regression model
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,) - values 0 or 1
        """
        n_samples, n_features = X_train.shape
        self.weights = np.random.normal(0,0.01,n_features)

        for iteration in range(self.epochs):
            z = np.dot(X_train,self.weights)
            h = self.sigmoid(z)
            # Compute cost (optional, for monitoring)
            epsilon = 1e-15
            cost = -np.mean(y_train * np.log(h + epsilon) + (1 - y_train) * np.log(1 - h + epsilon))
            # Compute gradients
            gradient = self.compute_gradients(X_train,h,y_train)
            #Update weights
            self.weights -= self.lr * gradient

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Cost: {cost:.4f}")

        self.trained = True
        print("Training completed!")
            
    def predict_proba(self, X):
        if not self.trained:
            raise ValueError("model must be trained first!")
        
        z = np.dot(X,self.weights)
        probabilites = self.sigmoid(z)
        print("probabilites",probabilites)
        return probabilites

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= 0.5).astype(int)
        print(predictions)        
        return predictions
    
    def fit_and_predict(self, train_texts, train_labels, test_texts):
        print("Starting spam classification pipeline...")
        
        # Step 1: Build vocabulary and convert to features
        print("Building vocabulary and extracting features...")
        self.build_vocabulary(train_texts)  # Use raw texts
        X_train = self.text_to_features(train_texts, self.vocab)

        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Training set shape: {X_train.shape}")

        # Step 2: Train the model
        print("Training logistic regression model...")
        y_train = np.array(train_labels)
        self.train(X_train, y_train)

        # Step 3: Preprocess and predict on test texts
        print("Making predictions on test texts...")
        X_test = self.text_to_features(test_texts, self.vocab)

        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)

        return predictions, probabilities
    
    def evaluate_model(y_true, y_pred):
        """
        Evaluate model performance: accuracy, precision, recall, F1-score

        Args:
            y_true: True labels (list or np.array)
            y_pred: Predicted labels (list or np.array)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        accuracy = np.mean(y_true == y_pred)

        # True Positives, False Positives, True Negatives, False Negatives
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}        
