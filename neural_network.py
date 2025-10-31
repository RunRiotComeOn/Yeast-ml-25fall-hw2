"""
Neural Network Implementation for ECS 171 Homework 2
"""

import numpy as np

def sigmoid(z):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    """Derivative of sigmoid function"""
    return a * (1.0 - a)

def relu(z):
    """ReLU activation function"""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU"""
    return (z > 0).astype(float)

class NeuralNetwork:
    """
    4-layer feedforward neural network with customizable architecture
    """
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, 
                 learning_rate=0.1, activation='sigmoid', loss='mse'):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_type = activation
        self.loss_type = loss
        
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden1_size) * 0.01
        self.b1 = np.zeros((1, hidden1_size))
        
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * 0.01
        self.b2 = np.zeros((1, hidden2_size))
        
        self.W3 = np.random.randn(hidden2_size, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))
        
        # For tracking training progress
        self.train_errors = []
        self.test_errors = []
        self.weight_history = {'W3': [], 'b3': []}
        
    def activation(self, z):
        """Apply activation function"""
        if self.activation_type == 'sigmoid':
            return sigmoid(z)
        elif self.activation_type == 'relu':
            return relu(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation_type}")
    
    def activation_derivative(self, a, z=None):
        """Derivative of activation function"""
        if self.activation_type == 'sigmoid':
            return sigmoid_derivative(a)
        elif self.activation_type == 'relu':
            return relu_derivative(z if z is not None else a)
        else:
            raise ValueError(f"Unknown activation: {self.activation_type}")
    
    def softmax(self, z):
        """Softmax activation for output layer"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward propagation"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.activation(self.z2)
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        
        if self.loss_type == 'cross_entropy':
            self.a3 = self.softmax(self.z3)
        else:  # MSE
            self.a3 = self.activation(self.z3)
        
        return self.a3
    
    def backward(self, X, y, output):
        """Backpropagation"""
        m = X.shape[0]
        
        # Output layer gradient
        if self.loss_type == 'cross_entropy':
            dz3 = output - y
        else:  # MSE
            dz3 = (output - y) * self.activation_derivative(output, self.z3)
        
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # Hidden layer 2 gradient
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.activation_derivative(self.a2, self.z2)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer 1 gradient
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.activation_derivative(self.a1, self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
    
    def compute_loss(self, y_true, y_pred):
        """Compute loss"""
        if self.loss_type == 'cross_entropy':
            # Cross-entropy loss
            return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
        else:  # MSE
            return np.mean(np.square(y_true - y_pred))
    
    def compute_error_rate(self, X, y):
        """Compute misclassification rate (0/1 error)"""
        predictions = self.predict(X)
        y_true_labels = np.argmax(y, axis=1)
        return np.mean(predictions != y_true_labels)
    
    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size=1, 
              track_cyt_weights=False, cyt_class_idx=None):
        """Train the network"""
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch or stochastic gradient descent
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward and backward pass
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)
            
            # Track training and test error
            train_error = self.compute_error_rate(X_train, y_train)
            test_error = self.compute_error_rate(X_test, y_test)
            self.train_errors.append(train_error)
            self.test_errors.append(test_error)
            
            # Track CYT weights if requested
            if track_cyt_weights and cyt_class_idx is not None:
                self.weight_history['W3'].append(self.W3[:, cyt_class_idx].copy())
                self.weight_history['b3'].append(self.b3[0, cyt_class_idx])
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return np.argmax(output, axis=1)
