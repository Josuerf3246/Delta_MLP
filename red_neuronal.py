import numpy as np

class RedNeuralAdaptativa:
    def __init__(self, input_size=1024, hidden_size=64, output_size=4, learning_rate=0.01):
        self.lr = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self._softmax(self.Z2)  
        return self.A2

    def train_step(self, X, y_target):
        m = X.shape[0]
        predictions = self.forward(X)
        
        dZ2 = (predictions - y_target) / m
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, self.W2.T) * self._relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        
        loss = -np.sum(y_target * np.log(predictions + 1e-9)) / m
        return loss

    def predecir_carta(self, imagen_normalizada):
        probabilidades = self.forward(imagen_normalizada)
        clase_predicha = np.argmax(probabilidades, axis=1)[0]
        confianza = np.max(probabilidades)
        return clase_predicha, confianza