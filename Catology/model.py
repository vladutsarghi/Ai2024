import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

class RN:
    def __init__(self):
        self.file_path = "Catology/static/final4.xlsx"
        self.train_X = None
        self.train_Y = None
        self.test_x = None
        self.test_y = None
        self.input_size = None
        self.hidden_size = 500
        self.output_size = 14
        self.learning_rate = 0.001
        self.batch_size = 100
        self.weights_input_hidden = None
        self.weights_hidden_output = None
        self.b1 = None
        self.b2 = None

    def prepare(self, att_dict):
        data = pd.read_excel(self.file_path).to_numpy()

        allData = data.copy()
        permutation = np.random.permutation(allData.shape[0])
        allData = allData[permutation]

        train_data = allData[:10000]
        test_data = allData[10000:]

        self.train_Y = train_data[:, 1].astype(int)
        self.test_y = test_data[:, 1].astype(int)

        dict_att = {
            "Number": 2,
            "Ext": 4,
            "Shy": 5,
            "Calm": 6,
            "Scared": 7,
            "Vigilant": 9,
            "Affectionate": 11,
            "Friendly": 12,
            "Solitary": 13,
            "Aggressive": 16,
            "PredatorMammal": 21,
            "Coat": 23,
            "Intelligence_Score": 24
        }
        ignored_att = [0, 1, 3, 8, 10, 14, 15, 17, 18, 19, 20, 22]
        for key, value in att_dict.items():
            if value == -1:
                ignored_att.append(dict_att[key])

        self.train_X = np.delete(train_data, ignored_att, axis=1).astype(int)
        self.test_x = np.delete(test_data, ignored_att, axis=1).astype(int)

        self.train_X = self.train_X / np.max(self.train_X)
        self.test_x = self.test_x / np.max(self.test_x)

        self.num_classes = len(np.unique(self.train_Y)) + 1
        self.train_Y = np.eye(self.num_classes)[self.train_Y]
        self.test_y = np.eye(self.num_classes)[self.test_y]

        self.input_size = 25 - len(ignored_att)
        np.random.seed(42)
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(y):
        return y * (1 - y)

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.b1
        hidden_output = self.sigmoid(hidden_input)
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.b2
        output_output = self.softmax(output_input)
        return hidden_output, output_output

    def backprop(self, X, y, hidden_output, output_output):
        output_error = output_output - y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

        weights_hidden_output_gradient = np.dot(hidden_output.T, output_error)
        weights_input_hidden_gradient = np.dot(X.T, hidden_delta)

        b2_output_gradient = np.sum(output_error, axis=0, keepdims=True)
        b1_hidden_gradient = np.sum(hidden_delta, axis=0, keepdims=True)

        self.weights_hidden_output -= self.learning_rate * weights_hidden_output_gradient
        self.weights_input_hidden -= self.learning_rate * weights_input_hidden_gradient
        self.b2 -= self.learning_rate * b2_output_gradient
        self.b1 -= self.learning_rate * b1_hidden_gradient

    @staticmethod
    def cross_entropy(y_pred, y_true):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]

    def train_dynamic(self, epochs=10):
        n_samples = self.train_X.shape[0]
        history_train_loss = []
        history_test_acc = []
        history_train_acc = []

        for epoch in range(epochs):
            permutation = np.random.permutation(n_samples)
            X_shuffled = self.train_X[permutation]
            y_shuffled = self.train_Y[permutation]

            total_loss = 0

            for start in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[start:start + self.batch_size]
                y_batch = y_shuffled[start:start + self.batch_size]

                hidden_output, output_output = self.forward(X_batch)
                self.backprop(X_batch, y_batch, hidden_output, output_output)

                batch_loss = self.cross_entropy(output_output, y_batch)
                total_loss += batch_loss

            test_accuracy = self.compute_accuracy(self.test_x, self.test_y)
            train_accuracy = self.compute_accuracy(self.train_X, self.train_Y)

            history_train_loss.append(total_loss)
            history_test_acc.append(test_accuracy)
            history_train_acc.append(train_accuracy)

            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, "
                  f"Test Accuracy: {test_accuracy:.4f}, Train Accuracy: {train_accuracy:.4f}")

    def compute_accuracy(self, X, y):
        _, output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        labels = np.argmax(y, axis=1)
        return np.mean(predictions == labels)

    def save_model(self):
        np.save('weights_input_hidden.npy', self.weights_input_hidden)
        np.save('weights_hidden_output.npy', self.weights_hidden_output)
        np.save('b1.npy', self.b1)
        np.save('b2.npy', self.b2)

