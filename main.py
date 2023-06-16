import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_features, learning_rate=0.1, max_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = np.zeros(num_features + 1)  # +1 for the bias term

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation >= 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.max_epochs):
            error_count = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                update = self.learning_rate * (label - prediction)
                self.weights[1:] += update * inputs
                self.weights[0] += update
                error_count += int(update != 0.0)
            if error_count == 0:
                break

    def plot_decision_boundary(self, training_inputs, labels):
        plt.scatter(training_inputs[:, 0], training_inputs[:, 1], c=labels, cmap='bwr')
        x_min, x_max = training_inputs[:, 0].min() - 1, training_inputs[:, 0].max() + 1
        y_min, y_max = training_inputs[:, 1].min() - 1, training_inputs[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        Z = np.array([self.predict([x, y]) for x, y in np.c_[xx.ravel(), yy.ravel()]])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap='bwr', alpha=0.5)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Perceptron Decision Boundary')
        plt.show()

# Get user input for training data
num_training_samples = int(input("Enter the number of training samples: "))
print("Enter training data (space-separated features followed by label):")
training_data = np.zeros((num_training_samples, 2))
labels = np.zeros(num_training_samples)
for i in range(num_training_samples):
    data = list(map(int, input(f"Sample {i+1}: ").split()))
    training_data[i] = data[:-1]
    labels[i] = data[-1]

# Get user input for test data
num_test_samples = int(input("Enter the number of test samples: "))
print("Enter test data (space-separated features):")
test_data = np.zeros((num_test_samples, 2))
for i in range(num_test_samples):
    data = list(map(int, input(f"Sample {i+1}: ").split()))
    test_data[i] = data

perceptron = Perceptron(num_features=2)
perceptron.train(training_data, labels)

# Test the trained perceptron
for data in test_data:
    prediction = perceptron.predict(data)
    print(f"Input: {data}  Prediction: {prediction}")

# Plot decision boundary and data points
perceptron.plot_decision_boundary(training_data, labels)
