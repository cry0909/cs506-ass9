import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # Learning rate
        self.activation_fn = self.get_activation(activation)
        self.activation_derivative = self.get_activation_derivative(activation)

        # Initialize weights and biases
        self.weights1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.bias1 = np.zeros((1, hidden_dim))
        self.weights2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.bias2 = np.zeros((1, output_dim))

        # For visualization
        self.hidden_activations = None
        self.gradients = None

    def get_activation(self, activation):
        if activation == 'tanh':
            return np.tanh
        elif activation == 'relu':
            return lambda x: np.maximum(0, x)
        elif activation == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def get_activation_derivative(self, activation):
        if activation == 'tanh':
            return lambda x: 1 - np.tanh(x) ** 2
        elif activation == 'relu':
            return lambda x: (x > 0).astype(float)
        elif activation == 'sigmoid':
            return lambda x: x * (1 - x)
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, X):
    # Input to hidden layer
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.activation_fn(self.z1)  # Hidden layer activation

        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.sigmoid(self.z2)  # Output layer activation

        # Store hidden activations for visualization
        self.hidden_activations = self.a1  # This stores activations of the hidden layer
        return self.a2


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, X, y):
        # Compute gradients using backpropagation
        m = X.shape[0]

        # Output layer error
        dz2 = self.a2 - y
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer error
        dz1 = np.dot(dz2, self.weights2.T) * self.activation_derivative(self.z1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.weights1 -= self.lr * dw1
        self.bias1 -= self.lr * db1
        self.weights2 -= self.lr * dw2
        self.bias2 -= self.lr * db2

        # Store gradients for visualization
        self.gradients = {"dw1": dw1, "dw2": dw2, "db1": db1, "db2": db2}

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps by calling forward and backward
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden features
    hidden_features = mlp.hidden_activations
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title("Hidden Space")

    # Decision boundary in input space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, levels=[0, 0.5, 1], cmap='bwr', alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k', alpha=0.7)
    ax_input.set_title("Input Space")

    # Gradients visualization
    grad_magnitudes = np.linalg.norm(mlp.gradients["dw1"], axis=0)
    ax_gradient.bar(range(len(grad_magnitudes)), grad_magnitudes)
    ax_gradient.set_title("Gradients")

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131)
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)