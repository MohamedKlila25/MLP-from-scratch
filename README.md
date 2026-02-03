# Multi-Layer Perceptron (MLP) from Scratch

A complete implementation of a Multi-Layer Perceptron neural network built from scratch using only NumPy. This project demonstrates the fundamental concepts of deep learning without relying on high-level frameworks like TensorFlow or PyTorch.

## Features

- **Pure NumPy Implementation**: No deep learning frameworks used
- **Flexible Architecture**: Support for arbitrary number of hidden layers
- **Complete Training Pipeline**: Forward propagation, backpropagation, and gradient descent
- **Visualization Tools**: 
  - Training loss and accuracy curves
  - Decision boundary visualization for 2D data
- **Binary Classification**: Sigmoid activation for binary classification tasks

##  Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [Key Functions](#key-functions)
- [Mathematical Background](#mathematical-background)
- [Results](#results)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

##  Installation

```bash
# Clone the repository
git clone https://github.com/MohamedKlila25/mlp-from-scratch.git
cd mlp-from-scratch

# Install required packages
pip install -r requirements.txt
```

## 📁 Project Structure

```
mlp-from-scratch/
│
├── MLP.py                          # Main implementation file
├── neural_network_review.ipynb     # Jupyter notebook with examples
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── trainset.hdf5                   # Training dataset (HDF5 format)
└── testset.hdf5                    # Test dataset (HDF5 format)
```

##  Quick Start

```python
from MLP import neural_network1, load_data

# Load your data
X_train, y_train, X_test, y_test = load_data()

# Train the network
# Architecture: input -> 32 -> 32 -> 32 -> output
parametres = neural_network1(
    X_train, 
    y_train, 
    hidden_layers=(32, 32, 32),
    learning_rate=0.01,
    n_iter=1000
)
```

##  Architecture

The MLP consists of:

1. **Input Layer**: Accepts feature vectors
2. **Hidden Layers**: Configurable number and size of hidden layers
3. **Output Layer**: Single neuron with sigmoid activation for binary classification

### Network Flow

```
Input → [Hidden Layer 1] → [Hidden Layer 2] → ... → [Output Layer] → Prediction
         (Sigmoid)          (Sigmoid)                  (Sigmoid)
```

##  Usage Examples

### Basic Training

```python
import numpy as np
from MLP import neural_network1

# Prepare your data (features: rows, samples: columns)
X_train = np.random.randn(10, 1000)  # 10 features, 1000 samples
y_train = np.random.randint(0, 2, (1, 1000))  # Binary labels

# Train with custom architecture
parametres = neural_network1(
    X_train, 
    y_train,
    hidden_layers=(64, 32, 16),  # 3 hidden layers
    learning_rate=0.001,
    n_iter=2000
)
```

### Making Predictions

```python
from MLP import predict

# Make predictions on new data
predictions = predict(X_test, parametres)

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test.flatten(), predictions.flatten())
print(f"Test Accuracy: {accuracy:.4f}")
```

### Visualizing Decision Boundaries (2D Data)

```python
from MLP import plot_decision_boundary

# For 2D feature space
plot_decision_boundary(X_train, y_train, parametres)
```

##  Key Functions

### `initialisation(dimensions)`
Initializes weights and biases for all layers using random normal distribution.

**Parameters:**
- `dimensions`: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]

**Returns:**
- Dictionary containing weights ('w1', 'w2', ...) and biases ('b1', 'b2', ...)

### `forward_propagation(x, parametres)`
Performs forward pass through the network.

**Parameters:**
- `x`: Input features (shape: features × samples)
- `parametres`: Network parameters from initialization

**Returns:**
- Dictionary of activations for each layer

### `back_propagation(y, activations, parametres)`
Computes gradients using backpropagation algorithm.

**Parameters:**
- `y`: True labels
- `activations`: Output from forward propagation
- `parametres`: Network parameters

**Returns:**
- Dictionary of gradients for weights and biases

### `neural_network1(x, y, hidden_layers, learning_rate, n_iter)`
Main training function that orchestrates the entire training process.

**Parameters:**
- `x`: Training features
- `y`: Training labels
- `hidden_layers`: Tuple specifying size of each hidden layer
- `learning_rate`: Learning rate for gradient descent
- `n_iter`: Number of training iterations

**Returns:**
- Trained parameters

##  Mathematical Background

### Forward Propagation

For each layer *l*:

```
Z^[l] = W^[l] × A^[l-1] + b^[l]
A^[l] = σ(Z^[l])
```

Where σ is the sigmoid function: σ(z) = 1 / (1 + e^(-z))

### Loss Function

Binary cross-entropy loss:

```
L(A, Y) = -1/m × Σ[y⋅log(a) + (1-y)⋅log(1-a)]
```

### Backpropagation

Gradient computation:

```
dZ^[L] = A^[L] - Y
dW^[l] = 1/m × dZ^[l] × (A^[l-1])^T
db^[l] = 1/m × Σ dZ^[l]
dZ^[l-1] = (W^[l])^T × dZ^[l] ⊙ A^[l-1] ⊙ (1 - A^[l-1])
```

### Parameter Update

```
W^[l] := W^[l] - α × dW^[l]
b^[l] := b^[l] - α × db^[l]
```

##  Results

The training process displays:
- **Loss curves**: Shows convergence of the model
- **Accuracy curves**: Tracks classification performance
- **Decision boundaries**: Visualizes learned decision regions (for 2D data)

Example output:
```
Training Progress: 100%|██████████| 1000/1000 [00:15<00:00, 65.43it/s]
Final Training Accuracy: 94.5%
```

##  Requirements

```
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
tqdm>=4.50.0
h5py>=2.10.0
```

Create a `requirements.txt` file:
```bash
numpy
matplotlib
scikit-learn
tqdm
h5py
jupyter  # For notebook support
```

## 🎓 Learning Objectives

This project helps understand:
- Neural network fundamentals
- Forward and backward propagation
- Gradient descent optimization
- Vectorized operations with NumPy
- Training dynamics and convergence
- Binary classification with neural networks

##  Known Limitations

- Only supports sigmoid activation (can be extended to ReLU, tanh, etc.)
- Binary classification only (can be extended to multi-class)
- No regularization techniques (L1/L2, dropout)
- No batch/mini-batch gradient descent
- No advanced optimizers (Adam, RMSprop, etc.)

##  Future Improvements

- [ ] Add ReLU and other activation functions
- [ ] Implement multi-class classification (softmax output)
- [ ] Add regularization (L2, dropout)
- [ ] Implement mini-batch gradient descent
- [ ] Add advanced optimizers (Adam, RMSprop)
- [ ] Save/load trained models
- [ ] Add more comprehensive testing
- [ ] Performance optimization

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



⭐ If you found this project helpful, please consider giving it a star!
