# 4. Fundamentals of Neural Networks

## 4.1 Single Neuron
Write a Python function that simulates a single neuron with a sigmoid activation function for binary classification, handling multidimensional input features. The function should take a list of feature vectors (each vector representing multiple features for an example), associated true binary labels, and the neuron's weights (one for each feature) and bias as input. It should return the predicted probabilities after sigmoid activation and the mean squared error between the predicted probabilities and the true labels, both rounded to four decimal places.

**Input**:
```python
features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], labels = [0, 1, 0], weights = [0.7, -0.4], bias = -0.1
```

**Output**:
```python
([0.4626, 0.4134, 0.6682], 0.3349)
```

## 4.2 Implement ReLU Activation Function
Write a Python function `relu` that implements the Rectified Linear Unit (ReLU) activation function. The function should take a single float as input and return the value after applying the ReLU function. The ReLU function returns the input if it's greater than 0, otherwise, it returns 0.

**Input**:
```python
print(relu(0)) 
print(relu(1)) 
print(relu(-1))
```

**Output**:
```python
0
1
0
```

## 4.3 Leaky ReLU Activation Function
Write a Python function leaky_relu that implements the Leaky Rectified Linear Unit (Leaky ReLU) activation function. The function should take a float z as input and an optional float alpha, with a default value of 0.01, as the slope for negative inputs. The function should return the value after applying the Leaky ReLU function.

**Input**
```python
print(leaky_relu(0)) 
print(leaky_relu(1))
print(leaky_relu(-1)) 
print(leaky_relu(-2, alpha=0.1))
```

**Output**:
```python
0
1
-0.01
-0.2
```

## 4.4 Implement the PReLU Activation Function
Implement the PReLU (Parametric ReLU) activation function, a variant of the ReLU activation function that introduces a learnable parameter for negative inputs. Your task is to compute the PReLU activation value for a given input.

**Input**
```python
prelu(-2.0, alpha=0.25)
```

**Output**:
```python
-0.5
```

## 4.5 Sigmoid Activation Function Understanding
Write a Python function that computes the output of the sigmoid activation function given an input value z. The function should return the output rounded to four decimal places.

**Input**
```python
z = 0
```

**Output**:
```python
0.5
```

## 4.6 Softmax Activation Function Implementation
Write a Python function that computes the softmax activation for a given list of scores. The function should return the softmax values as a list, each rounded to four decimal places.

**Input**
```python
scores = [1, 2, 3]
```

**Output**:
```python
[0.0900, 0.2447, 0.6652]
```

# 4.7 Implementation of Log Softmax Function
In machine learning and statistics, the softmax function is a generalization of the logistic function that converts a vector of scores into probabilities. The log-softmax function is the logarithm of the softmax function, and it is often used for numerical stability when computing the softmax of large numbers.

Given a 1D numpy array of scores, implement a Python function to compute the log-softmax of the array.

**Input**
```python
A = np.array([1, 2, 3])
print(log_softmax(A))
```

**Output**:
```python
array([-2.4076, -1.4076, -0.4076])
```

## 4.8 Implementing a Simple RNN
Write a Python function that implements a simple Recurrent Neural Network (RNN) cell. The function should process a sequence of input vectors and produce the final hidden state. Use the tanh activation function for the hidden state updates. The function should take as inputs the sequence of input vectors, the initial hidden state, the weight matrices for input-to-hidden and hidden-to-hidden connections, and the bias vector. The function should return the final hidden state after processing the entire sequence, rounded to four decimal places.

**Input**
```python
input_sequence = [[1.0], [2.0], [3.0]]
    initial_hidden_state = [0.0]
    Wx = [[0.5]]  # Input to hidden weights
    Wh = [[0.8]]  # Hidden to hidden weights
    b = [0.0]     # Bias
```

**Output**:
```python
final_hidden_state = [0.9993]
```

# 4.9 Implement Long Short-Term Memory (LSTM) Network
Task: Implement Long Short-Term Memory (LSTM) Network
Your task is to implement an LSTM network that processes a sequence of inputs and produces the final hidden state and cell state after processing all inputs.

Write a class LSTM with the following methods:

- `__init__(self, input_size, hidden_size)`: Initializes the LSTM with random weights and zero biases.
- `forward(self, x, initial_hidden_state, initial_cell_state)`: Processes a sequence of inputs and returns the hidden states at each time step, as well as the final hidden state and cell state.
The LSTM should compute the forget gate, input gate, candidate cell state, and output gate at each time step to update the hidden state and cell state.

**Input**
```python
input_sequence = np.array([[1.0], [2.0], [3.0]])
initial_hidden_state = np.zeros((1, 1))
initial_cell_state = np.zeros((1, 1))

lstm = LSTM(input_size=1, hidden_size=1)
outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)

print(final_h)
```

**Output**:
```python
[[0.73698596]] (approximate)
```

# 4.10 Simple Convolutional 2D Layer
In this problem, you need to implement a 2D convolutional layer in Python. This function will process an input matrix using a specified convolutional kernel, padding, and stride.
**Input**
```python
import numpy as np

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)
```

**Output**:
```python
[[ 1.  1. -4.],[ 9.  7. -4.],[ 0. 14. 16.]]
```

# 4.11 GPT-2 Text Generation
Implement a Simplified GPT-2-like Text Generation Function
You are tasked with implementing a simplified GPT-2-like text generation function in Python. This function will incorporate the following components of a minimal GPT-2 architecture:

- Token Embeddings: Map input tokens to dense vector representations.
- Positional Embeddings: Add positional information to token embeddings.
- Multi-head Attention: Attend to various parts of the sequence.
- Feed-Forward Network: Process attention outputs through a dense layer.
- Layer Normalization: Stabilize the training process.

The function must take in the following parameters:

- Prompt: The initial text to guide the generation process.
- Number of Tokens to Generate: Specify how many tokens to output.

Your function should output the generated text.
Additionally, utilize the helper function load_encoder_hparams_and_params to retrieve:

- A dummy encoder.
- Model hyperparameters.
- Model parameters.

Build your text generation logic around these components. This exercise is designed to help you understand the core concepts behind GPT-2's autoregressive text generation.

**Input**
```python
prompt="hello", n_tokens_to_generate=5
```

**Output**:
```python
world <UNK> <UNK> <UNK> <UNK>
```