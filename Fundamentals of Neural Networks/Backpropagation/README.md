# 5. Backpropagation

# 5.1 Single Neuron with Backpropagation
Write a Python function that simulates a single neuron with sigmoid activation, and implements backpropagation to update the neuron's weights and bias. The function should take a list of feature vectors, associated true binary labels, initial weights, initial bias, a learning rate, and the number of epochs. The function should update the weights and bias using gradient descent based on the MSE loss, and return the updated weights, bias, and a list of MSE values for each epoch, each rounded to four decimal places.

**Input**:
```python
features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]], labels = [1, 0, 0], initial_weights = [0.1, -0.2], initial_bias = 0.0, learning_rate = 0.1, epochs = 2
```

**Output**:
```python
updated_weights = [0.1036, -0.1425], updated_bias = -0.0167, mse_values = [0.3033, 0.2942]
```

# 5.2 Implementing Basic Autograd Operations
Special thanks to Andrej Karpathy for making a video about this, if you haven't already check out his videos on YouTube https://youtu.be/VMj-3S1tku0?si=gjlnFP4o3JRN9dTg. Write a Python class similar to the provided 'Value' class that implements the basic autograd operations: addition, multiplication, and ReLU activation. The class should handle scalar values and should correctly compute gradients for these operations through automatic differentiation.

**Input**:
```python
a = Value(2)
        b = Value(-3)
        c = Value(10)
        d = a + b * c
        e = d.relu()
        e.backward()
        print(a, b, c, d, e)
```

**Output**:
```python
Value(data=2, grad=0) Value(data=-3, grad=0) Value(data=10, grad=0)
```

# 5.3 Implement a Simple RNN with Backpropagation Through Time (BPTT)
Task: Implement a Simple RNN with Backpropagation Through Time (BPTT)
Your task is to implement a simple Recurrent Neural Network (RNN) and backpropagation through time (BPTT) to learn from sequential data. The RNN will process input sequences, update hidden states, and perform backpropagation to adjust weights based on the error gradient.

Write a class SimpleRNN with the following methods:

- `__init__(self, input_size, hidden_size, output_size)`: Initializes the RNN with random weights and zero biases.
- `forward(self, x)`: Processes a sequence of inputs and returns the hidden states and output.
- `backward(self, x, y, learning_rate)`: Performs backpropagation through time (BPTT) to adjust the weights based on the loss.

In this task, the RNN will be trained on sequence prediction, where the network will learn to predict the next item in a sequence. You should use 1/2 * Mean Squared Error (MSE) as the loss function and make sure to aggregate the losses at each time step by summing.

**Input**:
```python
import numpy as np
    input_sequence = np.array([[1.0], [2.0], [3.0], [4.0]])
    expected_output = np.array([[2.0], [3.0], [4.0], [5.0]])
    # Initialize RNN
    rnn = SimpleRNN(input_size=1, hidden_size=5, output_size=1)
    
    # Forward pass
    output = rnn.forward(input_sequence)
    
    # Backward pass
    rnn.backward(input_sequence, expected_output, learning_rate=0.01)
    
    print(output)
    
    # The output should show the RNN predictions for each step of the input sequence.
```

**Output**:
```python
[[x1], [x2], [x3], [x4]]
```