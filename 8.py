import numpy as np
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 0, 0, 1])
weights = np.random.rand(2)
bias = np.random.rand(1) 
print(weights)
print(bias)
learning_rate = 0.1

epochs = 10

def step_function(x):
    return 1 if x >= 0 else 0
for epoch in range(epochs):
    for i in range(len(inputs)):
        weighted_sum = np.dot(inputs[i], weights) + bias
        prediction = step_function(weighted_sum)
        error = outputs[i] - prediction
        weights += learning_rate * error * inputs[i]
        bias += learning_rate * error
    print(f"Epoch {epoch+1}: weights={weights}, bias={bias}")
print("\nTesting Perceptron for AND Gate:")
for i in range(len(inputs)):
    weighted_sum = np.dot(inputs[i], weights) + bias
    prediction = step_function(weighted_sum)
    print(f"Input: {inputs[i]}, Prediction: {prediction}, Expected: {outputs[i]}")