import numpy as np

# Load data
data = np.genfromtxt('traffic_light_data.csv', delimiter=',')

# Split data into features and target
X = data[:, :-1] # all columns except the last one
y = data[:, -1] # last column, which contains the target labels

# Define hyperparameters
n_features = X.shape[1]
n_hidden = 10
n_output = 1
learning_rate = 0.01
n_epochs = 100

# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define function to create RNN
def create_rnn(n_features, n_hidden, n_output):
    # Initialize weights and biases for input-to-hidden layer
    Wh = np.random.randn(n_features, n_hidden)
    bh = np.zeros((1, n_hidden))
    # Initialize weights and biases for hidden-to-output layer
    Wy = np.random.randn(n_hidden, n_output)
    by = np.zeros((1, n_output))
    # Define function to compute forward pass of RNN
    def forward(X, h_prev):
        h = sigmoid(np.dot(X, Wh) + np.dot(h_prev, Wh) + bh)
        y = sigmoid(np.dot(h, Wy) + by)
        return h, y
    # Define function to compute backward pass of RNN
    def backward(X, y, h, h_prev, dh_next):
        dy = y * (1 - y) * (y - y_true)
        dh = dh_next + np.dot(dy, Wy.T) * h * (1 - h)
        dWh = np.dot(X.T, dh) + np.dot(h_prev.T, dh)
        dbh = np.sum(dh, axis=0, keepdims=True)
        dWy = np.dot(h.T, dy)
        dby = np.sum(dy, axis=0, keepdims=True)
        return dWh, dbh, dWy, dby, dh
    
    return forward, backward

# Create RNN
forward, backward = create_rnn(n_features, n_hidden, n_output)

# Train RNN
for epoch in range(n_epochs):
    # Initialize hidden state
    h_prev = np.zeros((1, n_hidden))
    # Initialize gradients
    dWh = np.zeros((n_features, n_hidden))
    dbh = np.zeros((1, n_hidden))
    dWy = np.zeros((n_hidden, n_output))
    dby = np.zeros((1, n_output))
    dh_next = np.zeros((1, n_hidden))
    # Iterate over all samples in dataset
    for i in range(X.shape[0]):
        # Compute forward pass
        h, y = forward(X[i:i+1], h_prev)
        # Compute backward pass
        dWh_i, dbh_i, dWy_i, dby_i, dh_next = backward(X[i:i+1], y, h, h_prev, dh_next)
        # Accumulate gradients
        dWh += dWh_i
        dbh += dbh_i
        dWy += dWy_i
        dby += dby_i
        h_prev = h
    # Update weights and biases
    Wh -= learning_rate * dWh
    bh -= learning_rate * dbh
    Wy -= learning_rate * dWy
    by -= learning_rate * dby
    # Compute training loss
    h_prev = np.zeros((1, n_hidden))
    y_pred =
