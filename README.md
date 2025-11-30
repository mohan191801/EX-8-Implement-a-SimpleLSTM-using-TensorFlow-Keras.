# EX-8-Implement-a-SimpleLSTM-using-TensorFlow-Keras.
## Name: NISHAL K
## Register Number: 2305001021
## AIM
To implement a simple Long Short-Term Memory (LSTM) neural network model using TensorFlow-Keras for sequence prediction, and to understand the working of LSTM layers in handling time-dependent data.

## ALGORITHM
1.Import the required libraries such as NumPy and TensorFlow-Keras.

2.Prepare the dataset by creating input sequences and output targets for training.

3.Reshape the input data into 3D format: (samples, timesteps, features) as required by LSTM.

4.Construct the LSTM model by adding an LSTM layer followed by a Dense output layer.

5.Compile and train the model using a suitable optimizer and loss function.

6.Predict and visualize the output to evaluate model performance.

## PROGRAM
```
# Import required libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Prepare a simple dataset
# Example sequence: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
data = np.array([i for i in range(10)])

# Convert sequence to supervised learning input-output pairs
X = []
y = []
window_size = 3   # number of time steps

for i in range(len(data) - window_size):
    X.append(data[i:i+window_size])
    y.append(data[i+window_size])

X = np.array(X)
y = np.array(y)

# Step 2: Reshape X to 3D (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Step 3: Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
model.add(Dense(1))

# Step 4: Compile the model
model.compile(optimizer='adam', loss='mse')

# Step 5: Train the model
model.fit(X, y, epochs=300, verbose=0)

# Step 6: Test the model for next value prediction
test_input = np.array([7, 8, 9])
test_input = test_input.reshape((1, window_size, 1))

predicted_value = model.predict(test_input, verbose=0)
print("Predicted next number after [7, 8, 9] :", predicted_value)
```
## OUTPUT
<img width="533" height="106" alt="image" src="https://github.com/user-attachments/assets/d5e96436-ac10-4ea8-bbd8-5b302fda4f71" />

## RESULT
A simple LSTM model was successfully implemented using TensorFlow-Keras.
The model learned the sequence pattern and accurately predicted the next value in the series.
Thus, the experiment demonstrates the capability of LSTM networks in modeling sequential and time-dependent data.
