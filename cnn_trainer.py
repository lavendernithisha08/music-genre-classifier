# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# Sequential is the model type we'll use, a linear stack of layers.
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]

# Here we import all the "bricks" or layer types we need to build our CNN.
from tensorflow.keras.layers import ( # pyright: ignore[reportMissingImports]
    Conv1D,               # The 1D convolutional layer
    MaxPooling1D,         # The 1D max pooling layer
    BatchNormalization,   # The layer for stabilizing learning
    Dropout,              # The layer for preventing overfitting
    Flatten,              # The layer to bridge conv blocks and dense layers
    Dense                 # The classic fully-connected neural network layer
)


# Define the path to our dataset
CSV_PATH = "features.csv"

# Load the dataset
try:
    features_df = pd.read_csv(CSV_PATH)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    exit() # Exit the script if the data isn't there

# Separate features (X) and target (y)
X = features_df.drop('genre_label', axis=1)
y = features_df['genre_label']

# Split the data into training and testing sets
# We use the same random_state to ensure the split is identical to the one
# used for our traditional models, allowing for a fair comparison.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
# Fit on training data and transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData preparation complete. Shapes before reshaping:")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")

# --- 2. Reshape Data for CNN Input ---
# This is the core of the current task.

print("\n--- Reshaping data for CNN model ---")

# A 1D CNN expects a 3D array as input: (num_samples, num_features, num_channels)
# Our data has num_samples and num_features, we need to add the 'channels' dimension.
# For our data, we have only one channel.
# We can use numpy.expand_dims to add an extra dimension at the end (axis=-1).

X_train_cnn = np.expand_dims(X_train_scaled, axis=-1)
X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)

# --- Verification Step ---
# Let's check the new shapes to confirm the operation was successful.
print("\nShapes after reshaping for CNN:")
print(f"X_train_cnn shape: {X_train_cnn.shape}") # Expected: (7500, 28, 1)
print(f"X_test_cnn shape: {X_test_cnn.shape}")   # Expected: (2500, 28, 1)

# The target variables (y_train, y_test) are pandas Series of integers.
# This format is already perfect for use with TensorFlow/Keras, especially
# with the 'sparse_categorical_crossentropy' loss function, so no changes are needed.
print(f"\ny_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
model = Sequential()

print("Sequential model canvas created successfully.")
print("The model is now ready for layers to be added.")
model.add(Conv1D(
    # `filters=32`: The layer will learn 32 different filters, meaning it can
    # detect 32 different types of low-level patterns in the features.
    filters=32, 
    
    # `kernel_size=3`: Each filter will be a small window of size 3. It will
    # slide across our 28 features, looking at 3 features at a time.
    kernel_size=3, 
    
    # `activation='relu'`: The Rectified Linear Unit activation function. It
    # introduces non-linearity, allowing the model to learn complex patterns.
    activation='relu', 
    
    # `input_shape`: CRUCIAL for the first layer. We tell the model to expect
    # input data where each sample has a shape of (28, 1).
    # `X_train_cnn.shape[1]` programmatically gets the number of features (28).
    input_shape=(X_train_cnn.shape[1], 1) 
))
model.add(MaxPooling1D(
    # `pool_size=2`: The layer will slide a window of size 2 over the feature maps
    # and take the maximum value from each window. This effectively halves the
    # spatial dimension of the output from the previous layer.
    pool_size=2
))
model.add(BatchNormalization())
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

# --- THIRD CONV-POOL-NORM BLOCK ---
# We go even deeper, increasing the filter count to 128. This gives the
# network even more capacity to learn highly abstract representations.
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(
    # `units=64`: We define a hidden layer with 64 neurons.
    units=64, 
    # `activation='relu'`: We continue to use the ReLU activation function
    # for non-linearity in this hidden layer.
    activation='relu'
))

# Add a Dropout layer for regularization to prevent overfitting.
# `rate=0.3` means that 30% of the neurons from the previous Dense layer
# will be randomly set to zero during each training update.
model.add(Dropout(0.3))
model.add(Dense(
    # `units=10`: The number of neurons must match the number of classes we are
    # predicting. We have 10 music genres.
    units=10, 
    
    # `activation='softmax'`: The softmax activation function is essential for
    # multi-class classification. It converts the raw output scores (logits)
    # into a probability distribution, where each value is the probability
    # of the input belonging to a specific class.
    activation='softmax'
))
print("\n--- Compiling the CNN Model ---")

# The compile step configures the model for training.
model.compile(
    # `optimizer='adam'`: We use the Adam optimizer, an efficient and popular choice
    # that adapts the learning rate for each parameter.
    optimizer='adam', 
    
    # `loss='sparse_categorical_crossentropy'`: This is the ideal loss function for
    # multi-class classification where the true labels are integers (like our 0-9 genre labels).
    loss='sparse_categorical_crossentropy', 
    
    # `metrics=['accuracy']`: We ask the model to report its accuracy (the percentage
    # of correctly classified samples) during training and evaluation.
    metrics=['accuracy']
)

print("Model compiled successfully. It is now ready to be trained.")
print("\n--- Starting Model Training ---")

# The .fit() method starts the training loop.
# We store the output of this method in a 'history' object, which will contain
# the training and validation metrics for each epoch.
history = model.fit(
    X_train_cnn, y_train,         # The training data and its labels
    epochs=50,                     # The model will see the entire training set 50 times.
    batch_size=32,                 # The model updates its weights after every 32 samples.
    validation_split=0.2           # 20% of the training data will be held out for validation.
)

print("\n--- Model Training Complete ---")

print("\n--- Model Summary After Adding First Conv1D Layer ---")
model.summary()
print("\n--- Plotting Training and Validation History ---")

import matplotlib.pyplot as plt

# Create a function to plot the training history. This is good practice for reusability.
def plot_history(history):
    """Plots accuracy and loss for training and validation sets."""
    
    # Create a figure with two subplots: one for accuracy, one for loss
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # --- Plot Training & Validation Accuracy ---
    # Access the accuracy history from the history object
    axs[0].plot(history.history["accuracy"], label="Training Accuracy")
    # Access the validation accuracy history
    axs[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Training and Validation Accuracy")
    axs[0].legend(loc="lower right")

    # --- Plot Training & Validation Loss ---
    # Access the loss history
    axs[1].plot(history.history["loss"], label="Training Loss")
    # Access the validation loss history
    axs[1].plot(history.history["val_loss"], label="Validation Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Training and Validation Loss")
    axs[1].legend(loc="upper right")

    # Ensure the plots are nicely arranged
    plt.tight_layout()

    # Display the plots
    plt.show()

# Call the function with our history object
plot_history(history)
print("\n--- Saving the trained CNN model to disk ---")

# The model.save() method saves the entire model to a single HDF5 file.
# This file includes the model's architecture, weights, and training configuration.
# We give it a descriptive name for easy identification later.
model.save("music_genre_cnn.h5")

print("\nModel successfully saved as 'music_genre_cnn.h5' in your project directory.")
print("This file can now be loaded for future evaluation or deployment.")





