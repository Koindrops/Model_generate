import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tcn import TCN

# Constants
SEQUENCE_LENGTH = 36
MODEL_PATH = "balanced_hybrid_tcn_model.h5"
SCALER_PATH = "scaler.pkl"
TRANSITION_MATRIX_PATH = "transition_matrix.pkl"
CSV_FILE_PATH = "block_data.csv"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_CLASSES = 10

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fetch_and_preprocess_data(file_path):
    try:
        data = pd.read_csv(file_path)
        required_columns = ["Serial Number", "Block Height", "Last Numerical Digit"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"CSV file is missing required column: {col}.")
        
        last_digits = data["Last Numerical Digit"].dropna().astype(int).values
        return last_digits
    except Exception as e:
        logging.error(f"Error in fetching and preprocessing data: {str(e)}")
        raise

def compute_transition_features(data, sequence_length):
    # Compute first-order transition matrix
    transition_matrix = np.zeros((10, 10))
    for i in range(1, len(data)):
        transition_matrix[data[i-1], data[i]] += 1
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    
    # Compute transition features for each sequence
    transition_features = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        # Calculate transition probabilities for the sequence
        seq_probs = [transition_matrix[sequence[j], sequence[j+1]] 
                    for j in range(len(sequence)-1)]
        # Calculate statistical features
        features = [
            np.mean(seq_probs),  # Average transition probability
            np.std(seq_probs),   # Variance in transitions
            np.min(seq_probs),   # Minimum probability
            np.max(seq_probs),   # Maximum probability
            np.sum(seq_probs)/len(seq_probs)  # Normalized sum
        ]
        transition_features.append(features)
    return np.array(transition_features), transition_matrix

def create_digit_frequency_features(data, sequence_length):
    frequency_features = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        # Calculate frequency of each digit in the sequence
        freqs = np.bincount(sequence, minlength=10) / len(sequence)
        frequency_features.append(freqs)
    return np.array(frequency_features)

def create_sequences(data, sequence_length):
    sequences, labels = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        labels.append(data[i + sequence_length])
    return np.array(sequences), np.array(labels)

def create_improved_hybrid_tcn_model(sequence_length, n_features):
    # Sequence input branch
    sequence_input = Input(shape=(sequence_length, 1))
    tcn_output = TCN(128, activation="relu", return_sequences=False)(sequence_input)
    tcn_output = BatchNormalization()(tcn_output)
    tcn_output = Dropout(0.3)(tcn_output)
    
    # Transition features input branch
    transition_input = Input(shape=(5,))  # 5 transition features
    transition_dense = Dense(32, activation="relu")(transition_input)
    transition_dense = BatchNormalization()(transition_dense)
    
    # Frequency features input branch
    frequency_input = Input(shape=(10,))  # 10 frequency features (one per digit)
    frequency_dense = Dense(32, activation="relu")(frequency_input)
    frequency_dense = BatchNormalization()(frequency_dense)
    
    # Combine all features
    combined = Concatenate()([tcn_output, transition_dense, frequency_dense])
    dense1 = Dense(256, activation="relu")(combined)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.4)(dense1)
    dense2 = Dense(128, activation="relu")(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.3)(dense2)
    
    # Output layer for 10-class classification
    output = Dense(NUM_CLASSES, activation="softmax")(dense2)
    
    model = Model(
        inputs=[sequence_input, transition_input, frequency_input],
        outputs=output
    )
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def train_and_save_improved_model():
    try:
        # Fetch and preprocess data
        last_digits = fetch_and_preprocess_data(CSV_FILE_PATH)
        logging.info(f"Loaded {len(last_digits)} data points")
        
        # Create sequences and compute features
        X, y = create_sequences(last_digits, SEQUENCE_LENGTH)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Compute transition features
        transition_features, transition_matrix = compute_transition_features(last_digits, SEQUENCE_LENGTH)
        frequency_features = create_digit_frequency_features(last_digits, SEQUENCE_LENGTH)
        
        # Save transition matrix for later use
        with open(TRANSITION_MATRIX_PATH, "wb") as f:
            pickle.dump(transition_matrix, f)
        
        # Convert labels to categorical
        y_categorical = to_categorical(y, num_classes=NUM_CLASSES)
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
        trans_train, trans_val = train_test_split(transition_features, test_size=0.2, random_state=42)
        freq_train, freq_val = train_test_split(frequency_features, test_size=0.2, random_state=42)
        
        # Create and train the model
        model = create_improved_hybrid_tcn_model(SEQUENCE_LENGTH, transition_features.shape[1])
        
        # Callbacks
        callbacks = [
            CSVLogger("training_log.csv", append=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
            EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True)
        ]
        
        # Train the model
        history = model.fit(
            [X_train, trans_train, freq_train],
            y_train,
            validation_data=([X_val, trans_val, freq_val], y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the model and training history
        model.save(MODEL_PATH)
        
        with open("training_history.json", "w") as f:
            import json
            history_serializable = {key: [float(value) for value in values] 
                                 for key, values in history.history.items()}
            json.dump(history_serializable, f)
        
        logging.info("Model training completed successfully")
        
    except Exception as e:
        logging.error(f"Error in training process: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_improved_model()