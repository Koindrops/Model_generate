import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tcn import TCN

# Modified Constants
SEQUENCE_LENGTH = 24  # Reduced sequence length
MODEL_PATH = "hybrid_tcn_model.h5"
SCALER_PATH = "scaler.pkl"
CSV_FILE_PATH = "block_data.csv"
BATCH_THRESHOLD = 100
EPOCHS = 100  # Increased epochs
BATCH_SIZE = 32  # Increased batch size
LEARNING_RATE = 0.001  # Increased learning rate

# Logging setup remains the same
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fetch_and_preprocess_data(file_path):
    try:
        data = pd.read_csv(file_path)
        required_columns = ["Serial Number", "Block Height", "Last Numerical Digit"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"CSV file is missing required column: {col}.")
        
        last_digits = data["Last Numerical Digit"].dropna().astype(int).values
        # Modified binary grouping strategy
        binary_digits = np.where(last_digits < 5, 0, 1)
        return last_digits, binary_digits
    except Exception as e:
        logging.error(f"Error in fetching and preprocessing data: {str(e)}")
        raise

def create_sequences(data, binary_data, sequence_length):
    sequences, labels = [], []
    stride = 2  # Added stride for more diverse sequences
    for i in range(0, len(data) - sequence_length, stride):
        sequences.append(binary_data[i:i + sequence_length])
        labels.append(binary_data[i + sequence_length])
    return np.array(sequences), np.array(labels)

def compute_markov_features(binary_data, sequence_length):
    # Enhanced Markov features
    transition_matrix = np.zeros((2, 2))
    for i in range(1, len(binary_data)):
        transition_matrix[binary_data[i - 1], binary_data[i]] += 1
    
    # Smoothing to avoid division by zero
    transition_matrix += 0.1
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
    
    markov_features = []
    window_size = 5  # Added rolling window
    
    for i in range(len(binary_data) - sequence_length):
        current_sequence = binary_data[i:i + sequence_length]
        # Calculate multiple probability features
        short_term_prob = [transition_matrix[current_sequence[j], current_sequence[j + 1]] 
                          for j in range(len(current_sequence) - 1)]
        long_term_prob = [transition_matrix[current_sequence[max(0, j-window_size)], current_sequence[j]] 
                         for j in range(window_size, len(current_sequence))]
        
        features = [
            np.mean(short_term_prob),
            np.std(short_term_prob),
            np.mean(long_term_prob),
            np.std(long_term_prob)
        ]
        markov_features.append(features)
    
    return np.array(markov_features)

def create_hybrid_tcn_model(sequence_length, markov_features_dim=4):
    # Enhanced model architecture
    sequence_input = Input(shape=(sequence_length, 1))
    
    # Multiple TCN layers with residual connections
    tcn1 = TCN(128, activation="relu", return_sequences=True)(sequence_input)
    tcn1 = BatchNormalization()(tcn1)
    tcn1 = Dropout(0.3)(tcn1)
    
    tcn2 = TCN(64, activation="relu")(tcn1)
    tcn2 = BatchNormalization()(tcn2)
    tcn2 = Dropout(0.3)(tcn2)
    
    # Enhanced Markov features processing
    markov_input = Input(shape=(markov_features_dim,))
    markov_dense = Dense(32, activation="relu")(markov_input)
    markov_dense = BatchNormalization()(markov_dense)
    markov_dense = Dropout(0.2)(markov_dense)
    
    # Combine features
    combined = Concatenate()([tcn2, markov_dense])
    
    # Deep fully connected layers
    dense1 = Dense(64, activation="relu")(combined)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.3)(dense1)
    
    dense2 = Dense(32, activation="relu")(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.2)(dense2)
    
    output = Dense(1, activation="sigmoid")(dense2)
    
    model = Model(inputs=[sequence_input, markov_input], outputs=output)
    
    # Use AMSGrad optimizer variant
    optimizer = Adam(learning_rate=LEARNING_RATE, amsgrad=True)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    
    return model

def train_and_save_hybrid_model():
    try:
        # Data preparation remains similar
        last_digits, binary_digits = fetch_and_preprocess_data(CSV_FILE_PATH)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(last_digits.reshape(-1, 1))
        
        with open(SCALER_PATH, "wb") as scaler_file:
            pickle.dump(scaler, scaler_file)
        
        # Create sequences with enhanced features
        X, y = create_sequences(scaled_data.flatten(), binary_digits, SEQUENCE_LENGTH)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        markov_features = compute_markov_features(binary_digits, SEQUENCE_LENGTH)
        
        # Stratified split to maintain class distribution
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        markov_train, markov_val = train_test_split(
            markov_features, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create model with enhanced architecture
        model = create_hybrid_tcn_model(SEQUENCE_LENGTH)
        
        # Enhanced callbacks
        csv_logger = CSVLogger("training_log.csv", append=True)
        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            restore_best_weights=True
        )
        
        # Train with class weights if necessary
        class_weights = None
        if np.mean(y_train) < 0.4 or np.mean(y_train) > 0.6:
            class_counts = np.bincount(y_train.astype(int))
            total = len(y_train)
            class_weights = {i: total / (len(class_counts) * c) for i, c in enumerate(class_counts)}
        
        # Train the model
        history = model.fit(
            [X_train, markov_train],
            y_train,
            validation_data=([X_val, markov_val], y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            callbacks=[csv_logger, lr_scheduler, early_stopping],
            class_weight=class_weights
        )
        
        # Save model and history
        model.save(MODEL_PATH)
        
        with open("training_history.json", "w") as history_file:
            import json
            history_serializable = {key: [float(value) for value in values] 
                                 for key, values in history.history.items()}
            json.dump(history_serializable, history_file)
            
        logging.info("Model and training history saved successfully.")
        
    except Exception as e:
        logging.error(f"Error in training and saving the hybrid model: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_hybrid_model()