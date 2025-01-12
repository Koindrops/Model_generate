import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, LSTM, Bidirectional
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tcn import TCN

SEQUENCE_LENGTH = 12
MODEL_PATH = "hybrid_tcn_model.h5"
SCALER_PATH = "scaler.pkl"
CSV_FILE_PATH = "block_data.csv"
EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.002

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_advanced_features(data, binary_data, sequence_length):
    """Extract more sophisticated features from the sequence"""
    features = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        bin_seq = binary_data[i:i + sequence_length]
        
        # Statistical features
        feat = [
            np.mean(seq),
            np.std(seq),
            np.median(seq),
            np.percentile(seq, 25),
            np.percentile(seq, 75),
            np.sum(bin_seq) / len(bin_seq),  # Ratio of 1s
            np.diff(bin_seq).sum(),  # Number of changes
        ]
        
        # Pattern features
        runs = np.diff(np.where(np.concatenate(([1], np.diff(bin_seq) != 0, [1])))[0])
        feat.extend([
            len(runs),  # Number of runs
            np.mean(runs) if len(runs) > 0 else 0,  # Average run length
            np.max(runs) if len(runs) > 0 else 0,  # Longest run
        ])
        
        features.append(feat)
    
    return np.array(features)

def create_sequences_with_features(data, binary_data, sequence_length):
    """Create sequences with multiple feature views"""
    X, y = [], []
    features = []
    advanced_features = extract_advanced_features(data, binary_data, sequence_length)
    
    for i in range(len(data) - sequence_length):
        # Main sequence
        seq = binary_data[i:i + sequence_length]
        X.append(seq)
        y.append(binary_data[i + sequence_length])
        
        # Calculate additional sequential features
        seq_features = [
            np.convolve(seq, [1/3, 1/3, 1/3], 'valid'),  # Moving average
            np.diff(seq),  # First difference
            np.cumsum(seq) / (np.arange(len(seq)) + 1)  # Running mean
        ]
        features.append(np.concatenate([f.flatten() for f in seq_features]))
    
    return np.array(X), np.array(y), np.array(features), advanced_features

def create_enhanced_model(sequence_length, seq_features_dim, advanced_features_dim):
    """Create an enhanced hybrid model with multiple pathways"""
    # Sequence input branch
    sequence_input = Input(shape=(sequence_length, 1))
    
    # TCN pathway - removed kernel_regularizer
    tcn = TCN(128, activation='relu', return_sequences=True)(sequence_input)
    tcn = BatchNormalization()(tcn)
    tcn = Dropout(0.4)(tcn)
    
    tcn = TCN(64, activation='relu')(tcn)
    tcn = BatchNormalization()(tcn)
    tcn = Dropout(0.4)(tcn)
    
    # LSTM pathway
    lstm = Bidirectional(LSTM(64, return_sequences=True,
                            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))(sequence_input)
    lstm = BatchNormalization()(lstm)
    lstm = Dropout(0.4)(lstm)
    
    lstm = Bidirectional(LSTM(32,
                            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))(lstm)
    lstm = BatchNormalization()(lstm)
    lstm = Dropout(0.4)(lstm)
    
    # Sequential features pathway
    seq_features_input = Input(shape=(seq_features_dim,))
    seq_features_dense = Dense(32, activation='relu',
                             kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(seq_features_input)
    seq_features_dense = BatchNormalization()(seq_features_dense)
    seq_features_dense = Dropout(0.3)(seq_features_dense)
    
    # Advanced features pathway
    advanced_features_input = Input(shape=(advanced_features_dim,))
    advanced_features_dense = Dense(32, activation='relu',
                                  kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(advanced_features_input)
    advanced_features_dense = BatchNormalization()(advanced_features_dense)
    advanced_features_dense = Dropout(0.3)(advanced_features_dense)
    
    # Combine all pathways
    combined = Concatenate()([tcn, lstm, seq_features_dense, advanced_features_dense])
    
    # Deep fusion network
    dense = Dense(128, activation='relu',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(combined)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)
    
    dense = Dense(64, activation='relu',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)
    
    output = Dense(1, activation='sigmoid')(dense)
    
    model = Model(inputs=[sequence_input, seq_features_input, advanced_features_input],
                 outputs=output)
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_and_save_hybrid_model():
    try:
        # Load and preprocess data
        data = pd.read_csv(CSV_FILE_PATH)
        last_digits = data["Last Numerical Digit"].dropna().astype(int).values
        binary_digits = np.where(last_digits < 5, 0, 1)
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(last_digits.reshape(-1, 1))
        
        # Create sequences and features
        logging.info("Creating sequences and features...")
        X, y, seq_features, advanced_features = create_sequences_with_features(
            scaled_data.flatten(), binary_digits, SEQUENCE_LENGTH)
        
        # Reshape X for CNN input
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        logging.info(f"Data shapes - X: {X.shape}, seq_features: {seq_features.shape}, "
                    f"advanced_features: {advanced_features.shape}")
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        seq_feat_train, seq_feat_val = train_test_split(seq_features, test_size=0.2, random_state=42)
        adv_feat_train, adv_feat_val = train_test_split(advanced_features, test_size=0.2, random_state=42)
        
        # Create model
        logging.info("Creating model...")
        model = create_enhanced_model(
            SEQUENCE_LENGTH,
            seq_features.shape[1],
            advanced_features.shape[1]
        )
        
        # Enhanced callbacks
        callbacks = [
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            CSVLogger('training_log.csv', append=True)
        ]
        
        # Train the model
        logging.info("Starting model training...")
        history = model.fit(
            [X_train, seq_feat_train, adv_feat_train],
            y_train,
            validation_data=([X_val, seq_feat_val, adv_feat_val], y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the model and training history
        model.save(MODEL_PATH)
        with open('training_history.json', 'w') as f:
            import json
            history_serializable = {key: [float(value) for value in values] 
                                 for key, values in history.history.items()}
            json.dump(history_serializable, f)
        
        logging.info("Model training completed successfully")
        
    except Exception as e:
        logging.error(f"Error in training model: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_hybrid_model()