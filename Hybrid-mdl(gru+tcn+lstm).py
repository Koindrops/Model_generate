import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, LSTM, GRU, Conv1D, MaxPooling1D, 
                                   GlobalAveragePooling1D, Concatenate, Dropout, 
                                   BatchNormalization, Flatten, Attention)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# Constants
SEQUENCE_LENGTH = 20  # Moderate sequence length
BATCH_SIZE = 128     # Larger batch size for better gradient estimates
EPOCHS = 200         # More epochs with better early stopping
LEARNING_RATE = 0.001

class AdvancedBinaryPredictor:
    def __init__(self, sequence_length=SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.features_scaler = StandardScaler()
        logging.basicConfig(level=logging.INFO)
        
    def extract_pattern_features(self, sequence):
        """Extract sophisticated pattern features from the sequence"""
        features = []
        
        for i in range(len(sequence) - self.sequence_length):
            seq = sequence[i:i + self.sequence_length]
            
            # Basic statistical features
            basic_stats = [
                np.mean(seq),
                np.std(seq),
                np.median(seq),
                np.max(seq),
                np.min(seq)
            ]
            
            # Pattern recognition features
            runs = np.diff(np.where(np.concatenate(([1], np.diff(seq) != 0, [1])))[0])
            pattern_stats = [
                len(runs),                              # Number of runs
                np.mean(runs) if len(runs) > 0 else 0,  # Average run length
                np.max(runs) if len(runs) > 0 else 0,   # Longest run
                np.sum(seq == 1) / len(seq),           # Proportion of ones
                np.sum(np.diff(seq) != 0)              # Number of changes
            ]
            
            # Frequency domain features
            fft_features = np.abs(np.fft.fft(seq)[:5])  # First 5 FFT coefficients
            
            # Local structure features
            rolling_means = [np.mean(seq[j:j+3]) for j in range(0, len(seq)-2, 2)]
            rolling_stds = [np.std(seq[j:j+3]) for j in range(0, len(seq)-2, 2)]
            
            # Combine all features
            combined_features = (basic_stats + pattern_stats + 
                               fft_features.tolist() + 
                               rolling_means + rolling_stds)
            
            features.append(combined_features)
            
        return np.array(features)

    def create_advanced_model(self, feature_dim):
        """Create an advanced ensemble model with multiple pathways"""
        
        # 1. Sequence Input Branch
        sequence_input = Input(shape=(self.sequence_length, 1))
        
        # CNN pathway
        conv1 = Conv1D(64, 3, activation='relu', padding='same')(sequence_input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv1D(32, 3, activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = GlobalAveragePooling1D()(conv1)
        
        # LSTM pathway
        lstm = LSTM(64, return_sequences=True)(sequence_input)
        lstm = BatchNormalization()(lstm)
        lstm = LSTM(32)(lstm)
        
        # GRU pathway
        gru = GRU(64, return_sequences=True)(sequence_input)
        gru = BatchNormalization()(gru)
        gru = GRU(32)(gru)
        
        # 2. Pattern Features Branch
        pattern_input = Input(shape=(feature_dim,))
        pattern_dense = Dense(32, activation='relu')(pattern_input)
        pattern_dense = BatchNormalization()(pattern_dense)
        pattern_dense = Dropout(0.3)(pattern_dense)
        
        # Combine all pathways
        combined = Concatenate()([conv1, lstm, gru, pattern_dense])
        
        # Deep fusion network
        x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output with custom activation
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=[sequence_input, pattern_input], outputs=output)
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer,
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def prepare_sequences(self, data):
        """Prepare sequences and extract advanced features"""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)
            
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Extract and scale pattern features
        pattern_features = self.extract_pattern_features(data)
        pattern_features = self.features_scaler.fit_transform(pattern_features)
        
        return sequences, targets, pattern_features
    
    def train(self, data_file):
        """Train the model with advanced techniques"""
        try:
            # Load and preprocess data
            df = pd.read_csv(data_file)
            last_digits = df["Last Numerical Digit"].dropna().astype(int).values
            binary_data = np.where(last_digits < 5, 0, 1)
            
            # Prepare sequences and features
            sequences, targets, pattern_features = self.prepare_sequences(binary_data)
            sequences = sequences.reshape(-1, self.sequence_length, 1)
            
            # Split data
            X_seq_train, X_seq_val, X_pat_train, X_pat_val, y_train, y_val = train_test_split(
                sequences, pattern_features, targets, 
                test_size=0.2, random_state=42, shuffle=True
            )
            
            # Create model
            model = self.create_advanced_model(pattern_features.shape[1])
            
            # Callbacks
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
                    patience=10,
                    min_lr=1e-6,
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=1
                )
            ]
            
            # Custom learning rate schedule
            initial_learning_rate = LEARNING_RATE
            decay_steps = 1000
            decay_rate = 0.9
            
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=True)
            
            # Train with class weights if needed
            class_counts = np.bincount(y_train.astype(int))
            class_weights = {i: len(y_train) / (2 * count) for i, count in enumerate(class_counts)}
            
            # Train the model
            history = model.fit(
                [X_seq_train, X_pat_train],
                y_train,
                validation_data=([X_seq_val, X_pat_val], y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            return model, history
            
        except Exception as e:
            logging.error(f"Error in training: {str(e)}")
            raise

if __name__ == "__main__":
    predictor = AdvancedBinaryPredictor()
    model, history = predictor.train("block_data.csv")