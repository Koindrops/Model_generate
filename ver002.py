import numpy as np
import pandas as pd
import pickle
import logging
import os
import json
from datetime import datetime
from google.colab import drive
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tcn import TCN

class TrainingManager:
    def __init__(self, base_path="/content/drive/MyDrive/model_training"):
        """Initialize training manager with paths and configuration."""
        self.base_path = base_path
        self.checkpoint_dir = os.path.join(base_path, "checkpoints")
        self.logs_dir = os.path.join(base_path, "logs")
        self.data_dir = os.path.join(base_path, "data")
        
        # Create necessary directories
        for directory in [self.checkpoint_dir, self.logs_dir, self.data_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Training configuration
        self.config = {
            "SEQUENCE_LENGTH": 36,
            "EPOCHS": 50,
            "BATCH_SIZE": 32,
            "LEARNING_RATE": 0.001,
            "NUM_CLASSES": 10
        }
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.logs_dir, "training.log")),
                logging.StreamHandler()
            ]
        )
        
        self.training_state = self.load_training_state()

    class SaveTrainingStateCallback(Callback):
        """Custom callback to save training state after each epoch."""
        def __init__(self, manager):
            super().__init__()
            self.manager = manager

        def on_epoch_end(self, epoch, logs=None):
            self.manager.save_training_state(epoch + 1, logs)

    def mount_drive(self):
        """Mount Google Drive and create necessary directories."""
        try:
            drive.mount('/content/drive')
            logging.info("Google Drive mounted successfully")
        except Exception as e:
            logging.error(f"Failed to mount Google Drive: {str(e)}")
            raise

    def save_training_state(self, epoch, logs=None):
        """Save current training state and metrics."""
        state = {
            "last_epoch": epoch,
            "logs": logs,
            "timestamp": datetime.now().isoformat()
        }
        
        state_path = os.path.join(self.logs_dir, "training_state.json")
        with open(state_path, "w") as f:
            json.dump(state, f)
        
        logging.info(f"Training state saved at epoch {epoch}")

    def load_training_state(self):
        """Load previous training state if it exists."""
        state_path = os.path.join(self.logs_dir, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
            logging.info(f"Loaded training state from epoch {state['last_epoch']}")
            return state
        return {"last_epoch": 0, "logs": None}

    def save_data_state(self, data_dict):
        """Save preprocessed data and features."""
        data_path = os.path.join(self.data_dir, "processed_data.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(data_dict, f)
        logging.info("Data state saved successfully")

    def load_data_state(self):
        """Load preprocessed data if available."""
        data_path = os.path.join(self.data_dir, "processed_data.pkl")
        if os.path.exists(data_path):
            with open(data_path, "rb") as f:
                data_dict = pickle.load(f)
            logging.info("Data state loaded successfully")
            return data_dict
        return None

    def get_callbacks(self):
        """Configure training callbacks."""
        checkpoint_path = os.path.join(self.checkpoint_dir, "model_{epoch:02d}_{val_loss:.4f}.h5")
        
        callbacks = [
            ModelCheckpoint(
                checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            CSVLogger(
                os.path.join(self.logs_dir, "training_log.csv"),
                append=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                verbose=1,
                restore_best_weights=True
            ),
            self.SaveTrainingStateCallback(self)
        ]
        
        return callbacks

    def train_model(self, model, train_data, val_data):
        """Train the model with checkpointing and state management."""
        X_train, trans_train, freq_train, y_train = train_data
        X_val, trans_val, freq_val, y_val = val_data
        
        initial_epoch = self.training_state["last_epoch"]
        
        try:
            history = model.fit(
                [X_train, trans_train, freq_train],
                y_train,
                validation_data=([X_val, trans_val, freq_val], y_val),
                initial_epoch=initial_epoch,
                epochs=self.config["EPOCHS"],
                batch_size=self.config["BATCH_SIZE"],
                callbacks=self.get_callbacks(),
                verbose=1
            )
            
            # Save final model
            final_model_path = os.path.join(self.base_path, "final_model.h5")
            model.save(final_model_path)
            logging.info(f"Final model saved to {final_model_path}")
            
            return history
            
        except Exception as e:
            logging.error(f"Training interrupted: {str(e)}")
            raise

def main():
    # Initialize training manager
    manager = TrainingManager()
    
    try:
        # Mount Google Drive
        manager.mount_drive()
        
        # Check for existing processed data
        data_state = manager.load_data_state()
        
        if data_state is None:
            # Process data if not available
            # ... [Your existing data processing code here]
            data_state = {
                "X_train": X_train,
                "X_val": X_val,
                "trans_train": trans_train,
                "trans_val": trans_val,
                "freq_train": freq_train,
                "freq_val": freq_val,
                "y_train": y_train,
                "y_val": y_val
            }
            manager.save_data_state(data_state)
        
        # Check for existing model checkpoint
        latest_checkpoint = None
        if os.path.exists(manager.checkpoint_dir):
            checkpoints = [f for f in os.listdir(manager.checkpoint_dir) if f.endswith('.h5')]
            if checkpoints:
                latest_checkpoint = os.path.join(manager.checkpoint_dir, sorted(checkpoints)[-1])
        
        if latest_checkpoint:
            # Resume training from checkpoint
            model = load_model(latest_checkpoint)
            logging.info(f"Resumed training from checkpoint: {latest_checkpoint}")
        else:
            # Create new model
            model = create_improved_hybrid_tcn_model(
                manager.config["SEQUENCE_LENGTH"],
                data_state["trans_train"].shape[1]
            )
            logging.info("Created new model")
        
        # Train the model
        train_data = (
            data_state["X_train"],
            data_state["trans_train"],
            data_state["freq_train"],
            data_state["y_train"]
        )
        val_data = (
            data_state["X_val"],
            data_state["trans_val"],
            data_state["freq_val"],
            data_state["y_val"]
        )
        
        history = manager.train_model(model, train_data, val_data)
        
        # Save final training history
        history_path = os.path.join(manager.logs_dir, "final_history.json")
        with open(history_path, "w") as f:
            json.dump(history.history, f)
        
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()