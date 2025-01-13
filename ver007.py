import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import chi2_contingency, entropy
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

class DeepPatternAnalyzer:
    def __init__(self, sequence_length=50, pattern_lengths=[3, 5, 7, 9, 11, 13, 15]):
        self.sequence_length = sequence_length
        self.pattern_lengths = pattern_lengths
        self.significant_patterns = defaultdict(dict)
        self.markov_chain = defaultdict(float)
        self.scaler = StandardScaler()
        self.pattern_cache = {}
        
    def extract_deep_features(self, sequence):
        features = []
        
        # Frequency domain features
        freq_features = self._extract_frequency_features(sequence)
        features.extend(freq_features)
        
        # Advanced pattern features
        for length in self.pattern_lengths:
            if len(sequence) >= length:
                pattern = tuple(sequence[-length:])
                pattern_features = self._get_pattern_features(pattern, length)
                features.extend(pattern_features)
        
        # Statistical features with multiple windows
        for window in [5, 10, 15, 20, 25, 30]:
            if len(sequence) >= window:
                stats = self._calculate_statistical_features(sequence[-window:])
                features.extend(stats)
        
        # Add positional features
        pos_features = self._extract_positional_features(sequence)
        features.extend(pos_features)
        
        return np.array(features)
    
    def _extract_frequency_features(self, sequence):
        """Extract frequency domain features"""
        if len(sequence) < 4:
            return [0] * 4
            
        fft_vals = np.fft.fft(sequence)
        fft_freq = np.fft.fftfreq(len(sequence))
        
        # Get dominant frequencies
        sorted_idx = np.argsort(np.abs(fft_vals))[-4:]
        dominant_freqs = fft_freq[sorted_idx]
        
        return list(np.abs(dominant_freqs))
    
    def _get_pattern_features(self, pattern, length):
        """Get cached or calculate pattern features"""
        if pattern in self.pattern_cache:
            return self.pattern_cache[pattern]
            
        features = []
        
        # Basic pattern statistics
        pattern_arr = np.array(pattern)
        features.extend([
            np.mean(pattern_arr),
            np.std(pattern_arr),
            np.sum(np.diff(pattern_arr) != 0) / (length - 1),  # transition rate
            np.sum(pattern_arr) / length  # density
        ])
        
        # Pattern complexity
        complexity = self._calculate_pattern_complexity(pattern)
        features.append(complexity)
        
        self.pattern_cache[pattern] = features
        return features
    
    def _calculate_pattern_complexity(self, pattern):
        """Calculate pattern complexity using compression ratio"""
        pattern_str = ''.join(map(str, pattern))
        compressed = []
        count = 1
        current = pattern_str[0]
        
        for i in range(1, len(pattern_str)):
            if pattern_str[i] == current:
                count += 1
            else:
                compressed.append((current, count))
                current = pattern_str[i]
                count = 1
                
        compressed.append((current, count))
        return len(compressed) / len(pattern_str)
    
    def _calculate_statistical_features(self, sequence):
        """Calculate comprehensive statistical features"""
        if len(sequence) < 2:
            return [0] * 6
            
        return [
            np.mean(sequence),
            np.std(sequence),
            np.median(sequence),
            np.percentile(sequence, 75) - np.percentile(sequence, 25),  # IQR
            np.sum(np.diff(sequence) != 0) / (len(sequence) - 1),  # change rate
            len(np.unique(sequence)) / len(sequence)  # uniqueness ratio
        ]
    
    def _extract_positional_features(self, sequence):
        """Extract position-based features"""
        if len(sequence) < 3:
            return [0] * 4
            
        positions = np.where(np.array(sequence) == 1)[0]
        if len(positions) == 0:
            return [0] * 4
            
        return [
            np.mean(positions) / len(sequence),
            np.std(positions) / len(sequence),
            len(positions) / len(sequence),
            np.diff(positions).mean() if len(positions) > 1 else 0
        ]

class AdvancedEnsembleModel:
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length
        self.pattern_analyzer = DeepPatternAnalyzer(sequence_length=sequence_length)
        self.scaler = StandardScaler()
        
        # Enhanced model ensemble
        self.models = {
            'xgb': xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.005,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                objective='binary:logistic'
            ),
            'rf': RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                bootstrap=True,
                max_features='sqrt'
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.005,
                max_depth=7,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=1000,
                early_stopping=True
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                C=1.0,
                gamma='scale',
                class_weight='balanced'
            )
        }
        self.model_weights = None
        
    def prepare_deep_data(self, sequence):
        """Prepare enhanced feature set with deep features"""
        X, y = [], []
        
        for i in range(len(sequence) - self.sequence_length):
            seq = sequence[i:i+self.sequence_length]
            features = self.pattern_analyzer.extract_deep_features(seq)
            X.append(features)
            y.append(sequence[i+self.sequence_length])
            
        X = np.array(X)
        if len(X) > 0:
            X = self.scaler.fit_transform(X)
        return X, np.array(y)
    
    def fit(self, sequence):
        """Train the enhanced ensemble with advanced techniques"""
        X, y = self.prepare_deep_data(sequence)
        
        if len(X) == 0:
            raise ValueError("Not enough data to train the model")
        
        # Split data for stacking
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models with cross-validation
        model_predictions = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            model_predictions[name] = model.predict_proba(X_val)[:, 1]
        
        # Calculate optimal weights using validation performance
        self.model_weights = self._calculate_optimal_weights(model_predictions, y_val)
        
        # Final training on full dataset
        for name, model in self.models.items():
            model.fit(X, y)
            
        return self
    
    def _calculate_optimal_weights(self, predictions, true_values):
        """Calculate optimal weights using advanced metrics"""
        weights = {}
        for name, pred in predictions.items():
            accuracy = accuracy_score(true_values, pred > 0.5)
            # Add penalty for overconfident predictions
            confidence_penalty = np.mean(np.abs(pred - 0.5)) * 0.1
            weights[name] = max(0, accuracy - confidence_penalty)
        
        # Normalize weights
        total = sum(weights.values())
        return {name: w/total for name, w in weights.items()}
    
    def predict(self, sequence):
        """Make ensemble predictions with confidence weighting"""
        if len(sequence) < self.sequence_length:
            return 0.5
            
        features = self.pattern_analyzer.extract_deep_features(sequence[-self.sequence_length:])
        features = self.scaler.transform(features.reshape(1, -1))
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            pred_proba = model.predict_proba(features)[0][1]
            predictions.append(pred_proba)
            weights.append(self.model_weights[name])
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted average with confidence adjustment
        final_prob = np.average(predictions, weights=weights)
        return 1 if final_prob > 0.5 else 0

def train_and_evaluate(file_path, test_size=0.2):
    """Train and evaluate the enhanced model with comprehensive metrics"""
    df = pd.read_csv(file_path)
    sequence = df['Last Numerical Digit'].apply(lambda x: 0 if x < 5 else 1).values
    
    split_idx = int(len(sequence) * (1 - test_size))
    train_sequence = sequence[:split_idx]
    test_sequence = sequence[split_idx:]
    
    model = AdvancedEnsembleModel()
    model.fit(train_sequence)
    
    predictions = []
    for i in range(len(test_sequence) - model.sequence_length):
        seq = sequence[i:i+model.sequence_length]
        pred = model.predict(seq)
        predictions.append(pred)
    
    true_values = test_sequence[model.sequence_length:]
    accuracy = accuracy_score(true_values, predictions)
    
    print("\nEnhanced Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(true_values, predictions))
    
    return model

if __name__ == "__main__":
    model = train_and_evaluate('block_data.csv')