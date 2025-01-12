import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class BinaryPatternMatcher:
    def __init__(self, max_pattern_length=20):
        self.max_pattern_length = max_pattern_length
        self.pattern_frequencies = defaultdict(int)
        self.pattern_predictions = defaultdict(lambda: defaultdict(int))
        self.binary_conversion = {str(i): 0 for i in range(5)}
        self.binary_conversion.update({str(i): 1 for i in range(5, 10)})
        
    def convert_to_binary(self, digits):
        """Convert digits to binary sequence based on the rule:
        (0,1,2,3,4) -> 0
        (5,6,7,8,9) -> 1
        """
        return [self.binary_conversion[str(int(d))] for d in digits]
    
    def find_patterns(self, binary_sequence, min_length=3):
        """Find all recurring patterns in the binary sequence"""
        patterns = defaultdict(list)
        
        # Look for patterns of different lengths
        for length in range(min_length, self.max_pattern_length + 1):
            for i in range(len(binary_sequence) - length):
                pattern = tuple(binary_sequence[i:i+length])
                next_bit = binary_sequence[i+length] if i+length < len(binary_sequence) else None
                
                if next_bit is not None:
                    patterns[pattern].append(next_bit)
                    self.pattern_frequencies[pattern] += 1
                    self.pattern_predictions[pattern][next_bit] += 1
        
        return patterns
    
    def analyze_pattern_significance(self, patterns):
        """Analyze the statistical significance of patterns"""
        significant_patterns = {}
        
        for pattern, next_bits in patterns.items():
            if len(next_bits) < 10:  # Ignore patterns with too few occurrences
                continue
                
            zeros = sum(1 for bit in next_bits if bit == 0)
            ones = sum(1 for bit in next_bits if bit == 1)
            total = zeros + ones
            
            # Calculate probability and bias
            prob_zero = zeros / total
            prob_one = ones / total
            bias = abs(prob_zero - prob_one)
            
            if bias > 0.2:  # Only consider patterns with significant bias
                significant_patterns[pattern] = {
                    'total_occurrences': total,
                    'prob_zero': prob_zero,
                    'prob_one': prob_one,
                    'bias': bias,
                    'prediction': 0 if prob_zero > prob_one else 1
                }
        
        return significant_patterns
    
    def find_latest_pattern(self, sequence, significant_patterns):
        """Find the most recent matching pattern in the sequence"""
        for length in range(self.max_pattern_length, 2, -1):
            latest_pattern = tuple(sequence[-length:])
            if latest_pattern in significant_patterns:
                return latest_pattern
        return None

class PatternBasedPredictor(nn.Module):
    """Neural network that combines pattern matching with deep learning"""
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.pattern_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sequence, pattern_features):
        # Process sequence with LSTM
        lstm_out, _ = self.lstm(sequence.unsqueeze(-1))
        lstm_features = lstm_out[:, -1, :]
        
        # Process pattern features
        pattern_features = self.pattern_encoder(pattern_features)
        
        # Combine features
        combined = torch.cat([lstm_features, pattern_features], dim=1)
        return self.classifier(combined)

def create_pattern_features(sequence, pattern_matcher):
    """Create features based on pattern analysis"""
    features = []
    
    # Recent pattern statistics
    for window in [5, 10, 20]:
        recent = sequence[-window:]
        zeros = sum(1 for bit in recent if bit == 0)
        ones = sum(1 for bit in recent if bit == 1)
        features.extend([zeros/window, ones/window])
        
        # Pattern transitions
        transitions = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
        features.append(transitions/window)
    
    # Pattern matching features
    patterns = pattern_matcher.find_patterns(sequence)
    significant_patterns = pattern_matcher.analyze_pattern_significance(patterns)
    
    if significant_patterns:
        latest_pattern = pattern_matcher.find_latest_pattern(sequence, significant_patterns)
        if latest_pattern:
            pattern_stats = significant_patterns[latest_pattern]
            features.extend([
                pattern_stats['prob_zero'],
                pattern_stats['prob_one'],
                pattern_stats['bias'],
                pattern_stats['total_occurrences'] / len(sequence)
            ])
        else:
            features.extend([0.5, 0.5, 0, 0])
    else:
        features.extend([0.5, 0.5, 0, 0])
    
    return np.array(features)

def train_and_evaluate(data, window_size=100, train_ratio=0.8):
    """Train and evaluate the pattern-based prediction system"""
    # Convert original digits to binary sequence
    pattern_matcher = BinaryPatternMatcher()
    binary_sequence = pattern_matcher.convert_to_binary(data)
    
    # Prepare datasets
    X, y = [], []
    for i in range(len(binary_sequence) - window_size):
        sequence = binary_sequence[i:i+window_size]
        target = binary_sequence[i+window_size]
        features = create_pattern_features(sequence, pattern_matcher)
        
        X.append((sequence, features))
        y.append(target)
    
    # Split into train and test
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Initialize and train model
    model = PatternBasedPredictor(
        input_size=len(X[0][1])  # Size of pattern features
    )
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(100):
        model.train()
        epoch_loss = 0
        
        for i in range(0, len(X_train), 32):  # batch size = 32
            batch_sequences = []
            batch_features = []
            batch_targets = []
            
            for j in range(i, min(i+32, len(X_train))):
                sequence, features = X_train[j]
                batch_sequences.append(sequence)
                batch_features.append(features)
                batch_targets.append(y_train[j])
            
            batch_sequences = torch.FloatTensor(batch_sequences)
            batch_features = torch.FloatTensor(batch_features)
            batch_targets = torch.FloatTensor(batch_targets)
            
            optimizer.zero_grad()
            outputs = model(batch_sequences, batch_features)
            loss = criterion(outputs.squeeze(), batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {epoch_loss/len(X_train):.4f}')
    
    # Evaluation
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for sequence, features in X_test:
            sequence = torch.FloatTensor([sequence])
            features = torch.FloatTensor([features])
            output = model(sequence, features)
            predictions.append(1 if output.item() > 0.5 else 0)
    
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'model': model,
        'pattern_matcher': pattern_matcher
    }

def predict_next(sequence, model, pattern_matcher):
    """Predict the next value in the sequence"""
    sequence = pattern_matcher.convert_to_binary(sequence)
    features = create_pattern_features(sequence, pattern_matcher)
    
    with torch.no_grad():
        sequence_tensor = torch.FloatTensor([sequence])
        features_tensor = torch.FloatTensor([features])
        output = model(sequence_tensor, features_tensor)
        prediction = 1 if output.item() > 0.5 else 0
    
    return prediction