import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import chi2_contingency
import xgboost as xgb

class EnhancedBlockPredictor:
    def __init__(self, sequence_length=50, pattern_lengths=[3, 5, 7, 9, 11]):
        self.sequence_length = sequence_length
        self.pattern_lengths = pattern_lengths
        self.pattern_stats = defaultdict(dict)
        self.significant_patterns = defaultdict(dict)
        self.markov_chain = defaultdict(lambda: defaultdict(float))
        
    def create_features(self, sequence):
        """Create enhanced feature set from sequence"""
        features = []
        
        # Pattern-based features
        for length in self.pattern_lengths:
            if len(sequence) >= length:
                pattern = tuple(sequence[-length:])
                if pattern in self.significant_patterns[length]:
                    stats = self.significant_patterns[length][pattern]
                    features.extend([
                        stats['probability_0'],
                        stats['probability_1'],
                        stats['bias'],
                        stats['chi_square_stat']
                    ])
                else:
                    features.extend([0.5, 0.5, 0, 0])
        
        # Statistical features
        for window in [5, 10, 20, 30]:
            if len(sequence) >= window:
                recent = sequence[-window:]
                features.extend([
                    np.mean(recent),
                    np.std(recent),
                    sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1]) / (window-1),  # transition rate
                    sum(recent) / window,  # proportion of 1s
                ])
        
        # Markov chain features
        if len(sequence) >= 3:
            last_state = tuple(sequence[-2:])
            features.extend([
                self.markov_chain[last_state][0],
                self.markov_chain[last_state][1]
            ])
        
        return np.array(features)
    
    def analyze_patterns(self, sequence):
        """Enhanced pattern analysis with statistical testing"""
        for length in self.pattern_lengths:
            patterns = defaultdict(lambda: {'next_0': 0, 'next_1': 0, 'total': 0})
            
            # Count pattern occurrences
            for i in range(len(sequence) - length):
                pattern = tuple(sequence[i:i+length])
                next_value = sequence[i+length]
                patterns[pattern]['total'] += 1
                if next_value == 0:
                    patterns[pattern]['next_0'] += 1
                else:
                    patterns[pattern]['next_1'] += 1
            
            # Statistical analysis
            for pattern, stats in patterns.items():
                if stats['total'] >= 30:  # Minimum sample size for statistical significance
                    # Chi-square test for independence
                    contingency_table = np.array([[stats['next_0'], stats['next_1']],
                                                [stats['total']/2, stats['total']/2]])  # Expected uniform distribution
                    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
                    
                    prob_0 = stats['next_0'] / stats['total']
                    prob_1 = stats['next_1'] / stats['total']
                    bias = abs(prob_0 - prob_1)
                    
                    if p_value < 0.05 and bias > 0.1:  # Statistically significant patterns
                        self.significant_patterns[length][pattern] = {
                            'probability_0': prob_0,
                            'probability_1': prob_1,
                            'bias': bias,
                            'chi_square_stat': chi2_stat,
                            'p_value': p_value
                        }
    
    def build_markov_chain(self, sequence, order=2):
        """Build higher-order Markov chain"""
        for i in range(len(sequence) - order):
            state = tuple(sequence[i:i+order])
            next_value = sequence[i+order]
            self.markov_chain[state][next_value] += 1
        
        # Convert counts to probabilities
        for state in self.markov_chain:
            total = sum(self.markov_chain[state].values())
            for next_value in self.markov_chain[state]:
                self.markov_chain[state][next_value] /= total

class EnsembleModel:
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length
        self.pattern_predictor = EnhancedBlockPredictor(sequence_length=sequence_length)
        
        # Initialize base models
        self.models = {
            'xgb': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.01,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic'
            ),
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced'
            )
        }
        
    def prepare_data(self, sequence):
        """Prepare data for training"""
        X, y = [], []
        
        for i in range(len(sequence) - self.sequence_length):
            seq = sequence[i:i+self.sequence_length]
            features = self.pattern_predictor.create_features(seq)
            X.append(features)
            y.append(sequence[i+self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def fit(self, sequence):
        """Train the ensemble model"""
        # Analyze patterns
        self.pattern_predictor.analyze_patterns(sequence)
        self.pattern_predictor.build_markov_chain(sequence)
        
        # Prepare training data
        X, y = self.prepare_data(sequence)
        
        # Train base models
        for name, model in self.models.items():
            model.fit(X, y)
            
        return self
    
    def predict(self, sequence):
        """Make predictions using ensemble"""
        if len(sequence) < self.sequence_length:
            return 0.5  # Default prediction if sequence is too short
            
        features = self.pattern_predictor.create_features(sequence[-self.sequence_length:])
        features = features.reshape(1, -1)
        
        # Get predictions from all models
        predictions = []
        for model in self.models.values():
            pred_proba = model.predict_proba(features)[0][1]
            predictions.append(pred_proba)
        
        # Add Markov chain prediction
        last_state = tuple(sequence[-2:])
        markov_prob = self.pattern_predictor.markov_chain[last_state][1]
        predictions.append(markov_prob)
        
        # Weighted average of predictions
        final_prob = np.average(predictions, weights=[0.4, 0.3, 0.3])
        return 1 if final_prob > 0.5 else 0

def train_and_evaluate(file_path, test_size=0.2):
    """Train and evaluate the enhanced model"""
    # Load data
    df = pd.read_csv(file_path)
    sequence = df['Last Numerical Digit'].apply(lambda x: 0 if x < 5 else 1).values
    
    # Split data
    split_idx = int(len(sequence) * (1 - test_size))
    train_sequence = sequence[:split_idx]
    test_sequence = sequence[split_idx:]
    
    # Train model
    model = EnsembleModel()
    model.fit(train_sequence)
    
    # Evaluate
    predictions = []
    for i in range(len(test_sequence) - model.sequence_length):
        seq = sequence[i:i+model.sequence_length]
        pred = model.predict(seq)
        predictions.append(pred)
    
    true_values = test_sequence[model.sequence_length:]
    accuracy = accuracy_score(true_values, predictions)
    
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_values, predictions))
    
    return model

if __name__ == "__main__":
    model = train_and_evaluate('block_data.csv')
    
    # Print significant patterns
    print("\nMost Significant Patterns Found:")
    for length, patterns in model.pattern_predictor.significant_patterns.items():
        print(f"\nPattern Length {length}:")
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['bias'], reverse=True)
        for pattern, stats in sorted_patterns[:5]:
            print(f"\nPattern {pattern}:")
            print(f"Bias: {stats['bias']:.3f}")
            print(f"Chi-square statistic: {stats['chi_square_stat']:.3f}")
            print(f"P-value: {stats['p_value']:.3e}")