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

class AdvancedPatternAnalyzer:
    def __init__(self, sequence_length=50, pattern_lengths=[3, 5, 7, 9, 11, 13]):
        self.sequence_length = sequence_length
        self.pattern_lengths = pattern_lengths
        self.pattern_stats = defaultdict(dict)
        self.significant_patterns = defaultdict(dict)
        self.markov_chain = defaultdict(lambda: defaultdict(int))
        self.scaler = StandardScaler()
        
    def extract_advanced_features(self, sequence):
        """Extract advanced statistical and pattern-based features"""
        features = []
        
        # Enhanced pattern-based features
        for length in self.pattern_lengths:
            if len(sequence) >= length:
                pattern = tuple(sequence[-length:])
                if pattern in self.significant_patterns[length]:
                    stats = self.significant_patterns[length][pattern]
                    features.extend([
                        stats['probability_0'],
                        stats['probability_1'],
                        stats['bias'],
                        stats['chi_square_stat'],
                        stats['entropy'],
                        stats['pattern_frequency']
                    ])
                else:
                    features.extend([0.5, 0.5, 0, 0, 0, 0])
        
        # Advanced statistical features
        for window in [5, 10, 15, 20, 25, 30]:
            if len(sequence) >= window:
                recent = sequence[-window:]
                runs = self._count_runs(recent)
                features.extend([
                    np.mean(recent),
                    np.std(recent),
                    self._calculate_entropy(recent),
                    self._calculate_autocorrelation(recent),
                    runs['max_run_length'],
                    runs['avg_run_length'],
                    runs['run_count'],
                    self._calculate_trend_strength(recent)
                ])
        
        # Higher-order Markov features
        for order in [2, 3, 4]:
            if len(sequence) >= order:
                last_state = tuple(sequence[-order:])
                features.extend([
                    self.markov_chain.get((last_state, 0), 0),
                    self.markov_chain.get((last_state, 1), 0)
                ])
        
        return np.array(features)
    
    def _calculate_entropy(self, sequence):
        """Calculate Shannon entropy of sequence"""
        _, counts = np.unique(sequence, return_counts=True)
        return entropy(counts / len(sequence))
    
    def _calculate_autocorrelation(self, sequence, lag=1):
        """Calculate autocorrelation at given lag"""
        series = pd.Series(sequence)
        return series.autocorr(lag=lag)
    
    def _count_runs(self, sequence):
        """Analyze runs in the sequence"""
        runs = []
        current_run = 1
        
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        return {
            'max_run_length': max(runs),
            'avg_run_length': np.mean(runs),
            'run_count': len(runs)
        }
    
    def _calculate_trend_strength(self, sequence):
        """Calculate the strength of trend in the sequence"""
        diffs = np.diff(sequence)
        return np.abs(np.mean(diffs))
    
    def analyze_patterns(self, sequence):
        """Enhanced pattern analysis with comprehensive statistical testing"""
        for length in self.pattern_lengths:
            patterns = defaultdict(lambda: {'next_0': 0, 'next_1': 0, 'total': 0})
            
            for i in range(len(sequence) - length):
                pattern = tuple(sequence[i:i+length])
                next_value = sequence[i+length]
                patterns[pattern]['total'] += 1
                if next_value == 0:
                    patterns[pattern]['next_0'] += 1
                else:
                    patterns[pattern]['next_1'] += 1
            
            total_patterns = sum(stats['total'] for stats in patterns.values())
            
            for pattern, stats in patterns.items():
                if stats['total'] >= 20:  # Reduced minimum sample size
                    contingency_table = np.array([[stats['next_0'], stats['next_1']],
                                                [stats['total']/2, stats['total']/2]])
                    
                    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
                    
                    prob_0 = stats['next_0'] / stats['total']
                    prob_1 = stats['next_1'] / stats['total']
                    bias = abs(prob_0 - prob_1)
                    
                    if p_value < 0.1 or bias > 0.15:  # Relaxed significance criteria
                        self.significant_patterns[length][pattern] = {
                            'probability_0': prob_0,
                            'probability_1': prob_1,
                            'bias': bias,
                            'chi_square_stat': chi2_stat,
                            'p_value': p_value,
                            'entropy': self._calculate_entropy(pattern),
                            'pattern_frequency': stats['total'] / total_patterns
                        }
    
    def build_advanced_markov_chain(self, sequence, max_order=4):
        """Build multi-order Markov chain with sophisticated transition tracking"""
        for order in range(2, max_order + 1):
            for i in range(len(sequence) - order):
                state = tuple(sequence[i:i+order])
                next_value = sequence[i+order]
                self.markov_chain[(state, next_value)] += 1
        
        # Normalize probabilities
        states = set(key[0] for key in self.markov_chain.keys())
        for state in states:
            total = sum(self.markov_chain[(state, v)] for v in [0, 1])
            if total > 0:
                for v in [0, 1]:
                    self.markov_chain[(state, v)] /= total

class EnhancedEnsembleModel:
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length
        self.pattern_analyzer = AdvancedPatternAnalyzer(sequence_length=sequence_length)
        self.scaler = StandardScaler()
        
        # Enhanced base models with optimized parameters
        self.models = {
            'xgb': xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.01,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                objective='binary:logistic'
            ),
            'rf': RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=8,
                min_samples_leaf=4,
                class_weight='balanced',
                bootstrap=True,
                max_features='sqrt'
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.01,
                max_depth=6,
                min_samples_split=8,
                min_samples_leaf=4,
                subsample=0.8
            ),
            'ada': AdaBoostClassifier(
                n_estimators=200,
                learning_rate=0.01
            )
        }
        
    def prepare_advanced_data(self, sequence):
        """Prepare enhanced feature set for training"""
        X, y = [], []
        
        for i in range(len(sequence) - self.sequence_length):
            seq = sequence[i:i+self.sequence_length]
            features = self.pattern_analyzer.extract_advanced_features(seq)
            X.append(features)
            y.append(sequence[i+self.sequence_length])
            
        X = np.array(X)
        X = self.scaler.fit_transform(X)  # Scale features
        return X, np.array(y)
    
    def fit(self, sequence):
        """Train the enhanced ensemble model"""
        # Comprehensive pattern analysis
        self.pattern_analyzer.analyze_patterns(sequence)
        self.pattern_analyzer.build_advanced_markov_chain(sequence)
        
        # Prepare training data with advanced features
        X, y = self.prepare_advanced_data(sequence)
        
        # Split data for stacking
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train base models
        model_predictions = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            model_predictions[name] = model.predict_proba(X_val)[:, 1]
        
        # Calculate optimal weights based on validation performance
        weights = self._calculate_optimal_weights(model_predictions, y_val)
        self.model_weights = weights
        
        # Retrain on full dataset
        for name, model in self.models.items():
            model.fit(X, y)
            
        return self
    
    def _calculate_optimal_weights(self, predictions, true_values):
        """Calculate optimal weights for ensemble members based on performance"""
        weights = np.zeros(len(predictions))
        for i, (name, pred) in enumerate(predictions.items()):
            accuracy = accuracy_score(true_values, pred > 0.5)
            weights[i] = accuracy
        
        # Normalize weights
        weights = weights / np.sum(weights)
        return {name: weight for name, weight in zip(predictions.keys(), weights)}
    
    def predict(self, sequence):
        """Make weighted ensemble predictions"""
        if len(sequence) < self.sequence_length:
            return 0.5
            
        features = self.pattern_analyzer.extract_advanced_features(sequence[-self.sequence_length:])
        features = self.scaler.transform(features.reshape(1, -1))
        
        # Get weighted predictions from all models
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            pred_proba = model.predict_proba(features)[0][1]
            predictions.append(pred_proba)
            weights.append(self.model_weights[name])
        
        # Add Markov chain prediction with dynamic weight
        last_state = tuple(sequence[-3:])
        markov_prob = self.pattern_analyzer.markov_chain.get((last_state, 1), 0.5)
        predictions.append(markov_prob)
        weights.append(0.2)  # Fixed weight for Markov prediction
        
        # Weighted average of predictions
        final_prob = np.average(predictions, weights=weights)
        return 1 if final_prob > 0.5 else 0

def train_and_evaluate(file_path, test_size=0.2):
    """Train and evaluate the enhanced model with comprehensive metrics"""
    # Load and preprocess data
    df = pd.read_csv(file_path)
    sequence = df['Last Numerical Digit'].apply(lambda x: 0 if x < 5 else 1).values
    
    # Split data
    split_idx = int(len(sequence) * (1 - test_size))
    train_sequence = sequence[:split_idx]
    test_sequence = sequence[split_idx:]
    
    # Train enhanced model
    model = EnhancedEnsembleModel()
    model.fit(train_sequence)
    
    # Evaluate with sliding window
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
    
    # Print most significant patterns
    print("\nMost Significant Patterns Found:")
    for length, patterns in model.pattern_analyzer.significant_patterns.items():
        print(f"\nPattern Length {length}:")
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['bias'], reverse=True)
        for pattern, stats in sorted_patterns[:3]:  # Show top 3 patterns per length
            print(f"\nPattern {pattern}:")
            print(f"Bias: {stats['bias']:.3f}")
            print(f"Chi-square statistic: {stats['chi_square_stat']:.3f}")
            print(f"P-value: {stats['p_value']:.3e}")
            print(f"Pattern Frequency: {stats['pattern_frequency']:.3f}")
            print(f"Entropy: {stats['entropy']:.3f}")
    
    return model

if __name__ == "__main__":
    model = train_and_evaluate('block_data.csv')