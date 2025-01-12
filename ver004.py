import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class BlockDataset(Dataset):
    def __init__(self, sequence, sequence_length=50):
        self.sequence = sequence
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.sequence) - self.sequence_length
        
    def __getitem__(self, idx):
        # Get sequence and target
        seq = self.sequence[idx:idx + self.sequence_length]
        target = self.sequence[idx + self.sequence_length]
        
        # Convert to tensors
        return torch.FloatTensor(seq), torch.FloatTensor([target])

class PatternPredictor(nn.Module):
    def __init__(self, sequence_length=50, hidden_dim=64):
        super().__init__()
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Pattern detection layers
        self.pattern_layers = nn.Sequential(
            nn.Linear(sequence_length, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Combine LSTM and pattern features
        self.combined_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Process with LSTM
        x_seq = x.unsqueeze(-1)  # Add feature dimension
        lstm_out, _ = self.lstm(x_seq)
        lstm_features = lstm_out[:, -1, :]  # Take last LSTM output
        
        # Process for pattern detection
        pattern_features = self.pattern_layers(x)
        
        # Combine features
        combined = torch.cat([lstm_features, pattern_features], dim=1)
        return self.combined_layers(combined)

def load_and_preprocess_data(file_path):
    """Load and preprocess the block data"""
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Convert last digits to binary (0-4 → 0, 5-9 → 1)
    df['binary_digit'] = df['Last Numerical Digit'].apply(lambda x: 0 if x < 5 else 1)
    
    # Create sequences array
    binary_sequence = df['binary_digit'].values
    
    return df, binary_sequence

def analyze_patterns(sequence, window_sizes=[3, 5, 7]):
    """Analyze patterns in the sequence"""
    pattern_stats = {}
    
    for window in window_sizes:
        patterns = {}
        for i in range(len(sequence) - window):
            pattern = tuple(sequence[i:i+window])
            next_value = sequence[i+window]
            
            if pattern not in patterns:
                patterns[pattern] = {'count': 0, 'next_0': 0, 'next_1': 0}
            
            patterns[pattern]['count'] += 1
            if next_value == 0:
                patterns[pattern]['next_0'] += 1
            else:
                patterns[pattern]['next_1'] += 1
        
        # Calculate probabilities and significance
        significant_patterns = {}
        for pattern, stats in patterns.items():
            if stats['count'] >= 10:  # Minimum occurrences threshold
                total = stats['next_0'] + stats['next_1']
                prob_0 = stats['next_0'] / total
                prob_1 = stats['next_1'] / total
                bias = abs(prob_0 - prob_1)
                
                if bias > 0.1:  # Bias threshold
                    significant_patterns[pattern] = {
                        'probability_0': prob_0,
                        'probability_1': prob_1,
                        'bias': bias,
                        'count': stats['count']
                    }
        
        pattern_stats[window] = significant_patterns
    
    return pattern_stats

def train_model(train_loader, val_loader, sequence_length=50, epochs=100):
    """Train the pattern prediction model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatternPredictor(sequence_length=sequence_length).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                
                val_preds.extend((outputs > 0.5).cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate validation accuracy
        val_accuracy = accuracy_score(val_true, val_preds)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy:.4f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    return model, train_losses, val_losses

def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss history"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Load and preprocess data
    df, binary_sequence = load_and_preprocess_data('block_data_10000.csv')
    
    # Analyze patterns
    pattern_stats = analyze_patterns(binary_sequence)
    
    # Create datasets
    sequence_length = 50
    dataset = BlockDataset(binary_sequence, sequence_length=sequence_length)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train model
    model, train_losses, val_losses = train_model(
        train_loader,
        val_loader,
        sequence_length=sequence_length
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    return model, pattern_stats

def predict_next_value(model, sequence):
    """Predict the next value in the sequence"""
    model.eval()
    with torch.no_grad():
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        prediction = model(sequence_tensor)
        return 1 if prediction.item() > 0.5 else 0

if __name__ == "__main__":
    # Train the model and get pattern statistics
    model, pattern_stats = main()
    
    # Print some pattern statistics
    print("\nSignificant Patterns Found:")
    for window_size, patterns in pattern_stats.items():
        print(f"\nWindow Size {window_size}:")
        for pattern, stats in list(patterns.items())[:5]:  # Show top 5 patterns
            print(f"Pattern {pattern}:")
            print(f"Bias: {stats['bias']:.3f}")
            print(f"Probability of 0: {stats['probability_0']:.3f}")
            print(f"Probability of 1: {stats['probability_1']:.3f}")
            print(f"Occurrences: {stats['count']}")