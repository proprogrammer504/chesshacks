"""Neural network training function for AlphaZero chess"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

import config
from alpha_net import AlphaNet, create_alpha_net, save_checkpoint, load_checkpoint
from encoder_decoder import encode_move, create_policy_vector


class ChessDataset(Dataset):
    """Dataset for chess training examples"""
    
    def __init__(self, training_examples):
        """
        Initialize dataset
        
        Args:
            training_examples: List of dicts with 'board', 'policy', 'value' keys
        """
        self.examples = training_examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        board_state = example['board']
        policy = example['policy']
        value = example['value']
        
        # Convert policy dict to vector if needed
        if isinstance(policy, dict):
            policy_vector = np.zeros(4096, dtype=np.float32)
            for move, prob in policy.items():
                move_idx = encode_move(move)
                policy_vector[move_idx] = prob
        else:
            policy_vector = policy
        
        return {
            'board': torch.FloatTensor(board_state),
            'policy': torch.FloatTensor(policy_vector),
            'value': torch.FloatTensor([value])
        }


def train_network(model, training_examples, num_epochs=10, batch_size=256, 
                  learning_rate=0.001, weight_decay=1e-4, device='cpu'):
    """
    Train neural network on training examples
    
    Args:
        model: AlphaNet neural network
        training_examples: List of training examples
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        device: Device to train on ('cpu' or 'cuda')
        
    Returns:
        dict: Training statistics
    """
    model.to(device)
    model.train()
    
    # Create dataset and dataloader
    dataset = ChessDataset(training_examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Loss functions
    policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
    value_loss_fn = nn.MSELoss()
    
    # Training statistics
    stats = {
        'policy_losses': [],
        'value_losses': [],
        'total_losses': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in progress_bar:
            board_states = batch['board'].to(device)
            policy_targets = batch['policy'].to(device)
            value_targets = batch['value'].to(device)
            
            # Forward pass
            policy_logits, value_preds = model(board_states)
            
            # Calculate losses
            # Policy loss: KL divergence between predicted and target policies
            policy_loss = policy_loss_fn(policy_logits, policy_targets)
            
            # Value loss: MSE between predicted and target values
            value_loss = value_loss_fn(value_preds, value_targets)
            
            # Total loss
            total_loss = policy_loss + value_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Update statistics
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'total_loss': total_loss.item()
            })
        
        # Calculate average losses for epoch
        avg_policy_loss = epoch_policy_loss / num_batches
        avg_value_loss = epoch_value_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        
        stats['policy_losses'].append(avg_policy_loss)
        stats['value_losses'].append(avg_value_loss)
        stats['total_losses'].append(avg_total_loss)
        
        print(f"Epoch {epoch + 1} - "
              f"Policy Loss: {avg_policy_loss:.4f}, "
              f"Value Loss: {avg_value_loss:.4f}, "
              f"Total Loss: {avg_total_loss:.4f}")
    
    return stats


def load_training_data(data_dir):
    """
    Load training data from directory
    
    Args:
        data_dir: Directory containing training data files
        
    Returns:
        list: Training examples
    """
    training_examples = []
    
    if not os.path.exists(data_dir):
        return training_examples
    
    # Load all .npy files from directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.npy'):
            filepath = os.path.join(data_dir, filename)
            data = np.load(filepath, allow_pickle=True)
            training_examples.extend(data)
    
    return training_examples


def save_training_data(training_examples, filepath):
    """
    Save training data to file
    
    Args:
        training_examples: List of training examples
        filepath: Path to save data
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, training_examples)


def train_from_self_play(model, self_play_games, num_epochs=None, batch_size=None,
                         learning_rate=None, checkpoint_path=None, device=None):
    """
    Train model from self-play games
    
    Args:
        model: AlphaNet neural network
        self_play_games: List of self-play games (each game is a list of examples)
        num_epochs: Number of epochs (default from config)
        batch_size: Batch size (default from config)
        learning_rate: Learning rate (default from config)
        checkpoint_path: Path to save checkpoint
        device: Device to train on (default from config)
        
    Returns:
        dict: Training statistics
    """
    # Use config defaults if not specified
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if device is None:
        device = config.DEVICE
    
    # Flatten games into training examples
    training_examples = []
    for game in self_play_games:
        training_examples.extend(game)
    
    print(f"Training on {len(training_examples)} examples from {len(self_play_games)} games")
    
    # Train network
    stats = train_network(model, training_examples, num_epochs=num_epochs,
                         batch_size=batch_size, learning_rate=learning_rate,
                         weight_decay=config.WEIGHT_DECAY, device=device)
    
    # Save checkpoint if path provided
    if checkpoint_path:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                              weight_decay=config.WEIGHT_DECAY)
        save_checkpoint(model, optimizer, 0, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    return stats


def continue_training(checkpoint_path, training_examples, num_epochs=None, 
                     batch_size=None, device=None):
    """
    Continue training from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        training_examples: New training examples
        num_epochs: Number of epochs (default from config)
        batch_size: Batch size (default from config)
        device: Device to train on (default from config)
        
    Returns:
        tuple: (model, stats)
    """
    # Use config defaults if not specified
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if device is None:
        device = config.DEVICE
    
    # Load checkpoint
    model, optimizer, iteration = load_checkpoint(checkpoint_path, device)
    print(f"Loaded checkpoint from iteration {iteration}")
    
    # Train network
    stats = train_network(model, training_examples, num_epochs=num_epochs,
                         batch_size=batch_size, learning_rate=config.LEARNING_RATE,
                         weight_decay=config.WEIGHT_DECAY, device=device)
    
    # Save updated checkpoint
    save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)
    print(f"Updated checkpoint saved to {checkpoint_path}")
    
    return model, stats


if __name__ == "__main__":
    # Test training function
    print("Creating neural network...")
    model = create_alpha_net()
    
    # Create dummy training data
    print("Creating dummy training data...")
    dummy_examples = []
    for i in range(100):
        board_state = np.random.randn(config.NUM_PLANES, 8, 8).astype(np.float32)
        policy = np.random.rand(4096).astype(np.float32)
        policy = policy / policy.sum()  # Normalize
        value = np.random.choice([-1.0, 0.0, 1.0])
        
        dummy_examples.append({
            'board': board_state,
            'policy': policy,
            'value': value
        })
    
    # Test training
    print("Testing training...")
    if torch.cuda.is_available():
        device = 'cuda'
        print("Using CUDA")
    else:
        device = 'cpu'
        print("Using CPU")
    
    stats = train_network(model, dummy_examples, num_epochs=2, batch_size=32, device=device)
    print(f"Training completed. Final loss: {stats['total_losses'][-1]:.4f}")