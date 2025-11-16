"""PyTorch implementation of AlphaGoZero neural network architecture"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers"""
    
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class PolicyHead(nn.Module):
    """Policy head for move probability prediction"""
    
    def __init__(self, in_channels, board_size=8):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * board_size * board_size, 4096)  # 64*64 possible moves
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        # Return log probabilities for numerical stability
        x = F.log_softmax(x, dim=1)
        return x


class ValueHead(nn.Module):
    """Value head for position evaluation"""
    
    def __init__(self, in_channels, board_size=8):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(board_size * board_size, 256)
        self.fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.tanh(x)  # Output in range [-1, 1]
        return x


class AlphaNet(nn.Module):
    """
    AlphaGoZero neural network architecture for chess
    
    Architecture:
    - Initial convolutional block
    - N residual blocks
    - Policy head (outputs move probabilities)
    - Value head (outputs position evaluation)
    """
    
    def __init__(self, num_planes=18, num_filters=256, num_residual_blocks=19, board_size=8):
        super(AlphaNet, self).__init__()
        
        self.num_planes = num_planes
        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks
        self.board_size = board_size
        
        # Initial convolutional block
        self.conv_block = ConvBlock(num_planes, num_filters, kernel_size=3, padding=1)
        
        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])
        
        # Output heads
        self.policy_head = PolicyHead(num_filters, board_size)
        self.value_head = ValueHead(num_filters, board_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, num_planes, board_size, board_size)
            
        Returns:
            tuple: (policy_logits, value)
                - policy_logits: Log probabilities of shape (batch_size, 4096)
                - value: Position evaluation of shape (batch_size, 1)
        """
        # Initial convolution
        x = self.conv_block(x)
        
        # Residual tower
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # Policy and value heads
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
    
    def predict(self, board_state):
        """
        Make prediction for a single board state
        
        Args:
            board_state: numpy.ndarray of shape (num_planes, board_size, board_size)
            
        Returns:
            tuple: (policy_probs, value)
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            x = torch.FloatTensor(board_state).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            
            policy_log_probs, value = self.forward(x)
            
            # Convert log probs to probs
            policy_probs = torch.exp(policy_log_probs).squeeze(0).cpu().numpy()
            value = value.squeeze(0).cpu().numpy()[0]
            
        return policy_probs, value
    
    def predict_batch(self, board_states):
        """
        Make predictions for a batch of board states
        
        Args:
            board_states: numpy.ndarray of shape (batch_size, num_planes, board_size, board_size)
            
        Returns:
            tuple: (policy_probs, values)
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(board_states)
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            
            policy_log_probs, values = self.forward(x)
            
            # Convert log probs to probs
            policy_probs = torch.exp(policy_log_probs).cpu().numpy()
            values = values.squeeze(1).cpu().numpy()
            
        return policy_probs, values


def create_alpha_net(num_planes=None, num_filters=None, num_residual_blocks=None):
    """
    Factory function to create AlphaNet with config defaults
    
    Args:
        num_planes: Number of input planes (default from config)
        num_filters: Number of filters (default from config)
        num_residual_blocks: Number of residual blocks (default from config)
        
    Returns:
        AlphaNet instance
    """
    if num_planes is None:
        num_planes = config.NUM_PLANES
    if num_filters is None:
        num_filters = config.NUM_FILTERS
    if num_residual_blocks is None:
        num_residual_blocks = config.NUM_RESIDUAL_BLOCKS
    
    return AlphaNet(num_planes, num_filters, num_residual_blocks)


def save_checkpoint(model, optimizer, iteration, filepath):
    """
    Save model checkpoint
    
    Args:
        model: AlphaNet instance
        optimizer: PyTorch optimizer
        iteration: Current iteration number
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': {
            'num_planes': model.num_planes,
            'num_filters': model.num_filters,
            'num_residual_blocks': model.num_residual_blocks,
            'board_size': model.board_size
        }
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, device='cpu'):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        tuple: (model, optimizer, iteration)
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Recreate model with saved config
    model_config = checkpoint['model_config']
    model = AlphaNet(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Recreate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    iteration = checkpoint['iteration']
    
    return model, optimizer, iteration


if __name__ == "__main__":
    # Test network creation
    model = create_alpha_net()
    print(f"Created AlphaNet with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(4, config.NUM_PLANES, 8, 8)
    policy, value = model(dummy_input)
    print(f"Policy output shape: {policy.shape}")
    print(f"Value output shape: {value.shape}")