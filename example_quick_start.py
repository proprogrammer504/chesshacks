"""Quick start example for AlphaZero Chess Engine"""

import torch
import torch.multiprocessing as mp

from alpha_net import create_alpha_net
from chess_board import ChessBoard
from MCTS_chess import MCTS, self_play_game
from train import train_from_self_play
from evaluator import quick_evaluate


def example_1_create_model():
    """Example 1: Create and test the neural network"""
    print("="*60)
    print("Example 1: Creating Neural Network")
    print("="*60)
    
    # Create model
    model = create_alpha_net()
    print(f"✓ Created AlphaNet with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(1, 18, 8, 8)
    policy, value = model(dummy_input)
    print(f"✓ Policy output shape: {policy.shape}")
    print(f"✓ Value output shape: {value.shape}")
    print()
    
    return model


def example_2_mcts_search(model):
    """Example 2: Perform MCTS search on starting position"""
    print("="*60)
    print("Example 2: MCTS Search")
    print("="*60)
    
    # Create chess board
    board = ChessBoard()
    print("🎯 Starting position:")
    print(board)
    print()
    
    # Perform MCTS search
    mcts = MCTS(model, num_simulations=100)
    print("🔍 Running MCTS with 100 simulations...")
    move, policy = mcts.search(board, add_noise=True, temperature=1.0)
    
    print(f"✓ Selected move: {move}")
    print(f"✓ Policy has {len(policy)} legal moves")
    print(f"✓ Top 3 moves:")
    for i, (m, prob) in enumerate(sorted(policy.items(), key=lambda x: x[1], reverse=True)[:3]):
        print(f"  {i+1}. {m} (probability: {prob:.3f})")
    print()


def example_3_self_play(model, verbose=False):
    """Example 3: Generate one self-play game"""
    print("="*60)
    print("Example 3: Self-Play Game")
    print("="*60)
    
    print("🎮 Playing one self-play game (50 simulations per move)...")
    training_examples = self_play_game(model, num_simulations=50, verbose=verbose)
    
    print(f"✓ Game generated {len(training_examples)} training positions")
    print(f"✓ First position has {len(training_examples[0]['policy'])} possible moves")
    print(f"✓ Game result: {training_examples[-1]['value']:.1f}")
    print()
    
    return training_examples


def example_4_train(model, games, verbose=True):
    """Example 4: Train on self-play games"""
    print("="*60)
    print("Example 4: Training Neural Network")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Using device: {device}")
    
    print(f"🧠 Training on {len(games)} games for 2 epochs...")
    stats = train_from_self_play(
        model,
        games,
        num_epochs=2,
        batch_size=32,
        device=device,
        verbose=verbose
    )
    
    print(f"✓ Training complete")
    print(f"✓ Final loss: {stats['total_losses'][-1]:.4f}")
    print()


def example_5_evaluate(model1, model2):
    """Example 5: Evaluate two models"""
    print("="*60)
    print("Example 5: Model Evaluation")
    print("="*60)
    
    print("⚔️  Evaluating models (2 games, 50 simulations)...")
    results = quick_evaluate(model1, model2, num_games=2, num_simulations=50)
    
    print(f"✓ Model 1 wins: {results['model1_wins']}")
    print(f"✓ Model 2 wins: {results['model2_wins']}")
    print(f"✓ Draws: {results['draws']}")
    print()


def main(verbose=False):
    """Run all examples"""
    print("\n")
    print("*"*60)
    print("*" + " "*58 + "*")
    print("*" + "  AlphaZero Chess Engine - Quick Start Examples".center(58) + "*")
    print("*" + " "*58 + "*")
    print("*"*60)
    print()
    
    if verbose:
        print("🗣️  Verbose mode: ON (detailed game progress)")
    else:
        print("🗣️  Verbose mode: OFF (summary only)")
    print()
    
    # Example 1: Create model
    model = example_1_create_model()
    
    # Example 2: MCTS search
    example_2_mcts_search(model)
    
    # Example 3: Self-play
    print("🎮 Generating self-play games...")
    games = []
    for i in range(2):
        print(f"Game {i+1}/2...")
        game = example_3_self_play(model, verbose=verbose)
        games.append(game)
    print()
    
    # Example 4: Training
    example_4_train(model, games, verbose=verbose)
    
    # Example 5: Evaluation (compare model with itself)
    model2 = create_alpha_net()
    example_5_evaluate(model, model2)
    
    print("="*60)
    print("✅ All Examples Complete!")
    print("="*60)
    print()
    print("📚 Next steps:")
    print("1. 🚀 Run full training: python pipeline.py --iterations 5 --games 20")
    print("2. ⚙️  Adjust config.py for your hardware")
    print("3. ⚔️  Enable evaluation: python pipeline.py --iterations 10 --evaluate")
    print("4. 🗣️  Verbose mode: python pipeline.py --iterations 5 --verbose")
    print()


if __name__ == "__main__":
    import sys
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Check for verbose flag
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    main(verbose=verbose)