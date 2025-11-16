"""Evaluator arena to compare neural network versions"""

import torch
import numpy as np
from tqdm import tqdm
import time

import config
from chess_board import ChessBoard
from MCTS_chess import MCTS
from alpha_net import AlphaNet, load_checkpoint


class Arena:
    """Arena for evaluating two neural networks against each other"""
    
    def __init__(self, model1: AlphaNet, model2: AlphaNet, num_simulations=None):
        """
        Initialize arena
        
        Args:
            model1: First neural network (challenger)
            model2: Second neural network (current best)
            num_simulations: Number of MCTS simulations per move
        """
        self.model1 = model1
        self.model2 = model2
        
        if num_simulations is None:
            num_simulations = config.NUM_MCTS_SIMS
        
        self.num_simulations = num_simulations
        
        # Put models in eval mode
        self.model1.eval()
        self.model2.eval()
        
        # Create MCTS for each model
        self.mcts1 = MCTS(model1, num_simulations=num_simulations)
        self.mcts2 = MCTS(model2, num_simulations=num_simulations)
    
    def play_game(self, starting_player=1, verbose=False):
        """
        Play a single game between the two models
        
        Args:
            starting_player: 1 if model1 plays white, -1 if model2 plays white
            verbose: Whether to print game progress
            
        Returns:
            int: 1 if model1 wins, -1 if model2 wins, 0 if draw
        """
        board = ChessBoard()
        move_count = 0
        
        # Determine which model plays which color
        if starting_player == 1:
            white_mcts = self.mcts1
            black_mcts = self.mcts2
            white_model = "Model1"
            black_model = "Model2"
        else:
            white_mcts = self.mcts2
            black_mcts = self.mcts1
            white_model = "Model2"
            black_model = "Model1"
        
        if verbose:
            print(f"{white_model} (White) vs {black_model} (Black)")
        
        while not board.is_game_over() and move_count < config.MAX_GAME_LENGTH:
            move_count += 1
            
            # Select MCTS based on current turn
            if board.get_turn() == 1:  # White to move
                current_mcts = white_mcts
                current_model = white_model
            else:  # Black to move
                current_mcts = black_mcts
                current_model = black_model
            
            # Get move from MCTS (no noise, greedy selection)
            move, _ = current_mcts.search(board, add_noise=False, temperature=0.0)
            
            if verbose and move_count % 10 == 0:
                print(f"Move {move_count}: {current_model} plays {move}")
            
            # Make move
            board.make_move(move)
        
        # Determine winner
        winner = board.get_winner()
        
        if verbose:
            result = board.get_result()
            print(f"Game over after {move_count} moves. Result: {result}")
        
        # Return result from model1's perspective
        if winner is None:
            winner = 0
        
        # Convert to model1's perspective
        if starting_player == 1:
            # Model1 played white
            return winner
        else:
            # Model1 played black
            return -winner
    
    def evaluate(self, num_games, verbose=False):
        """
        Evaluate models by playing multiple games
        
        Args:
            num_games: Number of games to play
            verbose: Whether to print progress
            
        Returns:
            dict: Dictionary with evaluation results
        """
        results = {
            'model1_wins': 0,
            'model2_wins': 0,
            'draws': 0,
            'games': []
        }
        
        print(f"Starting evaluation: {num_games} games")
        
        # Play games alternating starting player
        for game_idx in tqdm(range(num_games), desc="Evaluating"):
            # Alternate starting player
            starting_player = 1 if game_idx % 2 == 0 else -1
            
            # Play game
            result = self.play_game(starting_player, verbose=verbose)
            
            # Record result
            results['games'].append({'result': result, 'starting_player': starting_player})
            
            if result == 1:
                results['model1_wins'] += 1
            elif result == -1:
                results['model2_wins'] += 1
            else:
                results['draws'] += 1
        
        # Calculate statistics
        results['total_games'] = num_games
        results['model1_win_rate'] = results['model1_wins'] / num_games
        results['model2_win_rate'] = results['model2_wins'] / num_games
        results['draw_rate'] = results['draws'] / num_games
        
        # Print summary
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        print(f"Total Games:     {num_games}")
        print(f"Model1 Wins:     {results['model1_wins']} ({results['model1_win_rate']:.1%})")
        print(f"Model2 Wins:     {results['model2_wins']} ({results['model2_win_rate']:.1%})")
        print(f"Draws:           {results['draws']} ({results['draw_rate']:.1%})")
        print("="*60)
        
        return results
    
    def should_replace_model(self, num_games=None, win_threshold=None):
        """
        Determine if model1 should replace model2
        
        Args:
            num_games: Number of evaluation games (default from config)
            win_threshold: Win rate threshold for replacement (default from config)
            
        Returns:
            bool: True if model1 should replace model2
        """
        if num_games is None:
            num_games = config.EVAL_GAMES
        if win_threshold is None:
            win_threshold = config.WIN_THRESHOLD
        
        results = self.evaluate(num_games)
        
        # Model1 should replace model2 if its win rate exceeds threshold
        should_replace = results['model1_win_rate'] > win_threshold
        
        if should_replace:
            print(f"\n✓ Model1 win rate ({results['model1_win_rate']:.1%}) "
                  f"exceeds threshold ({win_threshold:.1%})")
            print("Model1 will replace Model2")
        else:
            print(f"\n✗ Model1 win rate ({results['model1_win_rate']:.1%}) "
                  f"does not exceed threshold ({win_threshold:.1%})")
            print("Model2 remains the best model")
        
        return should_replace


def evaluate_checkpoints(checkpoint1_path, checkpoint2_path, num_games=None, device='cpu'):
    """
    Evaluate two checkpoints against each other
    
    Args:
        checkpoint1_path: Path to first checkpoint (challenger)
        checkpoint2_path: Path to second checkpoint (current best)
        num_games: Number of games to play (default from config)
        device: Device to use
        
    Returns:
        tuple: (should_replace, results)
    """
    if num_games is None:
        num_games = config.EVAL_GAMES
    
    print(f"Loading Model1 from {checkpoint1_path}")
    model1, _, iter1 = load_checkpoint(checkpoint1_path, device)
    
    print(f"Loading Model2 from {checkpoint2_path}")
    model2, _, iter2 = load_checkpoint(checkpoint2_path, device)
    
    # Create arena
    arena = Arena(model1, model2)
    
    # Evaluate
    should_replace = arena.should_replace_model(num_games)
    
    return should_replace


def quick_evaluate(model1: AlphaNet, model2: AlphaNet, num_games=10, num_simulations=100):
    """
    Quick evaluation with fewer games and simulations
    
    Args:
        model1: First model
        model2: Second model
        num_games: Number of games
        num_simulations: MCTS simulations per move
        
    Returns:
        dict: Evaluation results
    """
    arena = Arena(model1, model2, num_simulations=num_simulations)
    results = arena.evaluate(num_games, verbose=False)
    return results


def play_single_game_verbose(model1: AlphaNet, model2: AlphaNet):
    """
    Play a single game with verbose output for testing
    
    Args:
        model1: First model (plays white)
        model2: Second model (plays black)
        
    Returns:
        ChessBoard: Final board state
    """
    board = ChessBoard()
    mcts1 = MCTS(model1, num_simulations=100)
    mcts2 = MCTS(model2, num_simulations=100)
    
    move_count = 0
    print("Starting game (Model1=White, Model2=Black)")
    print(board)
    print()
    
    while not board.is_game_over() and move_count < 200:
        move_count += 1
        
        if board.get_turn() == 1:
            mcts = mcts1
            player = "Model1 (White)"
        else:
            mcts = mcts2
            player = "Model2 (Black)"
        
        move, _ = mcts.search(board, add_noise=False, temperature=0.0)
        print(f"Move {move_count}: {player} plays {move}")
        board.make_move(move)
        
        if move_count % 10 == 0:
            print(board)
            print()
    
    print("\nFinal position:")
    print(board)
    print(f"Result: {board.get_result()}")
    
    return board


if __name__ == "__main__":
    # Test evaluator
    from alpha_net import create_alpha_net
    
    print("Creating two models for testing...")
    model1 = create_alpha_net()
    model2 = create_alpha_net()
    
    model1.eval()
    model2.eval()
    
    print("\nTesting quick evaluation...")
    results = quick_evaluate(model1, model2, num_games=2, num_simulations=50)
    
    print("\nTest complete!")