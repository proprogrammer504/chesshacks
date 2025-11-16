"""Multiprocessing version of training with parallel self-play"""

import torch
import torch.multiprocessing as mp
import numpy as np
import os
from tqdm import tqdm
import time

import config
from alpha_net import create_alpha_net, load_checkpoint
from MCTS_chess import self_play_game
from train import train_from_self_play, save_training_data


def worker_self_play(worker_id, model_state_dict, num_games, result_queue, num_simulations, verbose=False):
    """
    Worker function for parallel self-play
    
    Args:
        worker_id: ID of this worker
        model_state_dict: State dict of neural network
        num_games: Number of games this worker should play
        result_queue: Queue to put results
        num_simulations: Number of MCTS simulations per move
        verbose: Whether to print detailed game progress
    """
    # Create model and load weights
    model = create_alpha_net()
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Move to CPU for workers (to avoid CUDA issues with multiprocessing)
    model = model.cpu()
    
    print(f"🤖 Worker {worker_id} starting {num_games} games...")
    
    games = []
    for game_idx in range(num_games):
        try:
            # Play one game
            training_examples = self_play_game(model, num_simulations=num_simulations, verbose=verbose)
            games.append(training_examples)
            
            # Progress updates
            if (game_idx + 1) % 5 == 0:
                print(f"   Worker {worker_id}: {game_idx + 1}/{num_games} games complete "
                      f"({len(training_examples)} examples in last game)")
            elif verbose:
                print(f"   Worker {worker_id}: Game {game_idx + 1} done - {len(training_examples)} examples")
        except Exception as e:
            print(f"❌ Worker {worker_id} error in game {game_idx}: {e}")
    
    # Put results in queue
    result_queue.put((worker_id, games))
    total_examples = sum(len(game) for game in games)
    print(f"✅ Worker {worker_id} finished! Generated {len(games)} games with {total_examples} total examples")


def generate_self_play_games_parallel(model, num_games, num_workers=None, num_simulations=None, verbose=False):
    """
    Generate self-play games in parallel using multiprocessing
    
    Args:
        model: Neural network model
        num_games: Total number of games to generate
        num_workers: Number of parallel workers (default: CPU count)
        num_simulations: Number of MCTS simulations per move (default from config)
        verbose: Whether to print detailed progress
        
    Returns:
        list: List of games (each game is a list of training examples)
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    if num_simulations is None:
        num_simulations = config.NUM_MCTS_SIMS
    
    print(f"\n🎮 Generating {num_games} self-play games using {num_workers} workers...")
    print(f"   {num_simulations} MCTS simulations per move")
    
    # Get model state dict
    model_state_dict = model.state_dict()
    
    # Create result queue
    result_queue = mp.Queue()
    
    # Calculate games per worker
    games_per_worker = num_games // num_workers
    remainder = num_games % num_workers
    
    # Create and start worker processes
    processes = []
    for worker_id in range(num_workers):
        # Distribute remainder games to first workers
        worker_games = games_per_worker + (1 if worker_id < remainder else 0)
        
        if worker_games > 0:
            p = mp.Process(
                target=worker_self_play,
                args=(worker_id, model_state_dict, worker_games, result_queue, num_simulations, verbose)
            )
            p.start()
            processes.append(p)
    
    # Collect results
    all_games = []
    completed_workers = 0
    total_examples = 0
    
    print(f"\n📊 Collecting results from workers...")
    
    while completed_workers < len(processes):
        try:
            worker_id, games = result_queue.get(timeout=1)
            examples = sum(len(game) for game in games)
            all_games.extend(games)
            total_examples += examples
            completed_workers += 1
            print(f"   ✓ Worker {worker_id}: {len(games)} games, {examples} examples "
                  f"({completed_workers}/{len(processes)} workers done)")
        except:
            pass
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    print(f"\n✅ Self-play complete: {len(all_games)} games, {total_examples} training examples")
    return all_games


def train_iteration_multiprocessing(model, iteration, num_self_play_games=None,
                                   num_workers=None, checkpoint_dir=None,
                                   data_dir=None, device=None, verbose=False):
    """
    Run one training iteration with multiprocessing self-play
    
    Args:
        model: Neural network model
        iteration: Current iteration number
        num_self_play_games: Number of self-play games (default from config)
        num_workers: Number of parallel workers (default: CPU count)
        checkpoint_dir: Directory to save checkpoints (default from config)
        data_dir: Directory to save training data (default from config)
        device: Device for training (default from config)
        verbose: Whether to print detailed progress
        
    Returns:
        tuple: (model, stats)
    """
    if num_self_play_games is None:
        num_self_play_games = config.NUM_SELF_PLAY_GAMES
    if checkpoint_dir is None:
        checkpoint_dir = config.CHECKPOINT_DIR
    if data_dir is None:
        data_dir = config.DATA_DIR
    if device is None:
        device = config.DEVICE
    
    print(f"\n{'='*60}")
    print(f"Iteration {iteration}")
    print(f"{'='*60}")
    
    # Generate self-play games
    print("\n[1/3] 🎮 SELF-PLAY PHASE")
    print("-" * 60)
    start_time = time.time()
    self_play_games = generate_self_play_games_parallel(
        model, num_self_play_games, num_workers=num_workers, verbose=verbose
    )
    self_play_time = time.time() - start_time
    print(f"⏱️  Self-play time: {self_play_time:.2f}s")
    
    # Save training data
    print(f"\n[2/3] 💾 SAVING DATA")
    print("-" * 60)
    os.makedirs(data_dir, exist_ok=True)
    data_filepath = os.path.join(data_dir, f"iteration_{iteration:04d}.npy")
    save_training_data(self_play_games, data_filepath)
    print(f"✅ Saved to {data_filepath}")
    
    # Train network
    print(f"\n[3/3] 🧠 TRAINING NEURAL NETWORK")
    print("-" * 60)
    start_time = time.time()
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration:04d}.pth")
    
    stats = train_from_self_play(
        model, self_play_games, 
        checkpoint_path=checkpoint_path,
        device=device
    )
    train_time = time.time() - start_time
    print(f"⏱️  Training time: {train_time:.2f}s")
    
    print(f"\n{'='*60}")
    print(f"✅ Iteration {iteration} complete!")
    print(f"⏱️  Total time: {self_play_time + train_time:.2f}s")
    print(f"{'='*60}")
    
    return model, stats


def train_pipeline_multiprocessing(num_iterations, num_self_play_games=None,
                                  num_workers=None, start_iteration=0,
                                  checkpoint_path=None, device=None):
    """
    Run full training pipeline with multiple iterations
    
    Args:
        num_iterations: Number of iterations to run
        num_self_play_games: Number of self-play games per iteration
        num_workers: Number of parallel workers
        start_iteration: Starting iteration number (for resuming)
        checkpoint_path: Path to checkpoint to load (for resuming)
        device: Device for training
        
    Returns:
        AlphaNet: Trained model
    """
    if device is None:
        device = config.DEVICE
        
    # Create or load model
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model, _, loaded_iteration = load_checkpoint(checkpoint_path, device)
        start_iteration = loaded_iteration + 1
    else:
        print("Creating new model")
        model = create_alpha_net()
        model.to(device)
    
    # Run iterations
    for iteration in range(start_iteration, start_iteration + num_iterations):
        model, stats = train_iteration_multiprocessing(
            model, iteration, 
            num_self_play_games=num_self_play_games,
            num_workers=num_workers,
            device=device
        )
    
    print("\nTraining pipeline complete!")
    return model


if __name__ == "__main__":
    # Test multiprocessing self-play
    print("Testing multiprocessing self-play")
    
    # Set start method for multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Create model
    model = create_alpha_net()
    model.eval()
    
    # Test with small number of games and workers
    print("\nGenerating 4 games with 2 workers...")
    games = generate_self_play_games_parallel(model, num_games=4, num_workers=2, num_simulations=50)
    
    print(f"\nGenerated {len(games)} games")
    if len(games) > 0:
        print(f"First game has {len(games[0])} positions")
        print("Test successful!")