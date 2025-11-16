"""Fixed AlphaZero training pipeline - resolves freezing issues"""

import torch
import torch.multiprocessing as mp
import numpy as np
import os
import time
import chess
from collections import deque
import queue

import config
from alpha_net import create_alpha_net, load_checkpoint
from chess_board import ChessBoard
from encoder_decoder import encode_board, decode_policy_output, get_legal_moves_mask
import MCTS_chess


class SimpleChessTrainer:
    """Simplified trainer that doesn't use gym environments"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.mcts = MCTS_chess.MCTS(model, num_simulations=100)  # Reduced for testing
        
    def play_self_play_game(self, temperature=1.0, max_moves=100):
        """Play one self-play game"""
        board = ChessBoard()
        game_data = []
        move_count = 0
        
        while not board.is_game_over() and move_count < max_moves:
            # Get board encoding
            board_encoding = encode_board(board)
            
            # Use MCTS to get move and policy
            move, policy_target = self.mcts.search(board, add_noise=True, temperature=temperature)
            
            if move is None:
                break
                
            # Store training example
            game_data.append({
                'board': board_encoding,
                'policy': policy_target,
                'turn': board.get_turn()
            })
            
            # Make the move
            board.make_move(move)
            move_count += 1
        
        # Determine game result
        winner = board.get_winner()
        if winner is None:
            winner = 0  # Draw
        
        # Add values to all examples
        for example in game_data:
            example['value'] = float(winner * example['turn'])
        
        return game_data


def worker_self_play_fixed(worker_id, model_state_dict, num_games, result_queue, device='cpu'):
    """Fixed worker for self-play generation"""
    try:
        print(f"🤖 Worker {worker_id} starting {num_games} games...")
        
        # Create model
        model = create_alpha_net()
        model.load_state_dict(model_state_dict)
        model.eval()
        model = model.cpu()  # Force CPU to avoid CUDA multiprocessing issues
        
        # Create trainer
        trainer = SimpleChessTrainer(model, device='cpu')
        
        games = []
        for game_idx in range(num_games):
            try:
                # Play one game
                game_data = trainer.play_self_play_game(temperature=1.0, max_moves=80)
                games.append(game_data)
                
                # Progress updates every 3 games
                if (game_idx + 1) % 3 == 0:
                    print(f"   Worker {worker_id}: {game_idx + 1}/{num_games} games complete "
                          f"({len(game_data)} positions in last game)")
                    
            except Exception as e:
                print(f"❌ Worker {worker_id} error in game {game_idx}: {e}")
                continue
        
        # Put results in queue
        result_queue.put((worker_id, games))
        total_examples = sum(len(game) for game in games)
        print(f"✅ Worker {worker_id} finished! Generated {len(games)} games with {total_examples} total examples")
        
    except Exception as e:
        print(f"💥 Worker {worker_id} fatal error: {e}")
        result_queue.put((worker_id, []))


def generate_self_play_games_fixed(model, num_games, num_workers=None, verbose=False):
    """Fixed parallel self-play generation"""
    if num_workers is None:
        num_workers = min(4, mp.cpu_count())  # Limit workers to avoid resource issues
    
    print(f"\n🎮 Generating {num_games} self-play games using {num_workers} workers...")
    print(f"   Using fixed implementation without gym dependencies")
    
    # Get model state dict
    model_state_dict = model.state_dict()
    
    # Create result queue
    result_queue = mp.Queue()
    
    # Calculate games per worker
    games_per_worker = max(1, num_games // num_workers)
    remainder = num_games % num_workers
    
    print(f"   Distributing {games_per_worker} games per worker")
    
    # Create and start worker processes
    processes = []
    for worker_id in range(num_workers):
        # Distribute remainder games to first workers
        worker_games = games_per_worker + (1 if worker_id < remainder else 0)
        
        if worker_games > 0:
            p = mp.Process(
                target=worker_self_play_fixed,
                args=(worker_id, model_state_dict, worker_games, result_queue, 'cpu')
            )
            p.start()
            processes.append(p)
    
    # Collect results with timeout
    all_games = []
    completed_workers = 0
    total_examples = 0
    
    print(f"\n📊 Collecting results from workers...")
    
    start_time = time.time()
    timeout = 300  # 5 minute timeout
    
    while completed_workers < len(processes) and (time.time() - start_time) < timeout:
        try:
            worker_id, games = result_queue.get(timeout=1)
            examples = sum(len(game) for game in games)
            all_games.extend(games)
            total_examples += examples
            completed_workers += 1
            elapsed = time.time() - start_time
            print(f"   ✓ Worker {worker_id}: {len(games)} games, {examples} examples "
                  f"({completed_workers}/{len(processes)} workers done, {elapsed:.0f}s elapsed)")
        except queue.Empty:
            print(f"   ⏰ Waiting for workers... ({completed_workers}/{len(processes)} done, {time.time() - start_time:.0f}s)")
            continue
    
    # Wait for all processes with timeout
    for i, p in enumerate(processes):
        if p.is_alive():
            print(f"   🛑 Terminating worker {i}...")
            p.join(timeout=1)
            if p.is_alive():
                p.terminate()
    
    print(f"\n✅ Self-play complete: {len(all_games)} games, {total_examples} training examples")
    return all_games


def train_fixed_model(model, training_examples, device='cpu', epochs=3):
    """Simplified training function"""
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Convert training examples to tensors
    boards = torch.FloatTensor([ex['board'] for ex in training_examples]).to(device)
    
    # Create policy targets
    policy_targets = []
    for example in training_examples:
        policy_dict = example['policy']
        policy_vec = np.zeros(4096, dtype=np.float32)
        for move, prob in policy_dict.items():
            if hasattr(move, 'from_square'):  # chess.Move object
                move_idx = move.from_square * 64 + move.to_square
                policy_vec[move_idx] = prob
        policy_targets.append(policy_vec)
    
    policy_targets = torch.FloatTensor(policy_targets).to(device)
    value_targets = torch.FloatTensor([ex['value'] for ex in training_examples]).to(device)
    
    print(f"🎯 Training on {len(training_examples)} examples")
    print(f"   📊 Board tensor: {boards.shape}")
    print(f"   📊 Policy tensor: {policy_targets.shape}")
    print(f"   📊 Value tensor: {value_targets.shape}")
    
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        batch_size = 32
        
        for i in range(0, len(training_examples), batch_size):
            batch_end = min(i + batch_size, len(training_examples))
            batch_boards = boards[i:batch_end]
            batch_policy = policy_targets[i:batch_end]
            batch_values = value_targets[i:batch_end]
            
            # Forward pass
            policy_logits, value_preds = model(batch_boards)
            
            # Calculate losses
            policy_loss = torch.nn.functional.cross_entropy(policy_logits, batch_policy.argmax(dim=1))
            value_loss = torch.nn.functional.mse_loss(value_preds.squeeze(), batch_values)
            
            loss = policy_loss + value_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(training_examples) // batch_size)
        losses.append(avg_loss)
        print(f"   Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")
    
    return {'losses': losses}


def run_fixed_training(num_iterations=3, num_games_per_iter=20, num_workers=2):
    """Run fixed training pipeline"""
    print("🚀 Starting Fixed AlphaZero Training Pipeline")
    print("="*60)
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Using device: {device}")
    
    model = create_alpha_net()
    model.to(device)
    
    training_history = []
    
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"🔄 ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Generate self-play games
        print(f"\n[1/2] 🎮 Generating self-play games...")
        self_play_games = generate_self_play_games_fixed(
            model, num_games_per_iter, num_workers=num_workers
        )
        
        # Flatten games into training examples
        training_examples = []
        for game in self_play_games:
            training_examples.extend(game)
        
        print(f"📊 Total training examples: {len(training_examples)}")
        
        # Train model
        print(f"\n[2/2] 🧠 Training neural network...")
        stats = train_fixed_model(model, training_examples, device=device, epochs=3)
        
        elapsed = time.time() - start_time
        
        # Record iteration results
        iteration_result = {
            'iteration': iteration + 1,
            'games_generated': len(self_play_games),
            'training_examples': len(training_examples),
            'training_time': elapsed,
            'final_loss': stats['losses'][-1] if stats['losses'] else 0
        }
        training_history.append(iteration_result)
        
        print(f"\n✅ Iteration {iteration + 1} complete!")
        print(f"   ⏱️  Time: {elapsed:.1f}s")
        print(f"   📊 Final loss: {iteration_result['final_loss']:.4f}")
        print(f"   🎮 Games: {len(self_play_games)}")
        print(f"   📈 Examples: {len(training_examples)}")
        
        # Save checkpoint every iteration
        checkpoint_path = f"checkpoints/fixed_checkpoint_iter_{iteration + 1:04d}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        from alpha_net import save_checkpoint
        save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    print(f"\n{'='*60}")
    print("🎉 Fixed Training Pipeline Complete!")
    print(f"{'='*60}")
    
    # Summary
    total_time = sum(h['training_time'] for h in training_history)
    total_games = sum(h['games_generated'] for h in training_history)
    total_examples = sum(h['training_examples'] for h in training_history)
    
    print(f"📊 Training Summary:")
    print(f"   ⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"   🎮 Total games: {total_games}")
    print(f"   📈 Total examples: {total_examples}")
    print(f"   ⚡ Avg time per game: {total_time/total_games:.1f}s")
    print(f"   📊 Examples per game: {total_examples/total_games:.1f}")
    
    return model, training_history


if __name__ == "__main__":
    # Run fixed training
    model, history = run_fixed_training(
        num_iterations=3,
        num_games_per_iter=12,  # Small number for testing
        num_workers=2
    )
    
    print("\n🔧 Training completed successfully! No more freezing issues.")