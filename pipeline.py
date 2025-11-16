"""Pipeline orchestrator for AlphaZero chess training"""

import torch
import torch.multiprocessing as mp
import os
import time
from datetime import datetime
import json

import config
from alpha_net import create_alpha_net, save_checkpoint, load_checkpoint
from train_multiprocessing import train_iteration_multiprocessing, generate_self_play_games_parallel
from train import train_from_self_play, load_training_data
from evaluator import Arena, evaluate_checkpoints
from MCTS_chess import self_play_game


class AlphaZeroPipeline:
    """Main pipeline for AlphaZero training"""
    
    def __init__(self, checkpoint_dir=None, data_dir=None, log_dir=None, device=None, verbose=False):
        """
        Initialize pipeline
        
        Args:
            checkpoint_dir: Directory for checkpoints (default from config)
            data_dir: Directory for training data (default from config)
            log_dir: Directory for logs (default from config)
            device: Device for training (default from config)
            verbose: Whether to print detailed progress
        """
        self.checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
        self.data_dir = data_dir or config.DATA_DIR
        self.log_dir = log_dir or config.LOG_DIR
        self.device = device or config.DEVICE
        self.verbose = verbose
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.best_model_path = None
        self.current_iteration = 0
        
        # Training history
        self.history = {
            'iterations': [],
            'training_stats': [],
            'evaluation_results': []
        }
    
    def initialize_model(self, checkpoint_path=None):
        """
        Initialize or load model
        
        Args:
            checkpoint_path: Path to checkpoint to load (optional)
        """
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\n📁 Loading model from {checkpoint_path}")
            self.model, _, self.current_iteration = load_checkpoint(checkpoint_path, self.device)
            self.best_model_path = checkpoint_path
            print(f"✅ Loaded model from iteration {self.current_iteration}")
        else:
            print("\n🆕 Creating new model")
            self.model = create_alpha_net()
            self.model.to(self.device)
            self.current_iteration = 0
            
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"   📊 Model parameters: {param_count:,}")
            print(f"   🏗️  Architecture: {config.NUM_RESIDUAL_BLOCKS} residual blocks, {config.NUM_FILTERS} filters")
            
            # Save initial model
            initial_path = os.path.join(self.checkpoint_dir, "checkpoint_iter_0000.pth")
            optimizer = torch.optim.Adam(self.model.parameters(),
                                        lr=config.LEARNING_RATE,
                                        weight_decay=config.WEIGHT_DECAY)
            save_checkpoint(self.model, optimizer, 0, initial_path)
            self.best_model_path = initial_path
            print(f"✅ Saved initial model to {initial_path}")
    
    def run_iteration(self, iteration, num_self_play_games=None, num_workers=None,
                     enable_evaluation=False, eval_frequency=5):
        """
        Run a single training iteration
        
        Args:
            iteration: Current iteration number
            num_self_play_games: Number of self-play games (default from config)
            num_workers: Number of parallel workers (default: CPU count)
            enable_evaluation: Whether to evaluate against previous best
            eval_frequency: Evaluate every N iterations
            
        Returns:
            dict: Iteration results
        """
        if num_self_play_games is None:
            num_self_play_games = config.NUM_SELF_PLAY_GAMES
        
        print(f"\n{'='*70}")
        print(f"🔄 ITERATION {iteration}")
        print(f"{'='*70}")
        
        iteration_start = time.time()
        
        # Step 1: Generate self-play games
        print(f"\n[Step 1/3] 🎮 SELF-PLAY GENERATION")
        print(f"{'─'*70}")
        self_play_start = time.time()
        
        self_play_games = generate_self_play_games_parallel(
            self.model, num_self_play_games, num_workers=num_workers, verbose=self.verbose
        )
        
        self_play_time = time.time() - self_play_start
        total_examples = sum(len(game) for game in self_play_games)
        print(f"\n✅ Generated {len(self_play_games)} games with {total_examples} examples in {self_play_time:.2f}s")
        
        # Save training data
        data_file = os.path.join(self.data_dir, f"iteration_{iteration:04d}.npy")
        import numpy as np
        np.save(data_file, self_play_games)
        print(f"💾 Saved training data to {data_file}")
        
        # Step 2: Train neural network
        print(f"\n[Step 2/3] 🧠 NEURAL NETWORK TRAINING")
        print(f"{'─'*70}")
        train_start = time.time()
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_iter_{iteration:04d}.pth")
        stats = train_from_self_play(
            self.model, self_play_games,
            checkpoint_path=checkpoint_path,
            device=self.device
        )
        
        train_time = time.time() - train_start
        print(f"\n✅ Training completed in {train_time:.2f}s")
        print(f"   📉 Final policy loss: {stats['policy_losses'][-1]:.4f}")
        print(f"   📉 Final value loss: {stats['value_losses'][-1]:.4f}")
        print(f"   📉 Final total loss: {stats['total_losses'][-1]:.4f}")
        
        # Step 3: Evaluation (optional)
        should_update_best = True
        eval_results = None
        
        if enable_evaluation and iteration > 0 and iteration % eval_frequency == 0:
            print(f"\n[Step 3/3] ⚔️  MODEL EVALUATION")
            print(f"{'─'*70}")
            eval_start = time.time()
            
            try:
                # Load current checkpoint
                current_model, _, _ = load_checkpoint(checkpoint_path, self.device)
                
                # Load best model
                best_model, _, _ = load_checkpoint(self.best_model_path, self.device)
                
                # Evaluate
                arena = Arena(current_model, best_model)
                should_update_best = arena.should_replace_model()
                
                eval_results = {
                    'win_rate': arena.evaluate(config.EVAL_GAMES)['model1_win_rate'],
                    'should_replace': should_update_best
                }
                
                eval_time = time.time() - eval_start
                print(f"\n⏱️  Evaluation time: {eval_time:.2f}s")
                
                if should_update_best:
                    self.best_model_path = checkpoint_path
                    print(f"🏆 Updated best model to iteration {iteration}")
                else:
                    best_iter = self.best_model_path.split('_')[-1].replace('.pth', '')
                    print(f"🥈 Best model remains from iteration {best_iter}")
            
            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
                print("   Skipping evaluation, keeping current model")
        else:
            self.best_model_path = checkpoint_path
            print(f"\n[Step 3/3] ⏭️  Evaluation skipped (early training phase)")
        
        # Record iteration results
        iteration_time = time.time() - iteration_start
        
        results = {
            'iteration': iteration,
            'num_games': len(self_play_games),
            'self_play_time': self_play_time,
            'train_time': train_time,
            'total_time': iteration_time,
            'training_stats': stats,
            'evaluation': eval_results,
            'best_model': self.best_model_path
        }
        
        self.history['iterations'].append(iteration)
        self.history['training_stats'].append(stats)
        if eval_results:
            self.history['evaluation_results'].append(eval_results)
        
        # Save history
        self._save_history()
        
        print(f"\n{'='*70}")
        print(f"✅ Iteration {iteration} complete!")
        print(f"⏱️  Total time: {iteration_time:.2f}s ({iteration_time/60:.1f} minutes)")
        print(f"{'='*70}")
        
        return results
    
    def run(self, num_iterations, num_self_play_games=None, num_workers=None,
            enable_evaluation=False, eval_frequency=5, resume_from=None):
        """
        Run full training pipeline
        
        Args:
            num_iterations: Number of iterations to run
            num_self_play_games: Games per iteration (default from config)
            num_workers: Number of parallel workers (default: CPU count)
            enable_evaluation: Whether to evaluate each iteration
            eval_frequency: Evaluate every N iterations
            resume_from: Checkpoint path to resume from (optional)
        """
        print("\n" + "="*70)
        print("🏁 ALPHAZERO CHESS TRAINING PIPELINE")
        print("="*70)
        print(f"📱 Device: {self.device.upper()}")
        print(f"🔢 Iterations: {num_iterations}")
        print(f"🎮 Self-play games/iteration: {num_self_play_games or config.NUM_SELF_PLAY_GAMES}")
        print(f"👥 Workers: {num_workers or 'CPU count'}")
        print(f"⚔️  Evaluation: {'Enabled' if enable_evaluation else 'Disabled'}")
        if enable_evaluation:
            print(f"📊 Eval frequency: Every {eval_frequency} iterations")
        print(f"🗣️  Verbose mode: {'ON' if self.verbose else 'OFF'}")
        print("="*70)
        
        # Initialize model
        self.initialize_model(resume_from)
        
        start_iteration = self.current_iteration
        
        # Run iterations
        for i in range(num_iterations):
            iteration = start_iteration + i + 1
            
            try:
                results = self.run_iteration(
                    iteration,
                    num_self_play_games=num_self_play_games,
                    num_workers=num_workers,
                    enable_evaluation=enable_evaluation,
                    eval_frequency=eval_frequency
                )
                
                self.current_iteration = iteration
                
            except KeyboardInterrupt:
                print("\n\n" + "="*70)
                print("⚠️  TRAINING INTERRUPTED BY USER")
                print("="*70)
                print(f"✅ Progress saved at iteration {iteration}")
                print(f"📁 Resume with: --resume {self.best_model_path}")
                print("="*70)
                break
            
            except Exception as e:
                print("\n\n" + "="*70)
                print(f"❌ ERROR in iteration {iteration}")
                print("="*70)
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                print("\n⏭️  Continuing to next iteration...")
                print("="*70)
                continue
        
        print("\n" + "="*70)
        print("🎉 TRAINING PIPELINE COMPLETE!")
        print("="*70)
        print(f"✅ Total iterations completed: {self.current_iteration}")
        print(f"🏆 Best model: {self.best_model_path}")
        print(f"📁 Checkpoints: {self.checkpoint_dir}")
        print(f"💾 Training data: {self.data_dir}")
        print(f"📊 Logs: {self.log_dir}")
        print("="*70)
    
    def _save_history(self):
        """Save training history to JSON"""
        history_file = os.path.join(self.log_dir, "training_history.json")
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    """Main entry point for training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AlphaZero Chess Training Pipeline")
    parser.add_argument("--iterations", type=int, default=10, 
                       help="Number of training iterations")
    parser.add_argument("--games", type=int, default=None,
                       help="Self-play games per iteration (default from config)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--evaluate", action="store_true",
                       help="Enable model evaluation")
    parser.add_argument("--eval-frequency", type=int, default=5,
                       help="Evaluate every N iterations")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cpu/cuda)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output for detailed progress")
    
    args = parser.parse_args()
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Create pipeline
    pipeline = AlphaZeroPipeline(device=args.device, verbose=args.verbose)
    
    # Run training
    pipeline.run(
        num_iterations=args.iterations,
        num_self_play_games=args.games,
        num_workers=args.workers,
        enable_evaluation=args.evaluate,
        eval_frequency=args.eval_frequency,
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()