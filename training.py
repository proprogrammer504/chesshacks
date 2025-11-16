import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from gym_chess.alphazero.move_encoding import MoveEncoding
from gym_chess.alphazero.board_encoding import BoardEncoding, BoardHistory
from collections import defaultdict
import queue
import threading
import time
import json
from pathlib import Path

np.int = np.int_
np.float = np.float64
np.bool = np.bool_


class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class DeepResNetChess(nn.Module):
    """Deeper residual network for chess (AlphaZero-style)"""
    def __init__(self, input_channels=21, num_actions=4672, num_res_blocks=10, num_filters=256):
        super().__init__()
        
        # Initial convolution
        self.conv_input = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, num_actions)
        
        # Value head
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Handle different input formats
        if x.dim() == 4 and x.shape[1] == 8:
            x = x.permute(0, 3, 1, 2)
        
        # Initial conv
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        policy = self.policy_fc(p)
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy, value


class OptimizedGPUBatchedEvaluator:
    """GPU-accelerated evaluator with aggressive optimizations"""
    def __init__(self, network, device='cuda', batch_size=256, max_wait_time=0.001):
        self.network = network.to(device)
        self.network.eval()
        self.device = device
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        
        # OPTIMIZATION: Mixed precision and cudnn benchmark
        if device == 'cuda':
            self.dtype = torch.float16  # 2x speedup
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True  # Hardware acceleration
        else:
            self.dtype = torch.float32
        
        self.request_queue = queue.Queue()
        self.running = True
        
        # OPTIMIZATION: More workers for better GPU utilization
        self.num_workers = 6 if device == 'cuda' else 2
        self.batch_threads = []
        for i in range(self.num_workers):
            thread = threading.Thread(target=self._batch_worker, daemon=True)
            thread.start()
            self.batch_threads.append(thread)
        
        # OPTIMIZATION: Pre-allocated tensors to avoid repeated allocation
        self._init_memory_pool()
    
    def _init_memory_pool(self):
        """Pre-allocate memory pools for faster tensor operations"""
        if self.device == 'cuda':
            # Pre-allocate GPU memory pools
            torch.cuda.empty_cache()
            # Create dummy tensor to warm up GPU
            dummy = torch.zeros(1, dtype=self.dtype, device=self.device)
            del dummy
    
    def evaluate(self, board_state):
        result_queue = queue.Queue()
        self.request_queue.put((board_state, result_queue))
        policy, value = result_queue.get()
        return policy, value
    
    def evaluate_batch(self, board_states):
        if len(board_states) == 0:
            return [], []
        
        with torch.no_grad():
            # OPTIMIZATION: Pre-allocated tensor with mixed precision
            states_tensor = torch.tensor(np.array(board_states), dtype=self.dtype, device=self.device)
            
            # OPTIMIZATION: Mixed precision evaluation with autocast
            if self.device == 'cuda' and self.dtype == torch.float16:
                with torch.amp.autocast('cuda'):
                    logits, values = self.network(states_tensor)
                    policies = F.softmax(logits, dim=-1)
            else:
                logits, values = self.network(states_tensor)
                policies = F.softmax(logits, dim=-1)
            
            return policies.cpu().numpy(), values.cpu().numpy()
    
    def _batch_worker(self):
        """Worker that dynamically batches requests for GPU efficiency"""
        while self.running:
            batch = []
            result_queues = []
            
            start_time = time.time()
            
            while len(batch) < self.batch_size:
                elapsed = time.time() - start_time
                remaining_time = max(0.0005, self.max_wait_time - elapsed)
                
                try:
                    request = self.request_queue.get(timeout=remaining_time)
                    
                    if request is None:
                        self.request_queue.put(None)
                        return
                    
                    state, result_q = request
                    batch.append(state)
                    result_queues.append(result_q)
                    
                    if len(batch) >= self.batch_size:
                        break
                        
                except queue.Empty:
                    if batch:
                        break
                    if not self.running:
                        return
                    continue
            
            if batch and self.running:
                try:
                    policies, values = self.evaluate_batch(batch)
                    for i, result_q in enumerate(result_queues):
                        result_q.put((policies[i], values[i].item()))
                except:
                    for result_q in result_queues:
                        result_q.put((None, None))
    
    def shutdown(self):
        self.running = False
        for _ in range(self.num_workers):
            self.request_queue.put(None)
        for thread in self.batch_threads:
            thread.join(timeout=1.0)


class OptimizedFixedMCTS:
    """Highly optimized MCTS with proper move validation and aggressive performance enhancements"""
    def __init__(self, evaluator, num_simulations=160, c_puct=1.5, 
                 top_k_initial=8, prune_threshold=0.08,
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
                 parallel_sims=32):
        self.evaluator = evaluator
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.top_k_initial = top_k_initial
        self.prune_threshold = prune_threshold
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.parallel_sims = parallel_sims

        self_play_env = gym.make("Chess-v0")
        self.board_encoder = BoardEncoding(self_play_env, history_length=1)
        self.move_encoder = MoveEncoding(self_play_env)
        
        # OPTIMIZATION: Transposition table with LRU eviction
        self.transposition_cache = {}
        self.cache_size_limit = 100000
        self.cache_hits = 0
        self.cache_misses = 0
        
        # OPTIMIZATION: Enhanced encoding cache with compression
        self.encoding_cache = {}
        self.encoding_cache_limit = 50000

        self.position_history = {}
        
        # OPTIMIZATION: Pre-computed patterns for faster evaluation
        self._init_optimization_patterns()
        
        # OPTIMIZATION: Move ordering heuristics
        self.capture_moves = []
        self.check_moves = []
        self.developing_moves = []
    
    def _init_optimization_patterns(self):
        """Initialize optimization patterns for faster move evaluation"""
        # Pre-compute material values for move ordering
        self.piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
        }
        
        # Pre-compute position evaluation patterns
        self.center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        self.developing_pieces = [chess.KNIGHT, chess.BISHOP, chess.QUEEN]
    
    def search(self, board, temperature=1.0):
        root = OptimizedFixedMCTSNode(board.copy())
        
        # OPTIMIZATION: Cache-aware search
        cache_key = self._get_cache_key(root.board)
        if cache_key in self.transposition_cache:
            cached_result = self.transposition_cache[cache_key]
            self.cache_hits += 1
            return cached_result['move'], cached_result['policy']
        
        self.encoding_cache.clear()
        
        self._update_position_history(root.board)
        
        # OPTIMIZATION: Faster root expansion
        self._fast_expand_root(root)
        
        num_batches = max(1, self.num_simulations // self.parallel_sims)
        
        # OPTIMIZATION: Parallel batched simulations
        for batch_idx in range(num_batches):
            simulation_data = []
            
            for _ in range(self.parallel_sims):
                node = root
                search_path = [node]
                
                # OPTIMIZATION: Fast selection with move ordering
                while node.children and not node.board.is_game_over():
                    node = self._fast_select_child(node)
                    search_path.append(node)
                
                if not node.board.is_game_over():
                    if self._should_fast_expand(node):
                        simulation_data.append(('expand', node, search_path))
                    else:
                        simulation_data.append(('evaluate', node, search_path))
                else:
                    value = self._get_terminal_value(node.board)
                    self._fast_backpropagate(search_path, value)
            
            # OPTIMIZATION: Batch processing with cache awareness
            self._batch_process_simulation_data(simulation_data)
            
            # OPTIMIZATION: Adaptive pruning
            if (batch_idx + 1) % 5 == 0:
                self._aggressive_prune(root)
        
        best_move = self._get_best_move(root, temperature)
        action_probs = self._get_action_probs(root)
        
        # OPTIMIZATION: Cache results
        self.transposition_cache[cache_key] = {
            'move': best_move,
            'policy': action_probs,
            'visit_count': root.visit_count
        }
        
        # Cache management
        if len(self.transposition_cache) > self.cache_size_limit:
            # Remove least recently used entries
            keys_to_remove = list(self.transposition_cache.keys())[:self.cache_size_limit // 4]
            for key in keys_to_remove:
                del self.transposition_cache[key]
        
        return best_move, action_probs
    
    def _fast_expand_root(self, node):
        """Optimized root expansion with move ordering"""
        board = node.board
        legal_moves = list(board.legal_moves)
        
        if len(legal_moves) == 0:
            return
        
        obs = self._encode_board_fast(board)
        policy, value = self.evaluator.evaluate(obs)
        
        self._expand_with_optimized_move_ordering(node, policy)
    
    def _fast_select_child(self, node):
        """Optimized child selection with UCT improvements"""
        best_score = -float('inf')
        best_child = None
        
        for child in node.children.values():
            ucb = child.ucb_score(node.visit_count, self.c_puct)
            if ucb > best_score:
                best_score = ucb
                best_child = child
        
        if best_child:
            best_child.virtual_loss += 1
        
        return best_child
    
    def _should_fast_expand(self, node):
        """Optimized expansion decision"""
        if node.visit_count < 2:
            return True
        if node.parent and node.q_value() < -0.9:
            return False
        return True
    
    def _fast_backpropagate(self, search_path, value):
        """Optimized backpropagation"""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            node.virtual_loss = max(0, node.virtual_loss - 1)
            value = -value
    
    def _expand_with_optimized_move_ordering(self, node, policy):
        """Enhanced move ordering for faster convergence with proper validation"""
        board = node.board
        legal_moves = list(board.legal_moves)
        
        if len(legal_moves) == 0:
            return
        
        self.move_encoder.unwrapped._board = board
        
        move_scores = []
        for move in legal_moves:
            # CRITICAL: Validate move is still legal
            if not board.is_legal(move):
                continue
                
            move_idx = self.move_encoder.encode(move)
            score = policy[move_idx] if move_idx < len(policy) else 1.0 / len(legal_moves)
            
            # OPTIMIZATION: Fast move ordering
            score = self._apply_move_ordering_boosts(board, move, score)
            
            # Repetition handling
            test_board = board.copy()
            test_board.push(move)
            
            repetition_count = self._count_repetitions(test_board)
            if repetition_count >= 2:
                score *= 0.00001
            elif repetition_count == 1:
                score *= 0.1
            
            move_scores.append((move, score))
        
        # OPTIMIZATION: Fast sorting and selection
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        num_to_expand = min(
            len(move_scores),
            self.top_k_initial + int(np.sqrt(node.visit_count))
        )
        
        top_moves = move_scores[:num_to_expand]
        
        # OPTIMIZATION: Efficient normalization
        total_score = sum(score for _, score in top_moves)
        if total_score > 1e-6:
            top_moves = [(move, score / total_score) for move, score in top_moves]
        else:
            uniform_prob = 1.0 / len(top_moves)
            top_moves = [(move, uniform_prob) for move, _ in top_moves]
        
        # Root node noise
        if node.parent is None:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(top_moves))
            for i, (move, prior) in enumerate(top_moves):
                noisy_prior = (1 - self.dirichlet_epsilon) * prior + self.dirichlet_epsilon * noise[i]
                
                # CRITICAL: Validate move before creating child
                if board.is_legal(move):
                    child_board = board.copy()
                    child_board.push(move)
                    node.children[move] = OptimizedFixedMCTSNode(child_board, parent=node, prior=noisy_prior, move=move)
        else:
            for move, prior in top_moves:
                # CRITICAL: Validate move before creating child
                if board.is_legal(move):
                    child_board = board.copy()
                    child_board.push(move)
                    node.children[move] = OptimizedFixedMCTSNode(child_board, parent=node, prior=prior, move=move)
    
    def _apply_move_ordering_boosts(self, board, move, score):
        """Apply move ordering heuristics for faster convergence"""
        # OPTIMIZATION: Capture moves get priority
        if board.is_capture(move):
            score *= 2.0
        
        # OPTIMIZATION: Check moves get priority
        test_board = board.copy()
        test_board.push(move)
        if test_board.is_check():
            score *= 1.5
        
        # OPTIMIZATION: Development moves
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.QUEEN]:
            if move.to_square in self.center_squares:
                score *= 1.2
        
        return score
    
    def _batch_process_simulation_data(self, simulation_data):
        """Optimized batch processing"""
        if not simulation_data:
            return
        
        expand_data = [(node, path) for action_type, node, path in simulation_data if action_type == 'expand']
        eval_data = [(node, path) for action_type, node, path in simulation_data if action_type == 'evaluate']
        
        all_nodes = [node for node, _ in expand_data + eval_data]
        
        if all_nodes:
            board_states = [self._encode_board_fast(node.board) for node in all_nodes]
            
            policies, values = self.evaluator.evaluate_batch(board_states)
            
            for i, (node, path) in enumerate(expand_data):
                policy = policies[i]
                value = values[i].item()
                self._expand_with_optimized_move_ordering(node, policy)
                self._fast_backpropagate(path, value)
            
            offset = len(expand_data)
            for i, (node, path) in enumerate(eval_data):
                value = values[offset + i].item()
                self._fast_backpropagate(path, value)
    
    def _aggressive_prune(self, node):
        """Aggressive pruning for performance"""
        if not node.children or node.visit_count < 25:
            return
        
        # More aggressive pruning
        sorted_children = sorted(node.children.items(), key=lambda x: x[1].visit_count, reverse=True)
        top_moves = sorted_children[:max(3, len(sorted_children) // 4)]
        
        node.children = dict(top_moves)
    
    def _encode_board_fast(self, board):
        """Optimized board encoding"""
        board_fen = board.fen()
        if board_fen in self.encoding_cache:
            return self.encoding_cache[board_fen]
        
        self.board_encoder.unwrapped._board = board
        self.board_encoder._history.reset()
        obs = self.board_encoder.observation(board)
        
        # Cache management
        if len(self.encoding_cache) > self.encoding_cache_limit:
            # Remove oldest entries
            keys_to_remove = list(self.encoding_cache.keys())[:self.encoding_cache_limit // 2]
            for key in keys_to_remove:
                del self.encoding_cache[key]
        
        self.encoding_cache[board_fen] = obs
        return obs
    
    def _get_cache_key(self, board):
        """Generate cache key for position"""
        return board.fen().split(' ')[0]  # Just the board position
    
    def _update_position_history(self, board):
        """Update position history from board's move stack"""
        fen = self._get_cache_key(board)
        self.position_history[fen] = self.position_history.get(fen, 0) + 1
    
    def _count_repetitions(self, board):
        """Count how many times this position has occurred"""
        fen = self._get_cache_key(board)
        return self.position_history.get(fen, 0)
    
    def _get_terminal_value(self, board):
        """Get terminal position value"""
        result = board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return -0.3
    
    def _get_best_move(self, root, temperature=0.0, epsilon=0.2):
        """Optimized best move selection with move validation"""
        if not root.children:
            return None

        moves = list(root.children.keys())
        
        # CRITICAL: Filter only legal moves from current position
        legal_moves = [move for move in moves if root.board.is_legal(move)]
        
        if not legal_moves:
            # Fallback: find any legal move
            legal_moves = list(root.board.legal_moves)
            if not legal_moves:
                return None
        
        visits = np.array([root.children[m].visit_count for m in legal_moves])

        if np.random.random() < epsilon:
            # Temperature-based selection
            visits_temp = visits ** 0.5
            probs = visits_temp / visits_temp.sum()
            return np.random.choice(legal_moves, p=probs)
        else:
            if temperature == 0:
                return legal_moves[np.argmax(visits)]
            else:
                visits = visits ** (1.0 / temperature)
                probs = visits / visits.sum()
                return np.random.choice(legal_moves, p=probs)
    
    def _get_action_probs(self, root):
        """Get action probabilities with move validation"""
        if not root.children:
            return {}
        
        self.move_encoder.unwrapped._board = root.board
        
        visits_sum = sum(child.visit_count for child in root.children.values())
        if visits_sum == 0:
            return {}
            
        action_probs = {}
        for move, child in root.children.items():
            # CRITICAL: Only include legal moves
            if root.board.is_legal(move):
                action_int = self.move_encoder.encode(move)
                action_probs[action_int] = child.visit_count / visits_sum
            
        return action_probs


class OptimizedFixedMCTSNode:
    """Optimized MCTS node with efficient memory layout"""
    __slots__ = ['board', 'parent', 'children', 'visit_count', 'value_sum', 
                 'prior', 'move', 'virtual_loss']
    
    def __init__(self, board, parent=None, prior=0, move=None):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.move = move
        self.virtual_loss = 0
        
    def q_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visits, c_puct=1.5):
        effective_visits = self.visit_count + self.virtual_loss
        q = self.q_value()
        u = c_puct * self.prior * np.sqrt(parent_visits) / (1 + effective_visits)
        return q + u


def parallel_self_play_worker_optimized_fixed(worker_id, network_state_dict, num_games, result_queue, net_config, device='cpu', move_limit=80):
    """Optimized self-play worker with aggressive performance enhancements and proper move validation"""
    try:
        env = gym.make("Chess-v0")
        env = BoardEncoding(env, history_length=1)
        env = MoveEncoding(env)
        
        network = DeepResNetChess(net_config['input_channels'], net_config['num_actions'], 
                                   net_config['num_res_blocks'], net_config['num_filters'])
        network.load_state_dict(network_state_dict)
        network.eval()
        
        # OPTIMIZATION: Larger batch size and more parallel simulations
        batch_size = 256 if device == 'cuda' else 64
        max_wait_time = 0.001
        evaluator = OptimizedGPUBatchedEvaluator(network, device=device, batch_size=batch_size, max_wait_time=max_wait_time)
        
        # OPTIMIZATION: Many more parallel simulations
        parallel_sims = 64 if device == 'cuda' else 16
        mcts = OptimizedFixedMCTS(evaluator, num_simulations=80, parallel_sims=parallel_sims,
                           top_k_initial=6, prune_threshold=0.1,
                           dirichlet_alpha=0.3, dirichlet_epsilon=0.2)
        
        worker_data = []
        game_statistics = []
        
        for game_num in range(num_games):
            encoded_state = env.reset()
            game_data = []
            move_count = 0
            
            mcts.position_history.clear()
            
            game_end_reason = "ongoing"
            game_start_time = time.time()
            
            while not env.unwrapped.unwrapped._board.is_game_over() and move_count < move_limit:
                board = env.unwrapped.unwrapped._board.copy()
                
                # OPTIMIZATION: Skip some checks for speed
                if move_count > 0 and (move_count % 10 == 0):
                    if board.can_claim_threefold_repetition():
                        game_end_reason = "threefold_repetition"
                        outcome = -0.3
                        break
                
                # OPTIMIZATION: Simplified temperature schedule
                if move_count < 15:
                    temperature = 1.0
                    epsilon = 0.25
                elif move_count < 30:
                    temperature = 0.5
                    epsilon = 0.15
                else:
                    temperature = 0.1
                    epsilon = 0.05
                
                move, action_probs = mcts.search(board, temperature)
                
                if move is None:
                    game_end_reason = "no_legal_moves"
                    outcome = -0.3
                    break
                
                # CRITICAL: Final move validation before applying
                if not board.is_legal(move):
                    print(f"Worker {worker_id}: ERROR - MCTS proposed illegal move {move}")
                    game_end_reason = "illegal_move"
                    outcome = -0.3
                    break
                
                # OPTIMIZATION: Fast step penalty calculation
                step_penalty = -0.05 * move_count
                game_data.append((encoded_state, action_probs, None, step_penalty))
                
                action_int = env.encode(move)
                
                encoded_state, _, done, _ = env.step(action_int)
                move_count += 1
            
            # Determine final outcome
            board = env.unwrapped.unwrapped._board.copy()
            game_end_time = time.time()
            game_duration = game_end_time - game_start_time
            
            if game_end_reason == "ongoing":
                if move_count >= move_limit:
                    outcome = -0.4
                    game_end_reason = "move_limit_timeout"
                else:
                    result = board.result()
                    if result == "1-0":
                        outcome = 1.0
                        game_end_reason = "white_wins"
                    elif result == "0-1":
                        outcome = -1.0
                        game_end_reason = "black_wins"
                    else:
                        outcome = -0.3
                        game_end_reason = "draw"
            
            # OPTIMIZATION: Fast outcome assignment
            for i, (state, probs, _, step_penalty) in enumerate(game_data):
                player_outcome = outcome if i % 2 == 0 else -outcome
                final_value = player_outcome + step_penalty
                worker_data.append((state, probs, final_value))
            
            # Collect detailed game statistics
            game_stats = {
                'game_number': game_num + 1,
                'moves_played': move_count,
                'game_duration': game_duration,
                'end_reason': game_end_reason,
                'outcome': outcome,
                'positions_collected': len(game_data),
                'moves_per_second': move_count / game_duration if game_duration > 0 else 0
            }
            game_statistics.append(game_stats)
            
            if (game_num + 1) % 5 == 0 or game_num == num_games - 1:
                print(f"Worker {worker_id}: Game {game_num + 1}/{num_games} completed")
                print(f"  Result: {game_end_reason}, Moves: {move_count}, Duration: {game_duration:.1f}s")
        
        # Print worker summary statistics
        if game_statistics:
            total_games = len(game_statistics)
            avg_moves = np.mean([stats['moves_played'] for stats in game_statistics])
            avg_duration = np.mean([stats['game_duration'] for stats in game_statistics])
            avg_mps = np.mean([stats['moves_per_second'] for stats in game_statistics])
            
            # Count game results
            result_counts = {}
            for stats in game_statistics:
                result = stats['end_reason']
                result_counts[result] = result_counts.get(result, 0) + 1
            
            print(f"Worker {worker_id} Summary:")
            print(f"  Total games: {total_games}")
            print(f"  Average moves per game: {avg_moves:.1f}")
            print(f"  Average game duration: {avg_duration:.1f}s")
            print(f"  Average moves per second: {avg_mps:.1f}")
            print(f"  Game results: {result_counts}")
        
        evaluator.shutdown()
        
        if device == 'cuda':
            del network
            del evaluator
            torch.cuda.empty_cache()
        
        env.close()
        
        result_queue.put(worker_data)
        
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put([])


def compute_metrics(network, data_sample, device):
    """Compute diagnostics: value error, policy entropy, accuracy"""
    if len(data_sample) == 0:
        return {}
    
    network.eval()
    with torch.no_grad():
        states = torch.FloatTensor([s for s, _, _ in data_sample]).to(device)
        target_values = torch.FloatTensor([v for _, _, v in data_sample]).to(device)
        
        logits, pred_values = network(states)
        pred_values = pred_values.squeeze()
        pred_policies = F.softmax(logits, dim=-1)
        
        value_mse = F.mse_loss(pred_values, target_values).item()
        value_mae = torch.mean(torch.abs(pred_values - target_values)).item()
        
        policy_entropy = -torch.mean(torch.sum(pred_policies * torch.log(pred_policies + 1e-8), dim=-1)).item()
        
        policy_targets = torch.zeros_like(pred_policies)
        for i, (_, target_probs, _) in enumerate(data_sample):
            for move_idx, prob in target_probs.items():
                if move_idx < policy_targets.shape[1]:
                    policy_targets[i, move_idx] = prob
        
        pred_top_moves = torch.argmax(pred_policies, dim=-1)
        target_top_moves = torch.argmax(policy_targets, dim=-1)
        policy_accuracy = (pred_top_moves == target_top_moves).float().mean().item()
        
    network.train()
    return {
        'value_mse': value_mse,
        'value_mae': value_mae,
        'policy_entropy': policy_entropy,
        'policy_accuracy': policy_accuracy
    }


def train_with_optimized_fixed_parallel_mcts(iterations=50, games_per_iter=200, num_workers=None,
                                       num_res_blocks=10, num_filters=256):
    """Optimized training function with aggressive performance improvements and fixed move validation"""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count())
    
    print(f"Training with {num_workers} parallel workers for maximum performance")
    print(f"Target: {games_per_iter} games per iteration")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    env = gym.make("Chess-v0")
    env = BoardEncoding(env, history_length=1)
    env = MoveEncoding(env)
    
    input_channels = env.observation_space.shape[2]
    num_actions = env.action_space.n
    
    net_config = {
        'input_channels': input_channels,
        'num_actions': num_actions,
        'num_res_blocks': num_res_blocks,
        'num_filters': num_filters
    }
    env.close()
    
    print(f"Network Config: Input={input_channels}ch, Actions={num_actions}, ResBlocks={num_res_blocks}, Filters={num_filters}")
    
    network = DeepResNetChess(input_channels, num_actions, num_res_blocks, num_filters)
    if device == 'cuda':
        network = network.cuda()
    
    optimizer = torch.optim.SGD(
        network.parameters(),
        lr=0.1,  # Will be adjusted by schedule
        momentum=0.9,
        weight_decay=1e-4
    )
    replay_buffer = []
    
    def get_learning_rate(iteration):
        """Learning rate schedule for stable training"""
        if iteration < 5:
            return 0.1
        elif iteration < 10:
            return 0.01
        elif iteration < 20:
            return 0.002
        elif iteration < 30:
            return 0.0002
        else:
            return 0.00002
    
    log_file = Path("training_log_combined.jsonl")
    
    for iteration in range(iterations):
        print(f"\n{'='*60}")
        print(f"OPTIMIZED + FIXED TRAINING ITERATION {iteration + 1}/{iterations}")
        print(f"{'='*60}")

        # Update learning rate
        current_lr = get_learning_rate(iteration)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        print(f"Learning rate set to: {current_lr}")

        # OPTIMIZATION: Shorter move limits for faster games
        move_limit = max(60, 100 - iteration)
        print(f"Move limit for this iteration: {move_limit}")

        network_state = {k: v.cpu() for k, v in network.state_dict().items()}
        
        games_per_worker = max(1, games_per_iter // num_workers)
        print(f"Each of {num_workers} workers will play {games_per_worker} games")
        
        result_queue = mp.Queue()
        processes = []
        
        worker_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        start_time = time.time()
        for worker_id in range(num_workers):
            p = mp.Process(
                target=parallel_self_play_worker_optimized_fixed,
                args=(worker_id, network_state, games_per_worker, result_queue, net_config, worker_device, move_limit)
            )
            p.start()
            processes.append(p)
        
        iteration_data = []
        completed_workers = 0
        timeout_per_game = 120  # Reduced timeout
        all_game_stats = []

        while completed_workers < num_workers:
            try:
                worker_data = result_queue.get(timeout=timeout_per_game)
                iteration_data.extend(worker_data)
                completed_workers += 1
                elapsed = time.time() - start_time
                print(f"Worker {completed_workers}/{num_workers} completed ({elapsed:.0f}s elapsed, {len(worker_data)} positions)")
            except queue.Empty:
                print(f"Worker timeout after {time.time() - start_time:.0f}s")
                break

        for p in processes:
            if p.is_alive():
                p.join(timeout=30)
                if p.is_alive():
                    p.terminate()

        
        self_play_time = time.time() - start_time
        
        # Calculate detailed performance statistics
        if iteration_data:
            total_positions = len(iteration_data)
            total_games = len(iteration_data) // 50  # Approximate
            games_per_second = total_games / self_play_time if self_play_time > 0 else 0
            positions_per_second = total_positions / self_play_time if self_play_time > 0 else 0
        else:
            total_positions = 0
            total_games = 0
            games_per_second = 0
            positions_per_second = 0
        
        print(f"\nSelf-play performance summary:")
        print(f"  Total self-play time: {self_play_time:.1f} seconds")
        print(f"  Games completed: {total_games}")
        print(f"  Positions collected: {total_positions}")
        print(f"  Games per second: {games_per_second:.2f}")
        print(f"  Positions per second: {positions_per_second:.0f}")
        print(f"  Average positions per game: {total_positions // max(1, total_games)}")
        
        replay_buffer.extend(iteration_data)
        max_buffer_size = 150000  # Increased buffer size
        if len(replay_buffer) > max_buffer_size:
            replay_buffer = replay_buffer[-max_buffer_size:]
        
        print(f"Replay buffer size: {len(replay_buffer)} positions")
        
        if len(replay_buffer) == 0:
            print("No data in replay buffer, skipping training.")
            continue
        
        sample_size = min(2000, len(replay_buffer))
        sample_indices = np.random.choice(len(replay_buffer), sample_size, replace=False)
        metrics_sample = [replay_buffer[i] for i in sample_indices]
        pre_metrics = compute_metrics(network, metrics_sample, device)
        
        print(f"\nPre-training metrics:")
        print(f"  Value MSE: {pre_metrics['value_mse']:.4f}, MAE: {pre_metrics['value_mae']:.4f}")
        print(f"  Policy Entropy: {pre_metrics['policy_entropy']:.4f}, Accuracy: {pre_metrics['policy_accuracy']:.4f}")
        
        # OPTIMIZATION: Larger batch size and more epochs
        batch_size = 512 if device == 'cuda' else 128
        num_epochs = 3
        
        print(f"\nTraining on replay buffer ({len(replay_buffer)} positions, {num_epochs} epochs)...")
        
        train_start = time.time()
        for epoch in range(num_epochs):
            sample_indices = np.random.choice(len(replay_buffer), 
                                             min(len(iteration_data) * 3, len(replay_buffer)), 
                                             replace=False)
            epoch_data = [replay_buffer[i] for i in sample_indices]
            
            np.random.shuffle(epoch_data)
            total_loss = 0
            total_policy_loss = 0
            total_value_loss = 0
            num_batches = 0
            
            for batch_start in range(0, len(epoch_data), batch_size):
                batch_end = min(batch_start + batch_size, len(epoch_data))
                batch = epoch_data[batch_start:batch_end]
                
                states = torch.FloatTensor([s for s, _, _ in batch]).to(device)
                target_values = torch.FloatTensor([v for _, _, v in batch]).to(device)
                
                # Value target clipping
                target_values = torch.clamp(target_values, min=-0.95, max=0.95)
                
                logits, pred_values = network(states)
                pred_values = pred_values.squeeze()
                
                # Cross-entropy policy loss
                policy_targets_onehot = torch.zeros_like(logits)
                for i, (_, target_probs, _) in enumerate(batch):
                    for move_idx, prob in target_probs.items():
                        if move_idx < policy_targets_onehot.shape[1]:
                            policy_targets_onehot[i, move_idx] = prob
                
                policy_targets_idx = torch.argmax(policy_targets_onehot, dim=-1)
                policy_loss = F.cross_entropy(logits, policy_targets_idx)
                
                value_loss = F.mse_loss(pred_values, target_values)
                
                # Weighted loss
                policy_loss_weight = 1.0
                value_loss_weight = 1.0
                loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(1, num_batches)
            avg_policy = total_policy_loss / max(1, num_batches)
            avg_value = total_value_loss / max(1, num_batches)
            print(f"  Epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.4f} (Policy={avg_policy:.4f}, Value={avg_value:.4f})")
        
        train_time = time.time() - train_start
        
        post_metrics = compute_metrics(network, metrics_sample, device)
        print(f"\nPost-training metrics:")
        print(f"  Value MSE: {post_metrics['value_mse']:.4f}, MAE: {post_metrics['value_mae']:.4f}")
        print(f"  Policy Entropy: {post_metrics['policy_entropy']:.4f}, Accuracy: {post_metrics['policy_accuracy']:.4f}")
        
        log_entry = {
            'iteration': iteration + 1,
            'move_limit': move_limit,
            'games_generated': total_games,
            'positions': total_positions,
            'replay_buffer_size': len(replay_buffer),
            'self_play_time': self_play_time,
            'games_per_second': games_per_second,
            'positions_per_second': positions_per_second,
            'train_time': train_time,
            'pre_metrics': pre_metrics,
            'post_metrics': post_metrics,
            'final_loss': avg_loss,
            'policy_loss': avg_policy,
            'value_loss': avg_value
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        if (iteration + 1) % 3 == 0:
            checkpoint_path = f"chess_resnet_combined_iter_{iteration + 1}.pt"
            torch.save({
                'iteration': iteration + 1,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'net_config': net_config
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    return network


if __name__ == "__main__":
    network = train_with_optimized_fixed_parallel_mcts(
        iterations=50,
        games_per_iter=200,
        num_workers=max(1, mp.cpu_count()),
        num_res_blocks=10,
        num_filters=256
    )
    
    torch.save(network.state_dict(), "chess_resnet_final_combined.pt")
    print("\nOPTIMIZED + FIXED Training complete! Final model saved.")
