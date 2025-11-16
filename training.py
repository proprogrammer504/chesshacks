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


class GPUBatchedNeuralNetEvaluator:
    """GPU-accelerated batched evaluator with dynamic batching"""
    def __init__(self, network, device='cuda', batch_size=64, max_wait_time=0.005):
        self.network = network.to(device)
        self.network.eval()
        self.device = device
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        
        self.request_queue = queue.Queue()
        self.running = True
        
        self.num_workers = 2 if device == 'cuda' else 1
        self.batch_threads = []
        for i in range(self.num_workers):
            thread = threading.Thread(target=self._batch_worker, daemon=True)
            thread.start()
            self.batch_threads.append(thread)
    
    def evaluate(self, board_state):
        result_queue = queue.Queue()
        self.request_queue.put((board_state, result_queue))
        policy, value = result_queue.get()
        return policy, value
    
    def evaluate_batch(self, board_states):
        if len(board_states) == 0:
            return [], []
        
        with torch.no_grad():
            states_tensor = torch.FloatTensor(np.array(board_states)).to(self.device)
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
                remaining_time = max(0.001, self.max_wait_time - elapsed)
                
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
            thread.join(timeout=2.0)


class ParallelGPUMCTS:
    def __init__(self, evaluator, num_simulations=200, c_puct=1.4, 
                 top_k_initial=10, prune_threshold=0.05,
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
                 parallel_sims=8):
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
        
        self.encoding_cache = {}
        self.position_history = {}
    
    def search(self, board, temperature=1.0):
        root = MCTSNode(board.copy())
        
        self.encoding_cache.clear()
        
        self._update_position_history(board)
        
        self._expand_with_progressive_widening(root)
        
        num_batches = max(1, self.num_simulations // self.parallel_sims)
        
        for batch_idx in range(num_batches):
            simulation_data = []
            
            for _ in range(self.parallel_sims):
                node = root
                search_path = [node]
                
                while node.children and not node.board.is_game_over():
                    node = self._select_child(node)
                    search_path.append(node)
                
                if not node.board.is_game_over():
                    if self._should_expand(node):
                        simulation_data.append(('expand', node, search_path))
                    else:
                        simulation_data.append(('evaluate', node, search_path))
                else:
                    value = self._get_terminal_value(node.board)
                    self._backpropagate(search_path, value)
            
            self._batch_process_simulations(simulation_data)
            
            if (batch_idx + 1) % 10 == 0:
                self._prune_low_visit_children(root)
        
        best_move = self._get_best_move(root, temperature)
        action_probs = self._get_action_probs(root)
        
        return best_move, action_probs
    
    def _update_position_history(self, board):
        """Update position history from board's move stack"""
        fen = self._get_position_fen(board)
        self.position_history[fen] = self.position_history.get(fen, 0) + 1
    
    def _get_position_fen(self, board):
        """Extract position-only part of FEN (board + side to move + castling + en passant)"""
        full_fen = board.fen()
        parts = full_fen.split(' ')
        return ' '.join(parts[:4])
    
    def _count_repetitions(self, board):
        """Count how many times this position has occurred"""
        fen = self._get_position_fen(board)
        return self.position_history.get(fen, 0)
    
    def _batch_process_simulations(self, simulation_data):
        if not simulation_data:
            return
        
        expand_data = [(node, path) for action_type, node, path in simulation_data if action_type == 'expand']
        eval_data = [(node, path) for action_type, node, path in simulation_data if action_type == 'evaluate']
        
        all_nodes = [node for node, _ in expand_data + eval_data]
        
        if all_nodes:
            board_states = [self._encode_board(node.board) for node in all_nodes]
            
            policies, values = self.evaluator.evaluate_batch(board_states)
            
            for i, (node, path) in enumerate(expand_data):
                policy = policies[i]
                value = values[i].item()
                self._expand_node_with_policy(node, policy)
                self._backpropagate(path, value)
            
            offset = len(expand_data)
            for i, (node, path) in enumerate(eval_data):
                value = values[offset + i].item()
                self._backpropagate(path, value)
    
    def _expand_node_with_policy(self, node, policy):
        board = node.board
        legal_moves = list(board.legal_moves)
        
        if len(legal_moves) == 0:
            return
        
        self.move_encoder.unwrapped._board = board
        
        move_scores = []
        for move in legal_moves:
            move_idx = self.move_encoder.encode(move)
            score = policy[move_idx] if move_idx < len(policy) else 1.0 / len(legal_moves)
            
            test_board = board.copy()
            test_board.push(move)
            
            repetition_count = self._count_repetitions(test_board)
            
            # STRENGTHENED: More aggressive repetition penalties
            if repetition_count >= 2:
                score *= 0.0001  # Nearly eliminate moves leading to 3rd repetition
            elif repetition_count == 1:
                score *= 0.1    # Heavily reduce moves leading to 2nd repetition
            
            if test_board.can_claim_threefold_repetition():
                score *= 0.00001
            
            move_scores.append((move, score))
        
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        total_score = sum(score for _, score in move_scores)
        if total_score < 1e-6:
            print("Warning: All moves lead to repetition, using uniform priors")
            move_scores = [(move, 1.0 / len(legal_moves)) for move, _ in move_scores]
            total_score = 1.0
        
        num_to_expand = min(
            len(move_scores),
            self.top_k_initial + int(np.sqrt(node.visit_count))
        )
        
        top_moves = move_scores[:num_to_expand]
        total_prob = sum(score for _, score in top_moves)
        
        is_root = (node.parent is None)
        if is_root and len(top_moves) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(top_moves))
            noisy_moves = []
            for i, (move, score) in enumerate(top_moves):
                prior = score / total_prob if total_prob > 0 else 1.0 / len(top_moves)
                noisy_prior = (1 - self.dirichlet_epsilon) * prior + self.dirichlet_epsilon * noise[i]
                noisy_moves.append((move, noisy_prior))
            top_moves = noisy_moves
        else:
            top_moves = [(move, score / total_prob if total_prob > 0 else 1.0 / len(top_moves)) 
                        for move, score in top_moves]
        
        for move, prior in top_moves:
            child_board = board.copy()
            child_board.push(move)
            node.children[move] = MCTSNode(child_board, parent=node, prior=prior, move=move)
    
    def _expand_with_progressive_widening(self, node):
        board = node.board
        legal_moves = list(board.legal_moves)
        
        if len(legal_moves) == 0:
            return self._get_terminal_value(board)
        
        obs = self._encode_board(board)
        policy, value = self.evaluator.evaluate(obs)
        
        self._expand_node_with_policy(node, policy)
        
        return value
    
    def _select_child(self, node):
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
    
    def _should_expand(self, node):
        if node.visit_count < 3:
            return True
        if node.parent and node.q_value() < -0.8:
            return False
        return True
    
    def _prune_low_visit_children(self, node):
        if not node.children or node.visit_count < 50:
            return
        
        avg_visits = sum(c.visit_count for c in node.children.values()) / len(node.children)
        threshold = avg_visits * self.prune_threshold
        
        to_remove = [
            move for move, child in node.children.items()
            if child.visit_count < threshold and child.visit_count < 5
        ]
        
        for move in to_remove:
            del node.children[move]
    
    def _backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            node.virtual_loss = max(0, node.virtual_loss - 1)
            value = -value
    
    def _evaluate_leaf(self, node):
        obs = self._encode_board(node.board)
        _, value = self.evaluator.evaluate(obs)
        return value
    
    def _get_terminal_value(self, board):
        result = board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            # CHANGED: Larger penalty for draws
            return -0.3
    
    def _get_best_move(self, root, temperature=0.0, epsilon=0.25):
        if not root.children:
            return None

        moves = list(root.children.keys())
        visits = np.array([root.children[m].visit_count for m in moves])

        if np.random.random() < epsilon:
            visits_temp = visits ** 0.5
            probs = visits_temp / visits_temp.sum()
            return np.random.choice(moves, p=probs)
        else:
            if temperature == 0:
                return moves[np.argmax(visits)]
            else:
                visits = visits ** (1.0 / temperature)
                probs = visits / visits.sum()
                return np.random.choice(moves, p=probs)
    
    def _get_action_probs(self, root):
        if not root.children:
            return {}
        
        self.move_encoder.unwrapped._board = root.board
        
        visits_sum = sum(child.visit_count for child in root.children.values())
        if visits_sum == 0:
            return {}
            
        action_probs = {}
        for move, child in root.children.items():
            action_int = self.move_encoder.encode(move)
            action_probs[action_int] = child.visit_count / visits_sum
            
        return action_probs
    
    def _encode_board(self, board):
        board_fen = board.fen()
        if board_fen in self.encoding_cache:
            return self.encoding_cache[board_fen]
        
        self.board_encoder.unwrapped._board = board
        self.board_encoder._history.reset()
        obs = self.board_encoder.observation(board)
        
        self.encoding_cache[board_fen] = obs
        return obs


class MCTSNode:
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
    
    def ucb_score(self, parent_visits, c_puct=1.4):
        effective_visits = self.visit_count + self.virtual_loss
        q = self.q_value()
        u = c_puct * self.prior * np.sqrt(parent_visits) / (1 + effective_visits)
        return q + u


def parallel_self_play_worker(worker_id, network_state_dict, num_games, result_queue, net_config, device='cpu', move_limit=151):
    try:
        env = gym.make("Chess-v0")
        env = BoardEncoding(env, history_length=1)
        env = MoveEncoding(env)
        
        network = DeepResNetChess(net_config['input_channels'], net_config['num_actions'], 
                                   net_config['num_res_blocks'], net_config['num_filters'])
        network.load_state_dict(network_state_dict)
        network.eval()
        
        batch_size = 64 if device == 'cuda' else 16
        evaluator = GPUBatchedNeuralNetEvaluator(network, device=device, batch_size=batch_size, max_wait_time=0.005)
        
        parallel_sims = 16 if device == 'cuda' else 4
        mcts = ParallelGPUMCTS(evaluator, num_simulations=60, top_k_initial=10,
                               dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
                               parallel_sims=parallel_sims)
        
        worker_data = []
        
        for game_num in range(num_games):
            encoded_state = env.reset()
            game_data = []
            move_count = 0
            
            mcts.position_history.clear()
            
            game_end_reason = "ongoing"
            
            while not env.unwrapped.unwrapped._board.is_game_over() and move_count < move_limit:
                board = env.unwrapped.unwrapped._board.copy()
                
                # Check for draws before making move
                if board.can_claim_threefold_repetition():
                    game_end_reason = "threefold_repetition"
                    outcome = -0.3
                    print(f"Worker {worker_id}: Game {game_num + 1} - Threefold repetition at move {move_count}")
                    break
                
                if board.is_fivefold_repetition():
                    game_end_reason = "fivefold_repetition"
                    outcome = -0.5
                    print(f"Worker {worker_id}: Game {game_num + 1} - Fivefold repetition at move {move_count}")
                    break
                
                if board.can_claim_fifty_moves() or board.is_fifty_moves():
                    game_end_reason = "fifty_moves"
                    outcome = -0.2
                    print(f"Worker {worker_id}: Game {game_num + 1} - Fifty-move rule at move {move_count}")
                    break
                
                # CHANGED: More aggressive temperature schedule
                if move_count < 20:
                    temperature = 1.0
                    epsilon = 0.25
                elif move_count < 40:
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
                
                # ADDED: Store with step penalty
                step_penalty = -0.002 * move_count
                game_data.append((encoded_state, action_probs, None, step_penalty))
                
                action_int = env.encode(move)
                
                encoded_state, _, done, _ = env.step(action_int)
                move_count += 1
            
            # Determine final outcome
            board = env.unwrapped.unwrapped._board.copy()
            
            if game_end_reason == "ongoing":
                # ADDED: Check if we hit move limit
                if move_count >= move_limit:
                    outcome = -0.5  # Heavy penalty for timeouts
                    game_end_reason = "move_limit_timeout"
                    print(f"Worker {worker_id}: Game {game_num + 1} - Move limit timeout")
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
            
            # CHANGED: Assign outcomes with step penalty
            for i, (state, probs, _, step_penalty) in enumerate(game_data):
                player_outcome = outcome if i % 2 == 0 else -outcome
                # Add step penalty to encourage shorter games
                final_value = player_outcome + step_penalty
                worker_data.append((state, probs, final_value))
            
            print(f"Worker {worker_id}: Game {game_num + 1}/{num_games} completed - {game_end_reason} ({move_count} moves, {len(game_data)} positions)")
        
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


def train_with_parallel_mcts(iterations=100, games_per_iter=100, num_workers=None,
                             num_res_blocks=10, num_filters=256):
    if num_workers is None:
        num_workers = max(1, mp.cpu_count())
    
    print(f"Using {num_workers} parallel workers")
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
    
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-4)
    replay_buffer = []
    
    log_file = Path("training_log.jsonl")
    
    for iteration in range(iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{iterations}")
        print(f"{'='*60}")

        # ADDED: Progressive move limit reduction
        move_limit = max(80, 151 - iteration)
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
                target=parallel_self_play_worker,
                args=(worker_id, network_state, games_per_worker, result_queue, net_config, worker_device, move_limit)
            )
            p.start()
            processes.append(p)
        
        iteration_data = []
        completed_workers = 0
        timeout_per_game = 180

        while completed_workers < num_workers:
            try:
                remaining_workers = num_workers - completed_workers
                expected_time = games_per_worker * timeout_per_game

                worker_data = result_queue.get(timeout=expected_time)
                iteration_data.extend(worker_data)
                completed_workers += 1
                elapsed = time.time() - start_time
                print(f"Worker {completed_workers}/{num_workers} completed ({elapsed:.0f}s elapsed, {len(worker_data)} positions)")
            except queue.Empty:
                print(f"Warning: Worker timeout after {time.time() - start_time:.0f}s")
                break

        for p in processes:
            if p.is_alive():
                print(f"Waiting for worker PID {p.pid}...")
            p.join(timeout=60)
            if p.is_alive():
                print(f"Forcefully terminating worker PID {p.pid}")
                p.terminate()
                p.join(timeout=2)

        
        self_play_time = time.time() - start_time
        print(f"\nSelf-play completed in {self_play_time:.1f}s")
        print(f"Collected {len(iteration_data)} training positions")
        
        replay_buffer.extend(iteration_data)
        max_buffer_size = 100000
        if len(replay_buffer) > max_buffer_size:
            replay_buffer = replay_buffer[-max_buffer_size:]
        
        print(f"Replay buffer size: {len(replay_buffer)} positions")
        
        if len(replay_buffer) == 0:
            print("No data in replay buffer, skipping training.")
            continue
        
        sample_size = min(1000, len(replay_buffer))
        sample_indices = np.random.choice(len(replay_buffer), sample_size, replace=False)
        metrics_sample = [replay_buffer[i] for i in sample_indices]
        pre_metrics = compute_metrics(network, metrics_sample, device)
        
        print(f"\nPre-training metrics:")
        print(f"  Value MSE: {pre_metrics['value_mse']:.4f}, MAE: {pre_metrics['value_mae']:.4f}")
        print(f"  Policy Entropy: {pre_metrics['policy_entropy']:.4f}, Accuracy: {pre_metrics['policy_accuracy']:.4f}")
        
        batch_size = 256 if device == 'cuda' else 64
        num_epochs = 5
        
        print(f"\nTraining on replay buffer ({len(replay_buffer)} positions, {num_epochs} epochs)...")
        
        train_start = time.time()
        for epoch in range(num_epochs):
            sample_indices = np.random.choice(len(replay_buffer), 
                                             min(len(iteration_data) * 2, len(replay_buffer)), 
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
                
                # ADDED: Amplify penalties for draws/timeouts
                target_values = torch.where(
                    (target_values < 0) & (target_values > -0.6),
                    target_values * 2.0,
                    target_values
                )
                
                logits, pred_values = network(states)
                pred_values = pred_values.squeeze()
                
                policy_targets = torch.zeros_like(logits)
                for i, (_, target_probs, _) in enumerate(batch):
                    for move_idx, prob in target_probs.items():
                        if move_idx < policy_targets.shape[1]:
                            policy_targets[i, move_idx] = prob
                
                log_probs = F.log_softmax(logits, dim=-1)
                policy_loss = -torch.sum(policy_targets * log_probs) / len(batch)
                
                value_loss = F.mse_loss(pred_values, target_values)
                
                loss = policy_loss + value_loss
                
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
            'games_generated': len(iteration_data) // 50,
            'positions': len(iteration_data),
            'replay_buffer_size': len(replay_buffer),
            'self_play_time': self_play_time,
            'train_time': train_time,
            'pre_metrics': pre_metrics,
            'post_metrics': post_metrics,
            'final_loss': avg_loss,
            'policy_loss': avg_policy,
            'value_loss': avg_value
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        if (iteration + 1) % 5 == 0:
            checkpoint_path = f"chess_resnet_iter_{iteration + 1}.pt"
            torch.save({
                'iteration': iteration + 1,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'net_config': net_config
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    return network


if __name__ == "__main__":
    network = train_with_parallel_mcts(
        iterations=100,
        games_per_iter=100,
        num_workers=max(1, mp.cpu_count()),
        num_res_blocks=10,
        num_filters=256
    )
    
    torch.save(network.state_dict(), "chess_resnet_final.pt")
    print("\nTraining complete! Final model saved.")
