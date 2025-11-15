import gym
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import chess
from gym_chess.alphazero.move_encoding import MoveEncoding
from gym_chess.alphazero.board_encoding import BoardEncoding, BoardHistory
from collections import defaultdict
import queue
import threading


class CNNChessNet(nn.Module):
    def __init__(self, input_channels=21, num_actions=4672):
        super().__init__()
        self.conv_base = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.fc_input_dim = 64 * 8 * 8 
        
        self.shared_fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 1024),
            nn.ReLU()
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(1024, num_actions),
            nn.Softmax(dim=-1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Tanh()
        )

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] == 8:
            x = x.permute(0, 3, 1, 2)
        
        shared_conv = self.conv_base(x)
        shared_flat = shared_conv.reshape(-1, self.fc_input_dim)
        
        shared_final = self.shared_fc(shared_flat)
        
        policy = self.policy_head(shared_final)
        value = self.value_head(shared_final)
        return policy, value


class BatchedNeuralNetEvaluator:
    def __init__(self, network, device='cuda', batch_size=32, timeout=0.01):
        self.network = network.to(device)
        self.network.eval()
        self.device = device
        self.batch_size = batch_size
        self.timeout = timeout
        
        self.request_queue = queue.Queue()
        self.running = True
        
        self.batch_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.batch_thread.start()
    
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
            policies, values = self.network(states_tensor)
            return policies.cpu().numpy(), values.cpu().numpy()
    
    def _batch_worker(self):
        while self.running:
            batch = []
            result_queues = []
            
            deadline = None
            while len(batch) < self.batch_size:
                try:
                    if deadline is None:
                        state, result_q = self.request_queue.get(timeout=0.1)
                        batch.append(state)
                        result_queues.append(result_q)
                        deadline = self.timeout
                    else:
                        state, result_q = self.request_queue.get(timeout=deadline)
                        batch.append(state)
                        result_queues.append(result_q)
                except queue.Empty:
                    break
            
            if batch:
                policies, values = self.evaluate_batch(batch)
                
                for i, result_q in enumerate(result_queues):
                    result_q.put((policies[i], values[i].item()))
    
    def shutdown(self):
        self.running = False
        self.batch_thread.join()


class PrunedMCTSWithBatching:
    def __init__(self, evaluator, num_simulations=200, c_puct=1.4, 
                 top_k_initial=10, prune_threshold=0.05):
        self.evaluator = evaluator
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.top_k_initial = top_k_initial
        self.prune_threshold = prune_threshold

        self_play_env = gym.make("Chess-v0")
        self.board_encoder = BoardEncoding(self_play_env, history_length=1)
        self.move_encoder = MoveEncoding(self_play_env)
        
        self.encoding_cache = {}
    
    def search(self, board, temperature=1.0):
        root = MCTSNode(board.copy())
        
        self.encoding_cache.clear()
        
        self._expand_with_progressive_widening(root)
        
        for sim in range(self.num_simulations):
            node = root
            search_path = [node]
            
            while node.children and not node.board.is_game_over():
                node = self._select_child(node)
                search_path.append(node)
            
            if not node.board.is_game_over():
                if self._should_expand(node):
                    value = self._expand_with_progressive_widening(node)
                else:
                    value = self._evaluate_leaf(node)
            else:
                value = self._get_terminal_value(node.board)
            
            self._backpropagate(search_path, value)
            
            if (sim + 1) % 50 == 0:
                self._prune_low_visit_children(root)
        
        best_move = self._get_best_move(root, temperature)
        action_probs = self._get_action_probs(root)
        
        return best_move, action_probs
    
    def _expand_with_progressive_widening(self, node):
        board = node.board
        legal_moves = list(board.legal_moves)
        
        if len(legal_moves) == 0:
            return self._get_terminal_value(board)
        
        # Get encoding and evaluate
        obs = self._encode_board(board)
        policy, value = self.evaluator.evaluate(obs)
        
        self.move_encoder.unwrapped._board = board
        
        move_scores = []
        for move in legal_moves:
            move_idx = self.move_encoder.encode(move)
            score = policy[move_idx] if move_idx < len(policy) else 1.0 / len(legal_moves)
            move_scores.append((move, score))
        
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        num_to_expand = min(
            len(move_scores),
            self.top_k_initial + int(np.sqrt(node.visit_count))
        )
        
        top_moves = move_scores[:num_to_expand]
        total_prob = sum(score for _, score in top_moves)
        
        for move, score in top_moves:
            prior = score / total_prob if total_prob > 0 else 1.0 / len(top_moves)
            child_board = board.copy()
            child_board.push(move)
            node.children[move] = MCTSNode(child_board, parent=node, prior=prior, move=move)
        
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
            return 0.0
    
    def _get_best_move(self, root, temperature=0.0):
        if not root.children:
            return None
        
        if temperature == 0:
            return max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        else:
            moves = list(root.children.keys())
            visits = np.array([root.children[m].visit_count for m in moves])
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


def parallel_self_play_worker(worker_id, network_state_dict, num_games, result_queue, net_config, device='cpu'):
    env = gym.make("Chess-v0")
    env = BoardEncoding(env, history_length=1) # Stateless (8,8,21) observations
    env = MoveEncoding(env)                     # 4672 actions
    
    network = CNNChessNet(net_config['input_channels'], net_config['num_actions'])    
    network.load_state_dict(network_state_dict)
    network.eval()
    
    evaluator = BatchedNeuralNetEvaluator(network, device=device, batch_size=8)
    mcts = PrunedMCTSWithBatching(evaluator, num_simulations=20, top_k_initial=10)
    
    worker_data = []
    
    for game_num in range(num_games):
        encoded_state = env.reset()
        game_data = []
        move_count = 0
        
        while not env.unwrapped.unwrapped._board.is_game_over() and move_count < 100:
            board = env.unwrapped.unwrapped._board.copy()
            temperature = 1.0 if move_count < 15 else 0.1
            
            move, action_probs = mcts.search(board, temperature)
            
            if move is None:
                break
            
            game_data.append((encoded_state, action_probs, None))
            
            action_int = env.encode(move)
            
            encoded_state, _, done, _ = env.step(action_int)
            move_count += 1
        
        board = env.unwrapped.unwrapped._board.copy()
        result = board.result()
        print(result)
        if result == "1-0":
            outcome = 1.0
        elif result == "0-1":
            outcome = -1.0
        else:
            outcome = 0.0
        
        for i, (state, probs, _) in enumerate(game_data):
            player_outcome = outcome if i % 2 == 0 else -outcome
            worker_data.append((state, probs, player_outcome))
        
        print(f"Worker {worker_id}: Game {game_num + 1}/{num_games} completed - {result}")
    
    evaluator.shutdown()
    env.close()
    
    # Return data to main process
    result_queue.put(worker_data)


def train_with_parallel_mcts(iterations=100, games_per_iter=25, num_workers=None):
    """
    Training with parallel self-play using root parallelization.
    This achieves near-linear speedup on multi-core CPUs.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    print(f"Using {num_workers} parallel workers")
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # <--- CHANGED: Create env manually to get dims
    env = gym.make("Chess-v0")
    env = BoardEncoding(env, history_length=1)
    env = MoveEncoding(env)
    
    input_channels = env.observation_space.shape[2]
    num_actions = env.action_space.n
    
    net_config = {'input_channels': input_channels, 'num_actions': num_actions}
    env.close()
    
    print(f"Network Config: Input Channels={input_channels}, Actions={num_actions}")
    
    network = CNNChessNet(input_channels, num_actions)
    if device == 'cuda':
        network = network.cuda()
    
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    replay_buffer = []
    
    for iteration in range(iterations):
        print(f"\n=== Iteration {iteration + 1}/{iterations} ===")
        
        # Parallel self-play using multiprocessing
        # <--- CHANGED: Send a CPU copy of the state dict
        network_state = {k: v.cpu() for k, v in network.state_dict().items()}
        
        # Distribute games across workers
        games_per_worker = max(1, games_per_iter // num_workers)
        
        # Create multiprocessing context
        mp.set_start_method('spawn', force=True)
        result_queue = mp.Queue()
        processes = []
        
        for worker_id in range(num_workers):
            p = mp.Process(
                target=parallel_self_play_worker,
                args=(worker_id, network_state, games_per_worker, result_queue, net_config, 'cpu')
            )
            p.start()
            processes.append(p)
        
        iteration_data = []
        for _ in range(num_workers):
            worker_data = result_queue.get()
            iteration_data.extend(worker_data)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        print(f"Collected {len(iteration_data)} training positions from parallel self-play")
        
        # Update replay buffer
        replay_buffer.extend(iteration_data)
        if len(replay_buffer) > 50000:
            replay_buffer = replay_buffer[-50000:]
        
        # Training on GPU with batching
        print(f"Training on {len(iteration_data)} positions...")
        if len(iteration_data) == 0:
            print("No data collected, skipping training.")
            continue
            
        batch_size = 128 if device == 'cuda' else 32
        
        for epoch in range(5):
            np.random.shuffle(iteration_data)
            total_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for batch_start in range(0, len(iteration_data), batch_size):
                batch_end = min(batch_start + batch_size, len(iteration_data))
                batch = iteration_data[batch_start:batch_end]
                
                # Prepare batch
                # <--- CHANGED: States are (8, 8, 21), not flat
                states = torch.FloatTensor([s for s, _, _ in batch]).to(device)
                target_values = torch.FloatTensor([v for _, _, v in batch]).to(device)
                
                # Forward pass
                pred_policies, pred_values = network(states)
                pred_values = pred_values.squeeze()
                
                # Prepare policy targets
                policy_targets = torch.zeros_like(pred_policies)
                for i, (_, target_probs, _) in enumerate(batch):
                    # <--- CHANGED: target_probs is now dict[int, float]
                    for move_idx, prob in target_probs.items():
                        if move_idx < policy_targets.shape[1]:
                            policy_targets[i, move_idx] = prob
                
                # <--- CHANGED: No need to normalize, MCTS already did
                
                # Loss calculation
                # <--- CHANGED: Corrected policy loss for batch
                policy_loss = -torch.sum(policy_targets * torch.log(pred_policies + 1e-8)) / len(batch)
                value_loss = torch.mean((pred_values - target_values) ** 2)
                loss = policy_loss + value_loss
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(1, num_batches)
            print(f"   Epoch {epoch + 1}/5: Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if (iteration + 1) % 10 == 0:
            torch.save(network.state_dict(), f"parallel_mcts_chess_{iteration + 1}.pt")
    
    return network


# Main execution
if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Train with parallelization
    network = train_with_parallel_mcts(
        iterations=50, 
        games_per_iter=20,
        num_workers=max(1, mp.cpu_count())
    )