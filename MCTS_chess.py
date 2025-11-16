"""Monte Carlo Tree Search (MCTS) implementation with PUCT for chess"""

import numpy as np
import math
import chess
from chess_board import ChessBoard
from encoder_decoder import encode_board, decode_policy_output, get_legal_moves_mask
import config


class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, board: ChessBoard, parent=None, move=None, prior_prob=0.0):
        """
        Initialize MCTS node
        
        Args:
            board: ChessBoard object representing this state
            parent: Parent MCTSNode
            move: chess.Move that led to this state
            prior_prob: Prior probability from neural network
        """
        self.board = board
        self.parent = parent
        self.move = move
        self.prior_prob = prior_prob
        
        self.children = {}  # Dictionary mapping moves to child nodes
        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0
        
        self.is_expanded = False
    
    def is_leaf(self):
        """Check if node is a leaf (not expanded)"""
        return not self.is_expanded
    
    def get_value(self):
        """Get mean value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.mean_value
    
    def select_child(self, c_puct=config.CPUCT):
        """
        Select child with highest PUCT value
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            tuple: (move, child_node)
        """
        best_score = -float('inf')
        best_move = None
        best_child = None
        
        # Calculate sum of visit counts for parent
        sqrt_parent_visits = math.sqrt(self.visit_count)
        
        for move, child in self.children.items():
            # PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            q_value = child.get_value()
            u_value = c_puct * child.prior_prob * sqrt_parent_visits / (1 + child.visit_count)
            puct_score = q_value + u_value
            
            if puct_score > best_score:
                best_score = puct_score
                best_move = move
                best_child = child
        
        return best_move, best_child
    
    def expand(self, policy_probs):
        """
        Expand node by creating children for all legal moves
        
        Args:
            policy_probs: Dictionary mapping moves to prior probabilities
        """
        legal_moves = self.board.get_legal_moves()
        
        for move in legal_moves:
            if move not in self.children:
                # Get prior probability for this move
                prior = policy_probs.get(move, 0.0)
                
                # Create child board state
                child_board = self.board.copy()
                child_board.make_move(move)
                
                # Create child node
                child_node = MCTSNode(child_board, parent=self, move=move, prior_prob=prior)
                self.children[move] = child_node
        
        self.is_expanded = True
    
    def update(self, value):
        """
        Update node statistics after simulation
        
        Args:
            value: Value to backpropagate (from perspective of current player)
        """
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count
    
    def get_visit_counts(self):
        """
        Get visit counts for all children
        
        Returns:
            dict: Mapping moves to visit counts
        """
        return {move: child.visit_count for move, child in self.children.items()}
    
    def get_policy_target(self, temperature=1.0):
        """
        Get policy target distribution from visit counts
        
        Args:
            temperature: Temperature parameter (0 = greedy, >1 = more exploration)
            
        Returns:
            dict: Mapping moves to probabilities
        """
        visit_counts = self.get_visit_counts()
        
        if temperature == 0:
            # Greedy selection
            best_move = max(visit_counts.keys(), key=lambda m: visit_counts[m])
            return {move: 1.0 if move == best_move else 0.0 for move in visit_counts.keys()}
        
        # Apply temperature
        visits_temp = {move: count ** (1.0 / temperature) 
                      for move, count in visit_counts.items()}
        
        # Normalize to get probabilities
        total = sum(visits_temp.values())
        policy = {move: count / total for move, count in visits_temp.items()}
        
        return policy


class MCTS:
    """Monte Carlo Tree Search implementation"""
    
    def __init__(self, neural_net, num_simulations=800, c_puct=1.0, 
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        """
        Initialize MCTS
        
        Args:
            neural_net: Neural network for policy and value prediction
            num_simulations: Number of simulations per search
            c_puct: PUCT exploration constant
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_epsilon: Weight of Dirichlet noise at root
        """
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
    
    def search(self, board: ChessBoard, add_noise=True, temperature=1.0):
        """
        Perform MCTS search from current board position
        
        Args:
            board: ChessBoard object
            add_noise: Whether to add Dirichlet noise to root node
            temperature: Temperature for move selection
            
        Returns:
            tuple: (move, policy_target)
                - move: Selected chess.Move
                - policy_target: Target policy distribution
        """
        # Create root node
        root = MCTSNode(board.copy())
        
        # Expand root node
        policy_probs, _ = self._evaluate_leaf(root)
        
        # Add Dirichlet noise to root node for exploration
        if add_noise:
            policy_probs = self._add_dirichlet_noise(policy_probs, root.board)
        
        root.expand(policy_probs)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse tree until leaf
            while not node.is_leaf() and not node.board.is_game_over():
                move, node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # Evaluate leaf
            if node.board.is_game_over():
                # Terminal node
                value = self._get_terminal_value(node.board)
            else:
                # Expansion and evaluation
                policy_probs, value = self._evaluate_leaf(node)
                node.expand(policy_probs)
            
            # Backpropagation
            self._backpropagate(search_path, value)
        
        # Get policy target from visit counts
        policy_target = root.get_policy_target(temperature)
        
        # Select move based on visit counts
        if temperature == 0:
            # Greedy selection
            move = max(root.children.keys(), 
                      key=lambda m: root.children[m].visit_count)
        else:
            # Sample from policy
            moves = list(policy_target.keys())
            probs = list(policy_target.values())
            move = np.random.choice(moves, p=probs)
        
        return move, policy_target
    
    def _evaluate_leaf(self, node: MCTSNode):
        """
        Evaluate leaf node using neural network
        
        Args:
            node: MCTSNode to evaluate
            
        Returns:
            tuple: (policy_probs, value)
        """
        # Encode board state
        board_encoding = encode_board(node.board)
        
        # Get neural network prediction
        policy_logits, value = self.neural_net.predict(board_encoding)
        
        # Convert policy to dictionary of move -> probability
        legal_moves = node.board.get_legal_moves()
        moves_with_probs = decode_policy_output(policy_logits, node.board, temperature=1.0)
        
        # Create dictionary
        policy_probs = {move: prob for move, prob in moves_with_probs}
        
        return policy_probs, value
    
    def _add_dirichlet_noise(self, policy_probs, board: ChessBoard):
        """
        Add Dirichlet noise to policy for exploration
        
        Args:
            policy_probs: Dictionary of move -> probability
            board: ChessBoard object
            
        Returns:
            dict: Policy with added noise
        """
        legal_moves = board.get_legal_moves()
        num_legal_moves = len(legal_moves)
        
        if num_legal_moves == 0:
            return policy_probs
        
        # Generate Dirichlet noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_legal_moves)
        
        # Add noise to policy
        noisy_policy = {}
        for i, move in enumerate(legal_moves):
            original_prob = policy_probs.get(move, 0.0)
            noisy_policy[move] = (1 - self.dirichlet_epsilon) * original_prob + \
                                self.dirichlet_epsilon * noise[i]
        
        # Normalize
        total = sum(noisy_policy.values())
        if total > 0:
            noisy_policy = {move: prob / total for move, prob in noisy_policy.items()}
        
        return noisy_policy
    
    def _get_terminal_value(self, board: ChessBoard):
        """
        Get value for terminal board state
        
        Args:
            board: ChessBoard object
            
        Returns:
            float: Value from current player's perspective
        """
        winner = board.get_winner()
        current_player = board.get_turn()
        
        if winner is None:
            return 0.0
        
        # Value is from current player's perspective
        # Note: At terminal state, it's the previous player who made the winning move
        # So we need to flip the value
        return -float(winner * current_player)
    
    def _backpropagate(self, search_path, value):
        """
        Backpropagate value through search path
        
        Args:
            search_path: List of MCTSNode objects
            value: Value to backpropagate
        """
        # Value is from the perspective of the player at the leaf
        # As we go up the tree, we alternate perspectives
        for node in reversed(search_path):
            node.update(value)
            value = -value  # Flip value for parent's perspective


def self_play_game(neural_net, num_simulations=800, temperature_threshold=30, verbose=False):
    """
    Play a single self-play game using MCTS
    
    Args:
        neural_net: Neural network for move selection
        num_simulations: Number of MCTS simulations per move
        temperature_threshold: Move number after which to use greedy selection
        verbose: Whether to print game progress
        
    Returns:
        list: List of (board_state, policy, value) tuples
    """
    board = ChessBoard()
    mcts = MCTS(neural_net, num_simulations=num_simulations)
    
    training_examples = []
    move_count = 0
    
import time
import numpy as np

def self_play_game(neural_net, num_simulations=800, temperature_threshold=30, verbose=False):
    """
    Play a single self-play game using MCTS
    
    Args:
        neural_net: Neural network for move selection
        num_simulations: Number of MCTS simulations per move
        temperature_threshold: Move number after which to use greedy selection
        verbose: Whether to print game progress
        
    Returns:
        list: List of (board_state, policy, value) tuples
    """
    board = ChessBoard()
    mcts = MCTS(neural_net, num_simulations=num_simulations)
    
    training_examples = []
    move_count = 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎮 Starting Self-Play Game")
        print(f"{'='*60}")
        print("Starting position:")
        print(board)
    
    move_times = []
    
    while not board.is_game_over() and move_count < config.MAX_GAME_LENGTH:
        move_count += 1
        
        # Use temperature for exploration in early game
        temperature = config.TEMPERATURE if move_count < temperature_threshold else 0.0
        
        if verbose:
            start_time = time.time()
        
        # Perform MCTS search
        move, policy_target = mcts.search(board, add_noise=True, temperature=temperature)
        
        if verbose:
            end_time = time.time()
            move_time = end_time - start_time
            move_times.append(move_time)
            avg_move_time = np.mean(move_times)
            
            print(f"\n⚡ Move {move_count:3d}/{config.MAX_GAME_LENGTH}")
            print(f"   ⏱️  Time: {move_time:.2f}s (avg: {avg_move_time:.2f}s)")
            print(f"   🌡️  Temperature: {temperature:.1f}")
            print(f"   🎯 Selected move: {move}")
            
            # Show legal moves count and policy distribution
            legal_moves = board.get_legal_moves()
            print(f"   📊 Legal moves: {len(legal_moves)}")
            
            # Show top 3 moves by probability
            top_moves = sorted(policy_target.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   🏆 Top moves:")
            for i, (m, prob) in enumerate(top_moves):
                print(f"      {i+1}. {m}: {prob:.3f}")
            
            print(f"   📋 Board:")
            print(board)
        
        if not verbose:
            if move_count % 10 == 0:
                print(f"Move {move_count}: Searching with {num_simulations} simulations...")
                print(f"  → Selected: {move} (temp={temperature:.1f})")
        
        if verbose and move_count % 10 == 0:
            print(f"Move {move_count}: Searching with {num_simulations} simulations...")
        
        # Perform MCTS search
        move, policy_target = mcts.search(board, add_noise=True, temperature=temperature)
        
        if verbose and move_count % 10 == 0:
            print(f"  → Selected: {move} (temp={temperature:.1f})")
        
        # Store training example
        board_encoding = encode_board(board)
        training_examples.append({
            'board': board_encoding,
            'policy': policy_target,
            'turn': board.get_turn()
        })
        
        # Make move
        board.make_move(move)
    
    # Get game result
    winner = board.get_winner()
    if winner is None:
        winner = 0  # Draw
    
    if verbose:
        result_str = "White wins  ✅" if winner == 1 else "Black wins ✅" if winner == -1 else "Draw 🤝"
        avg_game_moves = move_count / len(move_times) if move_times else 0
        
        print(f"\n{'='*60}")
        print(f"🏁 Game Finished!")
        print(f"{'='*60}")
        print(f"   🏆 Result: {result_str}")
        print(f"   ⚡ Moves: {move_count}")
        print(f"   ⏱️  Total time: {sum(move_times):.2f}s")
        print(f"   📊 Avg time/move: {np.mean(move_times):.2f}s")
        print(f"   📈 Examples generated: {len(training_examples)}")
        print(f"{'='*60}\n")
    
    if not verbose:
        result_str = "White wins" if winner == 1 else "Black wins" if winner == -1 else "Draw"
        print(f"\nGame finished: {result_str} after {move_count} moves")
        print(f"Generated {len(training_examples)} training examples")
        print(f"{'='*60}\n")
    
    # Add game result to all training examples
    for example in training_examples:
        # Value is from the perspective of the player at that position
        example['value'] = float(winner * example['turn'])
    
    return training_examples


if __name__ == "__main__":
    # Test MCTS implementation
    from alpha_net import create_alpha_net
    
    print("Creating neural network...")
    net = create_alpha_net()
    net.eval()
    
    print("Testing MCTS search...")
    board = ChessBoard()
    mcts = MCTS(net, num_simulations=100)
    
    move, policy = mcts.search(board, add_noise=True)
    print(f"Selected move: {move}")
    print(f"Policy has {len(policy)} legal moves")