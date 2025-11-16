"""Functions to encode/decode chess board states and moves for neural network"""

import numpy as np
import chess
from chess_board import ChessBoard


def encode_board(board: ChessBoard):
    """
    Encode chess board state as neural network input
    
    Creates 18 planes (channels):
    - 12 planes for pieces (6 types × 2 colors)
    - 2 planes for castling rights
    - 2 planes for en passant
    - 1 plane for turn color
    - 1 plane for move count
    
    Args:
        board: ChessBoard object
        
    Returns:
        numpy.ndarray: Shape (18, 8, 8) representing board state
    """
    # Initialize 18 planes
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    
    # Get current turn (1 for white, -1 for black)
    current_turn = board.get_turn()
    
    # Piece type mapping
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                   chess.ROOK, chess.QUEEN, chess.KING]
    
    # Encode pieces (planes 0-11)
    for square in chess.SQUARES:
        piece = board.board.piece_at(square)
        if piece:
            rank = square // 8
            file = square % 8
            
            # Find piece type index
            piece_idx = piece_types.index(piece.piece_type)
            
            # Determine plane index (0-5 for white, 6-11 for black)
            if piece.color == chess.WHITE:
                plane_idx = piece_idx
            else:
                plane_idx = piece_idx + 6
            
            planes[plane_idx, rank, file] = 1.0
    
    # Encode castling rights (planes 12-13)
    # Plane 12: Current player's castling rights
    if current_turn == 1:  # White to move
        if board.board.has_kingside_castling_rights(chess.WHITE):
            planes[12, :, :] = 1.0
        if board.board.has_queenside_castling_rights(chess.WHITE):
            planes[13, :, :] = 1.0
    else:  # Black to move
        if board.board.has_kingside_castling_rights(chess.BLACK):
            planes[12, :, :] = 1.0
        if board.board.has_queenside_castling_rights(chess.BLACK):
            planes[13, :, :] = 1.0
    
    # Encode en passant (planes 14-15)
    ep_square = board.board.ep_square
    if ep_square is not None:
        rank = ep_square // 8
        file = ep_square % 8
        planes[14, rank, file] = 1.0
        # Mark entire file for easier recognition
        planes[15, :, file] = 1.0
    
    # Encode current turn (plane 16)
    # 1 for white to move, 0 for black to move
    if current_turn == 1:
        planes[16, :, :] = 1.0
    
    # Encode move count normalized (plane 17)
    move_count = len(board.board.move_stack)
    planes[17, :, :] = min(move_count / 100.0, 1.0)
    
    return planes


def decode_move_index(move_idx, board: ChessBoard):
    """
    Decode move index back to chess.Move
    
    Args:
        move_idx: Integer index of move (0-4095)
        board: ChessBoard object for context
        
    Returns:
        chess.Move object or None if invalid
    """
    # Move encoding: from_square (0-63) * 64 + to_square (0-63)
    # This gives us 4096 basic moves
    from_square = move_idx // 64
    to_square = move_idx % 64
    
    # Check if this is a valid move
    move = chess.Move(from_square, to_square)
    
    # Check for promotion moves
    if board.board.piece_at(from_square):
        piece = board.board.piece_at(from_square)
        if piece.piece_type == chess.PAWN:
            # Check if pawn reaches last rank
            to_rank = to_square // 8
            if (piece.color == chess.WHITE and to_rank == 7) or \
               (piece.color == chess.BLACK and to_rank == 0):
                # Try queen promotion by default
                move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
    
    if move in board.board.legal_moves:
        return move
    
    return None


def encode_move(move: chess.Move):
    """
    Encode chess.Move as integer index
    
    Args:
        move: chess.Move object
        
    Returns:
        int: Move index (0-4095)
    """
    from_square = move.from_square
    to_square = move.to_square
    
    # Basic encoding: from * 64 + to
    move_idx = from_square * 64 + to_square
    
    return move_idx


def create_policy_vector(legal_moves, move_probabilities):
    """
    Create policy vector for all possible moves
    
    Args:
        legal_moves: List of chess.Move objects
        move_probabilities: List of probabilities corresponding to legal_moves
        
    Returns:
        numpy.ndarray: Shape (4096,) with probabilities for all moves
    """
    policy = np.zeros(4096, dtype=np.float32)
    
    for move, prob in zip(legal_moves, move_probabilities):
        move_idx = encode_move(move)
        policy[move_idx] = prob
    
    return policy


def get_legal_moves_mask(board: ChessBoard):
    """
    Create mask of legal moves
    
    Args:
        board: ChessBoard object
        
    Returns:
        numpy.ndarray: Shape (4096,) with 1.0 for legal moves, 0.0 otherwise
    """
    mask = np.zeros(4096, dtype=np.float32)
    
    for move in board.get_legal_moves():
        move_idx = encode_move(move)
        mask[move_idx] = 1.0
    
    return mask


def flip_board_perspective(encoded_board):
    """
    Flip board perspective (for black player)
    
    Args:
        encoded_board: numpy.ndarray shape (18, 8, 8)
        
    Returns:
        numpy.ndarray: Flipped board from opposite perspective
    """
    flipped = np.copy(encoded_board)
    
    # Flip all planes vertically
    flipped = np.flip(flipped, axis=1)
    
    # Swap white and black pieces (planes 0-5 with 6-11)
    temp = np.copy(flipped[0:6])
    flipped[0:6] = flipped[6:12]
    flipped[6:12] = temp
    
    # Flip turn indicator
    flipped[16] = 1.0 - flipped[16]
    
    return flipped


def decode_policy_output(policy_logits, board: ChessBoard, temperature=1.0):
    """
    Decode policy network output to move probabilities
    
    Args:
        policy_logits: numpy.ndarray or torch.Tensor of shape (4096,)
        board: ChessBoard object
        temperature: Temperature parameter for softmax
        
    Returns:
        list: List of (move, probability) tuples for legal moves
    """
    # Convert to numpy if needed
    if hasattr(policy_logits, 'cpu'):
        policy_logits = policy_logits.cpu().numpy()
    
    # Get legal moves mask
    legal_mask = get_legal_moves_mask(board)
    
    # Apply mask (set illegal moves to very negative value)
    masked_logits = np.where(legal_mask > 0, policy_logits, -1e10)
    
    # Apply temperature
    if temperature != 1.0:
        masked_logits = masked_logits / temperature
    
    # Softmax
    exp_logits = np.exp(masked_logits - np.max(masked_logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    # Get legal moves with probabilities
    moves_with_probs = []
    for move in board.get_legal_moves():
        move_idx = encode_move(move)
        moves_with_probs.append((move, probabilities[move_idx]))
    
    # Sort by probability (descending)
    moves_with_probs.sort(key=lambda x: x[1], reverse=True)
    
    return moves_with_probs


def encode_training_sample(board: ChessBoard, policy_target, value_target):
    """
    Encode a complete training sample
    
    Args:
        board: ChessBoard object
        policy_target: Target policy (dict of move -> probability or array)
        value_target: Target value (-1 to 1)
        
    Returns:
        tuple: (encoded_board, policy_vector, value)
    """
    # Encode board state
    board_encoding = encode_board(board)
    
    # Create policy vector
    if isinstance(policy_target, dict):
        policy_vector = np.zeros(4096, dtype=np.float32)
        for move, prob in policy_target.items():
            if isinstance(move, str):
                move = chess.Move.from_uci(move)
            move_idx = encode_move(move)
            policy_vector[move_idx] = prob
    else:
        policy_vector = policy_target
    
    return board_encoding, policy_vector, float(value_target)


def encode_board_history(boards, num_history=2):
    """
    Encode board with move history
    
    Args:
        boards: List of ChessBoard objects (most recent last)
        num_history: Number of historical positions to include
        
    Returns:
        numpy.ndarray: Concatenated encodings
    """
    # Start with most recent board
    current_encoding = encode_board(boards[-1])
    
    if len(boards) > 1 and num_history > 0:
        history_encodings = []
        for i in range(min(num_history, len(boards) - 1)):
            hist_idx = -(i + 2)  # -2, -3, -4, etc.
            if hist_idx >= -len(boards):
                history_encodings.append(encode_board(boards[hist_idx]))
        
        if history_encodings:
            full_encoding = np.concatenate([current_encoding] + history_encodings, axis=0)
            return full_encoding
    
    return current_encoding