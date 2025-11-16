"""Chess board implementation with game rules and move generation"""

import chess
import numpy as np


class ChessBoard:
    """Wrapper class for chess.Board with additional functionality"""
    
    def __init__(self, board=None):
        """
        Initialize chess board
        
        Args:
            board: Optional chess.Board object, creates new game if None
        """
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board.copy()
    
    def reset(self):
        """Reset board to starting position"""
        self.board.reset()
    
    def copy(self):
        """Return a copy of the current board"""
        return ChessBoard(self.board)
    
    def get_legal_moves(self):
        """
        Get list of all legal moves in current position
        
        Returns:
            list: List of chess.Move objects
        """
        return list(self.board.legal_moves)
    
    def get_legal_moves_count(self):
        """Get count of legal moves"""
        return self.board.legal_moves.count()
    
    def make_move(self, move):
        """
        Make a move on the board
        
        Args:
            move: chess.Move object or UCI string
            
        Returns:
            bool: True if move was legal and executed, False otherwise
        """
        if isinstance(move, str):
            try:
                move = chess.Move.from_uci(move)
            except:
                return False
        
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False
    
    def undo_move(self):
        """Undo last move"""
        if len(self.board.move_stack) > 0:
            self.board.pop()
            return True
        return False
    
    def is_game_over(self):
        """Check if game is over"""
        return self.board.is_game_over()
    
    def get_result(self):
        """
        Get game result
        
        Returns:
            str: "1-0" (white wins), "0-1" (black wins), "1/2-1/2" (draw), or "*" (ongoing)
        """
        if not self.is_game_over():
            return "*"
        
        result = self.board.result()
        return result
    
    def get_winner(self):
        """
        Get winner of the game
        
        Returns:
            int: 1 (white wins), -1 (black wins), 0 (draw), None (ongoing)
        """
        if not self.is_game_over():
            return None
        
        result = self.get_result()
        if result == "1-0":
            return 1
        elif result == "0-1":
            return -1
        else:
            return 0
    
    def get_value_from_perspective(self, player):
        """
        Get game value from a player's perspective
        
        Args:
            player: 1 for white, -1 for black
            
        Returns:
            float: 1.0 (win), -1.0 (loss), 0.0 (draw), None (ongoing)
        """
        winner = self.get_winner()
        if winner is None:
            return None
        return float(winner * player)
    
    def is_checkmate(self):
        """Check if current position is checkmate"""
        return self.board.is_checkmate()
    
    def is_stalemate(self):
        """Check if current position is stalemate"""
        return self.board.is_stalemate()
    
    def is_insufficient_material(self):
        """Check if position has insufficient material for checkmate"""
        return self.board.is_insufficient_material()
    
    def is_check(self):
        """Check if current player is in check"""
        return self.board.is_check()
    
    def get_turn(self):
        """
        Get current player to move
        
        Returns:
            int: 1 for white, -1 for black
        """
        return 1 if self.board.turn == chess.WHITE else -1
    
    def get_fen(self):
        """Get FEN string representation of board"""
        return self.board.fen()
    
    def set_fen(self, fen):
        """Set board from FEN string"""
        self.board.set_fen(fen)
    
    def get_piece_at(self, square):
        """
        Get piece at square
        
        Args:
            square: chess.Square or integer (0-63)
            
        Returns:
            chess.Piece or None
        """
        return self.board.piece_at(square)
    
    def get_board_array(self):
        """
        Get numerical representation of board
        
        Returns:
            numpy.ndarray: 8x8 array with piece values
                Positive values for white pieces, negative for black
                1: Pawn, 2: Knight, 3: Bishop, 4: Rook, 5: Queen, 6: King
        """
        board_array = np.zeros((8, 8), dtype=np.int8)
        
        piece_to_value = {
            chess.PAWN: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.ROOK: 4,
            chess.QUEEN: 5,
            chess.KING: 6
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                rank = square // 8
                file = square % 8
                value = piece_to_value[piece.piece_type]
                if piece.color == chess.BLACK:
                    value = -value
                board_array[rank][file] = value
        
        return board_array
    
    def __str__(self):
        """String representation of board"""
        return str(self.board)
    
    def __repr__(self):
        """String representation for debugging"""
        return f"ChessBoard('{self.get_fen()}')"
    
    def get_move_count(self):
        """Get number of moves made in the game"""
        return len(self.board.move_stack)
    
    def can_claim_draw(self):
        """Check if current player can claim a draw"""
        return (self.board.can_claim_fifty_moves() or 
                self.board.can_claim_threefold_repetition())
    
    def is_repetition(self, count=3):
        """Check if position has been repeated"""
        return self.board.is_repetition(count)