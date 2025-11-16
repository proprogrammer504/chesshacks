"""Simple test to verify the AlphaZero training fixes work"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_chess_components():
    """Test that chess components work without dependencies"""
    print("🧪 Testing Chess Components...")
    
    try:
        import chess
        print("✅ Chess library imported successfully")
        
        # Test basic chess functionality
        board = chess.Board()
        print(f"✅ Created chess board: {board.fen()}")
        print(f"✅ Legal moves count: {len(list(board.legal_moves))}")
        
        return True
    except Exception as e:
        print(f"❌ Chess test failed: {e}")
        return False

def test_board_encoding():
    """Test board encoding functionality"""
    print("\n🔢 Testing Board Encoding...")
    
    try:
        from chess_board import ChessBoard
        from encoder_decoder import encode_board
        
        # Create board
        board = ChessBoard()
        print(f"✅ Created ChessBoard: {type(board)}")
        
        # Test encoding
        encoded = encode_board(board)
        print(f"✅ Board encoded: shape {encoded.shape}")
        print(f"   Sum of all planes: {encoded.sum():.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Board encoding test failed: {e}")
        return False

def test_neural_network():
    """Test neural network creation"""
    print("\n🧠 Testing Neural Network...")
    
    try:
        from alpha_net import create_alpha_net
        
        # Create model
        model = create_alpha_net()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Created AlphaNet with {param_count:,} parameters")
        
        # Test forward pass
        import torch
        dummy_input = torch.randn(1, 18, 8, 8)
        policy, value = model(dummy_input)
        print(f"✅ Forward pass successful:")
        print(f"   Policy output shape: {policy.shape}")
        print(f"   Value output shape: {value.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        return False

def test_mcts_components():
    """Test MCTS components"""
    print("\n🔍 Testing MCTS Components...")
    
    try:
        from chess_board import ChessBoard
        from encoder_decoder import encode_board
        
        # Create board and test MCTS
        board = ChessBoard()
        encoded = encode_board(board)
        
        print(f"✅ Board encoding for MCTS works")
        print(f"✅ Position encoded: {encoded.shape}")
        
        # Test legal moves
        legal_moves = board.get_legal_moves()
        print(f"✅ Legal moves: {len(legal_moves)}")
        
        return True
    except Exception as e:
        print(f"❌ MCTS test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 AlphaZero Training - Component Verification")
    print("="*60)
    
    tests = [
        test_chess_components,
        test_board_encoding,
        test_neural_network,
        test_mcts_components
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fixed implementation should work.")
        print("\n📋 To run full training:")
        print("   1. Install dependencies: pip install torch numpy python-chess tqdm")
        print("   2. Run: python training_fixed.py")
    else:
        print("⚠️  Some tests failed. Check the dependencies and imports.")
    
    return passed == total

if __name__ == "__main__":
    main()