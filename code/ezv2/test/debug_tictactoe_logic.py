import sys
import pathlib
import jax
import jax.numpy as jnp

# Setup path
TEST_PATH = pathlib.Path(__file__).resolve().parent
SRC_PATH = TEST_PATH.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(TEST_PATH) not in sys.path:
    sys.path.insert(0, str(TEST_PATH))

import tictactoe_env as ttt

def debug_logic():
    # Reconstruct the problematic board state
    # Board indices:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    
    # History from chat:
    # X: 8, 5, 7
    # O: 4, 2
    # Turn: O (current_player = -1)
    
    board = jnp.zeros((9,), dtype=jnp.float32)
    board = board.at[8].set(1.0)
    board = board.at[4].set(-1.0)
    board = board.at[5].set(1.0)
    board = board.at[2].set(-1.0)
    board = board.at[7].set(1.0)
    
    state = ttt.TicTacToeState(board=board, current_player=-1)
    
    print("Board state:")
    print(board.reshape(3, 3))
    print(f"Current player: {state.current_player}")
    
    # Check if 6 is a winning move
    print("\nChecking move 6...")
    next_board_6 = board.at[6].set(-1.0)
    winner_6 = ttt.check_winner(next_board_6)
    print(f"Winner after playing 6: {winner_6}")
    
    # Check _winning_moves
    winning_mask = ttt._winning_moves(board, -1)
    print(f"Winning moves mask for O (-1): {winning_mask}")
    
    # Check logits
    logits = ttt.heuristic_prior_logits(state)
    print(f"\nLogits: {logits}")
    
    # Check probability
    probs = jax.nn.softmax(logits)
    print(f"Probs: {probs}")
    
    # Check values for MCTS
    # If MCTS expands node 6, what is the value?
    next_state_6 = ttt.TicTacToeState(next_board_6, current_player=1)
    # Note: transition_reward should handle the win
    reward = ttt.transition_reward(state, next_state_6, 6)
    print(f"\nTransition reward for 6: {reward}")
    
    val = ttt.evaluate_state(next_state_6)
    print(f"Leaf value for state after 6: {val}")

if __name__ == "__main__":
    debug_logic()

