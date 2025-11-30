from __future__ import annotations

from dataclasses import dataclass
import importlib
import pathlib
import sys

import jax
import jax.numpy as jnp

TEST_PATH = pathlib.Path(__file__).resolve().parent
SRC_PATH = TEST_PATH.parent / "src"
for path in (SRC_PATH, TEST_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

action_space_mod = importlib.import_module("mcts.action_space")
node_mod = importlib.import_module("mcts.node")
simulation_mod = importlib.import_module("simulation.base")

DiscreteActionSpace = action_space_mod.DiscreteActionSpace
make_embedding_state_class = node_mod.make_embedding_state_class
SimulationModel = simulation_mod.SimulationModel

BOARD_SIZE = 3
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
EMBEDDING_SHAPE = (NUM_CELLS + 1,)
WIN_PATTERNS = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)
@dataclass(frozen=True)
class TicTacToeState:
    board: jnp.ndarray  # shape: (NUM_CELLS,), values in {-1, 0, 1}
    current_player: int  # +1 for X, -1 for O

    @staticmethod
    def empty() -> "TicTacToeState":
        return TicTacToeState(jnp.zeros((NUM_CELLS,), dtype=jnp.float32), +1)

    def apply_action(self, action_idx: int) -> "TicTacToeState":
        board = jnp.asarray(self.board)
        # Handle both Python int (eager) and JAX array (JIT)
        val = jnp.array(self.current_player, dtype=jnp.float32)
        next_board = board.at[action_idx].set(val)
        return TicTacToeState(next_board, -self.current_player)

    def winner(self) -> int:
        return check_winner(self.board)

    def is_draw(self) -> bool:
        return is_draw(self.board)

# Register TicTacToeState as a Pytree
jax.tree_util.register_pytree_node(
    TicTacToeState,
    lambda s: ((s.board, s.current_player), None),
    lambda aux, children: TicTacToeState(children[0], children[1])
)

def encode_state(state: TicTacToeState) -> jnp.ndarray:
    return jnp.concatenate(
        [
            jnp.asarray(state.board, dtype=jnp.float32).reshape((NUM_CELLS,)),
            jnp.asarray([state.current_player], dtype=jnp.float32),
        ],
        axis=0,
    )


def decode_state(embedding: jnp.ndarray) -> TicTacToeState:
    board = jnp.asarray(embedding[:NUM_CELLS], dtype=jnp.float32)
    # Do not cast to Python int inside JIT/scan; keep as JAX array or use astype
    player = jnp.asarray(embedding[NUM_CELLS], dtype=jnp.int32) 
    return TicTacToeState(board=board, current_player=player)


def mask_invalid_actions(board: jnp.ndarray) -> jnp.ndarray:
    flat = jnp.asarray(board, dtype=jnp.float32).reshape((NUM_CELLS,))
    return jnp.where(flat == 0.0, 0, 1).astype(jnp.uint8)


def legal_action_indices(board: jnp.ndarray) -> jnp.ndarray:
    mask = mask_invalid_actions(board)
    return jnp.where(mask == 0)[0]


def check_winner(board: jnp.ndarray) -> int:
    flat = jnp.asarray(board, dtype=jnp.float32).reshape((NUM_CELLS,))
    
    # Scan all winning patterns
    # This loop is small (8 iters) and can be unrolled or vectorized.
    # Since we need to return early in Python if, we must change logic for JIT.
    
    # Vectorized approach:
    # Gather all lines into a matrix of shape (8, 3)
    patterns = jnp.array(WIN_PATTERNS) # (8, 3)
    lines = flat[patterns] # (8, 3)
    line_sums = jnp.sum(lines, axis=1) # (8,)
    
    # Check if any line sum is 3.0 or -3.0
    has_x_won = jnp.any(line_sums == 3.0)
    has_o_won = jnp.any(line_sums == -3.0)
    
    # Return 1 if X won, -1 if O won, else 0
    # If both won (impossible in valid game), this prioritizes X (or whatever select order)
    # But we can just sum them: if X won (1), O won (0) -> 1. If O won (-1) -> -1.
    
    winner = jax.lax.select(
        has_x_won, 
        jnp.array(1, dtype=jnp.int32), 
        jax.lax.select(has_o_won, jnp.array(-1, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))
    )
    return winner



def is_draw(board: jnp.ndarray) -> bool:
    flat = jnp.asarray(board, dtype=jnp.float32).reshape((NUM_CELLS,))
    return jnp.all(flat != 0.0)


def transition_reward(parent_state: TicTacToeState, child_state: TicTacToeState, _: int) -> float:
    winner = check_winner(child_state.board)
    # winner is 0 if no winner, else 1 or -1.
    # No 'if' check allowed in JIT.
    return (winner * parent_state.current_player).astype(jnp.float32)


def evaluate_state(state: TicTacToeState) -> float:
    """Heuristic value: wins/losses plus line/center bonuses."""

    winner = check_winner(state.board)
    # winner is tracer. Use select/logic to handle terminal check.
    is_terminal = jnp.logical_or(winner != 0, is_draw(state.board))

    board = jnp.asarray(state.board)
    player = state.current_player
    friend_score = _player_line_score(board, player)
    foe_score = _player_line_score(board, -player)
    
    center_val = board[4]
    center_bonus = jax.lax.select(
        center_val == player, 
        0.25, 
        jax.lax.select(center_val == -player, -0.25, 0.0)
    )
    
    raw_score = 0.2 * (friend_score - foe_score) + center_bonus
    heuristic = jnp.clip(raw_score, -1.0, 1.0).astype(jnp.float32)
    
    # If terminal, value should be 0.0 (handled by transition_reward)
    return jax.lax.select(is_terminal, 0.0, heuristic)


def heuristic_prior_logits(state: TicTacToeState) -> jnp.ndarray:
    """Returns simple prior logits favoring wins, blocks, center, and forks."""

    board = jnp.asarray(state.board)
    mask = mask_invalid_actions(board)
    
    # Base logits: -inf
    logits = jnp.full((NUM_CELLS,), -jnp.inf, dtype=jnp.float32)

    friend = state.current_player
    foe = -friend
    
    # Get winning masks (vectorized)
    opponent_winning_mask = _winning_moves(board, foe)
    my_winning_mask = _winning_moves(board, friend)

    # Vectorized scoring function per move
    def score_move(idx):
        # Basic validity check is done by mask later, but useful for logic
        
        # 1. Win score
        win_score = jax.lax.select(my_winning_mask[idx], 3.0, 0.0)
        
        # 2. Block score
        block_score = jax.lax.select(opponent_winning_mask[idx], 2.5, 0.0)
        
        # 3. Line advantage
        # Look ahead one step
        next_board = board.at[idx].set(jnp.array(friend, dtype=jnp.float32))
        line_adv = 0.1 * (
            _player_line_score(next_board, friend) - _player_line_score(next_board, foe)
        )
        
        # 4. Center bonus
        center_bonus = jax.lax.select(idx == 4, 0.5, 0.0)
        
        return win_score + block_score + line_adv + center_bonus

    # Compute scores for all cells
    scores = jax.vmap(score_move)(jnp.arange(NUM_CELLS))
    
    # Apply invalid mask (set invalid to -inf)
    valid_scores = jnp.where(mask == 0, scores, -jnp.inf)
    
    return valid_scores.astype(jnp.float32)


def _player_line_score(board: jnp.ndarray, player: int) -> float:
    # Vectorized implementation for JIT compatibility
    patterns = jnp.array(WIN_PATTERNS)
    lines = board[patterns] # (8, 3)
    
    # Check if opponent is present in line
    has_foe = jnp.any(lines == -player, axis=1) # (8,)
    
    # Count my pieces in line
    my_pieces = jnp.sum(lines == player, axis=1) # (8,)
    
    # Score: sum of pieces where no foe
    valid_lines = jnp.logical_not(has_foe)
    score = jnp.sum(jnp.where(valid_lines, my_pieces, 0.0))
    
    return score.astype(jnp.float32)


def _winning_moves(board: jnp.ndarray, player: int) -> jnp.ndarray:
    # Must return fixed-size array (mask) instead of set for JIT
    mask = mask_invalid_actions(board)
    
    # Try all moves
    def check_move(idx):
        is_invalid = mask[idx]
        # If invalid, can't win here.
        # If valid, place piece and check winner
        # Use JAX array for set value
        candidate = board.at[idx].set(jnp.array(player, dtype=jnp.float32))
        is_winner = (check_winner(candidate) == player)
        return jnp.logical_and(jnp.logical_not(is_invalid), is_winner)
        
    # vmap over all indices
    winning_mask = jax.vmap(check_move)(jnp.arange(NUM_CELLS))
    return winning_mask


class TicTacToeSimulation(SimulationModel[TicTacToeState]):
    """Simulation wrapper exposing TicTacToe as a MuZero-style model."""

    def __init__(self):
        self._action_space = DiscreteActionSpace(NUM_CELLS)
        self._embedding_cls = make_embedding_state_class(jnp.float32, EMBEDDING_SHAPE)

    @property
    def action_space(self) -> DiscreteActionSpace:
        return self._action_space

    @property
    def embedding_state_cls(self):
        return self._embedding_cls

    def initial_state(self) -> TicTacToeState:
        return TicTacToeState.empty()

    def encode(self, state: TicTacToeState) -> jnp.ndarray:
        return encode_state(state)

    def decode(self, embedding: jnp.ndarray) -> TicTacToeState:
        return decode_state(embedding)

    def invalid_actions(self, state: TicTacToeState) -> jnp.ndarray:
        return mask_invalid_actions(state.board)

    def apply_action(self, state: TicTacToeState, action: int) -> TicTacToeState:
        return state.apply_action(action)

    def is_terminal(self, state: TicTacToeState) -> bool:
        # Use jnp.logical_or to work inside JIT/scan tracers
        return jnp.logical_or(state.is_draw(), state.winner() != 0)

    def transition_reward(
        self, parent_state: TicTacToeState, child_state: TicTacToeState, action: int
    ) -> float:
        return transition_reward(parent_state, child_state, action)

    def value(self, state: TicTacToeState) -> float:
        return evaluate_state(state)

    def policy(self, state: TicTacToeState) -> jnp.ndarray:
        return heuristic_prior_logits(state)


_SIMULATION = TicTacToeSimulation()


def get_simulation() -> TicTacToeSimulation:
    """Return the shared TicTacToe simulation model."""

    return _SIMULATION


def make_tictactoe_embedding_state():
    return _SIMULATION.embedding_state_cls


def make_tictactoe_action_space() -> DiscreteActionSpace:
    return _SIMULATION.action_space


def make_tictactoe_callbacks():
    return _SIMULATION.make_callbacks()
