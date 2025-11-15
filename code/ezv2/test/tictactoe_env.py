from __future__ import annotations

from dataclasses import dataclass
import importlib
import pathlib
import sys

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
        if board[action_idx] != 0.0:
            raise ValueError(f"Illegal TicTacToe move at cell {action_idx}.")
        next_board = board.at[action_idx].set(float(self.current_player))
        return TicTacToeState(next_board, -self.current_player)

    def winner(self) -> int:
        return check_winner(self.board)

    def is_draw(self) -> bool:
        return is_draw(self.board)


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
    player = int(jnp.asarray(embedding[NUM_CELLS], dtype=jnp.float32))
    return TicTacToeState(board=board, current_player=player)


def mask_invalid_actions(board: jnp.ndarray) -> jnp.ndarray:
    flat = jnp.asarray(board, dtype=jnp.float32).reshape((NUM_CELLS,))
    return jnp.where(flat == 0.0, 0, 1).astype(jnp.uint8)


def legal_action_indices(board: jnp.ndarray) -> jnp.ndarray:
    mask = mask_invalid_actions(board)
    return jnp.where(mask == 0)[0]


def check_winner(board: jnp.ndarray) -> int:
    flat = jnp.asarray(board, dtype=jnp.float32).reshape((NUM_CELLS,))
    for (a, b, c) in WIN_PATTERNS:
        line_sum = flat[a] + flat[b] + flat[c]
        if line_sum == 3.0:
            return +1
        if line_sum == -3.0:
            return -1
    return 0


def is_draw(board: jnp.ndarray) -> bool:
    flat = jnp.asarray(board, dtype=jnp.float32).reshape((NUM_CELLS,))
    return jnp.all(flat != 0.0)


def transition_reward(parent_state: TicTacToeState, child_state: TicTacToeState, _: int) -> float:
    winner = check_winner(child_state.board)
    if winner == 0:
        return 0.0
    return float(winner * parent_state.current_player)


def evaluate_state(state: TicTacToeState) -> float:
    """Heuristic value: wins/losses plus line/center bonuses."""

    winner = check_winner(state.board)
    if winner != 0:
        return float(-winner * state.current_player)
    if is_draw(state.board):
        return 0.0

    board = jnp.asarray(state.board)
    player = state.current_player
    friend_score = _player_line_score(board, player)
    foe_score = _player_line_score(board, -player)
    center_bonus = 0.25 if board[4] == player else (-0.25 if board[4] == -player else 0.0)
    return float(0.2 * (friend_score - foe_score) + center_bonus)


def heuristic_prior_logits(state: TicTacToeState) -> jnp.ndarray:
    """Returns simple prior logits favoring wins, blocks, center, and forks."""

    board = jnp.asarray(state.board)
    mask = mask_invalid_actions(board)
    logits = jnp.full((NUM_CELLS,), -jnp.inf, dtype=jnp.float32)

    friend = state.current_player
    foe = -friend
    opponent_winning_moves = _winning_moves(board, foe)

    for idx in range(NUM_CELLS):
        if mask[idx]:
            continue

        next_board = board.at[idx].set(float(friend))
        win_score = 3.0 if check_winner(next_board) == friend else 0.0

        block_score = 2.5 if idx in opponent_winning_moves else 0.0

        line_advantage = 0.1 * (
            _player_line_score(next_board, friend) - _player_line_score(next_board, foe)
        )
        center_bonus = 0.5 if idx == 4 else 0.0
        logits = logits.at[idx].set(
            jnp.float32(win_score + block_score + line_advantage + center_bonus)
        )

    return logits


def _player_line_score(board: jnp.ndarray, player: int) -> float:
    score = 0.0
    for (a, b, c) in WIN_PATTERNS:
        line = jnp.array([board[a], board[b], board[c]])
        if jnp.any(line == -player):
            continue
        score += float(jnp.sum(line == player))
    return score


def _winning_moves(board: jnp.ndarray, player: int) -> set[int]:
    mask = mask_invalid_actions(board)
    winning = set()
    for idx in range(NUM_CELLS):
        if mask[idx]:
            continue
        candidate = board.at[idx].set(float(player))
        if check_winner(candidate) == player:
            winning.add(idx)
    return winning


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
        return state.is_draw() or state.winner() != 0

    def transition_reward(
        self, parent_state: TicTacToeState, child_state: TicTacToeState, action: int
    ) -> float:
        return transition_reward(parent_state, child_state, action)

    def value(self, state: TicTacToeState) -> float:
        return evaluate_state(state)


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

