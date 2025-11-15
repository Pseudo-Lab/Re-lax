"""Simple CLI for playing TicTacToe against the MCTS policy."""

from __future__ import annotations

import importlib
import pathlib
import sys
from typing import Tuple

import jax
import jax.numpy as jnp

TEST_PATH = pathlib.Path(__file__).resolve().parent
SRC_PATH = TEST_PATH.parent / "src"

for path in (SRC_PATH, TEST_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

ttt = importlib.import_module("tictactoe_env")
mcts_node = importlib.import_module("mcts.node")
mcts_tree = importlib.import_module("mcts.tree")
mcts_policy = importlib.import_module("mcts.policy")

make_node_class = mcts_node.make_node_class
MCTSCallbacks = mcts_tree.MCTSCallbacks
make_tree_class = mcts_tree.make_tree_class
muzero_policy = mcts_policy.muzero_policy


def build_mcts(max_nodes: int = 256) -> Tuple[type, MCTSCallbacks]:
    embedding_cls = ttt.make_tictactoe_embedding_state()
    action_space = ttt.make_tictactoe_action_space()
    node_cls = make_node_class(embedding_cls, action_space)
    tree_cls = make_tree_class(node_cls, max_nodes, action_space.get_shape())

    callbacks = MCTSCallbacks(
        encode=ttt.encode_state,
        decode=ttt.decode_state,
        invalid_actions=lambda state: ttt.mask_invalid_actions(state.board),
        apply_action=lambda state, action: state.apply_action(int(action)),
        is_terminal=lambda state: state.is_draw() or state.winner() != 0,
        transition_reward=ttt.transition_reward,
        value=ttt.evaluate_state,
    )
    return tree_cls, callbacks


def render_board(state: ttt.TicTacToeState) -> None:
    symbols = {1.0: "X", -1.0: "O", 0.0: "."}
    board = jnp.asarray(state.board).reshape((3, 3))
    for row in board:
        print(" ".join(symbols[float(cell)] for cell in row))
    print()


def prompt_human_action(state: ttt.TicTacToeState) -> int:
    mask = ttt.mask_invalid_actions(state.board)
    legal = [idx for idx in range(ttt.NUM_CELLS) if mask[idx] == 0]
    while True:
        try:
            move = input(f"Choose your move {legal}: ").strip()
            if move.lower() in {"q", "quit", "exit"}:
                raise KeyboardInterrupt
            action = int(move)
        except ValueError:
            print("Please enter a number between 0 and 8.")
            continue
        if action in legal:
            return action
        print("Invalid move; that cell is already occupied.")


def play_game():
    tree_cls, callbacks = build_mcts()
    state = ttt.TicTacToeState.empty()
    rng_key = jax.random.PRNGKey(0)

    print("You are X (indices 0-8, row-major). Enter 'q' to quit.\n")
    render_board(state)

    while True:
        if state.current_player == 1:
            action = prompt_human_action(state)
            state = state.apply_action(action)
            print(f"You played {action}")
        else:
            rng_key, subkey = jax.random.split(rng_key)
            output = muzero_policy(
                tree_cls=tree_cls,
                callbacks=callbacks,
                root_state=state,
                rng_key=subkey,
                num_simulations=32,
            )
            state = state.apply_action(output.action)
            print(f"AI played {output.action}")

        render_board(state)

        winner = state.winner()
        if winner != 0:
            print("You win!" if winner == 1 else "AI wins!")
            break
        if state.is_draw():
            print("Draw!")
            break


if __name__ == "__main__":
    try:
        play_game()
    except KeyboardInterrupt:
        print("\nGame aborted.")

