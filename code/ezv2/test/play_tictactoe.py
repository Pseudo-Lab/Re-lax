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
search_visit_policy = mcts_policy.search_visit_policy


def build_mcts(max_nodes: int = 256) -> Tuple[type, MCTSCallbacks]:
    simulation = ttt.get_simulation()
    embedding_cls = simulation.embedding_state_cls
    action_space = simulation.action_space
    metadata = action_space.metadata()
    print(
        f"[MCTS] action space kind={metadata.kind}, size={metadata.size}, shape={metadata.shape}"
    )
    node_cls = make_node_class(embedding_cls, action_space)
    tree_cls = make_tree_class(node_cls, max_nodes, action_space.get_shape())

    callbacks = simulation.make_callbacks()
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
    state = ttt.get_simulation().initial_state()
    rng_key = jax.random.PRNGKey(0)

    print("You are X (indices 0-8, row-major). Enter 'q' to quit.\n")
    render_board(state)

    # JIT compile the policy function
    jit_search_policy = jax.jit(
        search_visit_policy, 
        static_argnames=("tree_cls", "callbacks", "num_simulations", "return_gumbel_noise")
    )

    while True:
        if state.current_player == 1:
            action = prompt_human_action(state)
            state = state.apply_action(action)
            print(f"You played {action}")
        else:
            rng_key, subkey = jax.random.split(rng_key)
            # Increase simulations for stronger play
            # Lower temperature (e.g. 0.1 or 0.0) makes it play more greedily w.r.t visit counts
            output = jit_search_policy(
                tree_cls=tree_cls,
                callbacks=callbacks,
                root_state=state,
                rng_key=subkey,
                num_simulations=200,
                temperature=0.1, 
            )
            state = state.apply_action(output.action)
            # Note: JITted function returns JAX arrays, so we might see DeviceArray in print
            print(f"AI played {output.action} (visits: {output.action_weights})")

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

