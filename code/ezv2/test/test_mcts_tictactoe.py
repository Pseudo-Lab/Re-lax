import importlib
import pathlib
import sys
from typing import Tuple

import jax
import jax.numpy as jnp

SRC_PATH = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

TEST_PATH = pathlib.Path(__file__).resolve().parent
if str(TEST_PATH) not in sys.path:
    sys.path.insert(0, str(TEST_PATH))

import tictactoe_env as ttt

annotate_mod = importlib.import_module("mcts.annotate")
node_mod = importlib.import_module("mcts.node")
tree_mod = importlib.import_module("mcts.tree")
policy_mod = importlib.import_module("mcts.policy")
action_selection_mod = importlib.import_module("mcts.action_selection")

ROOT_INDEX = annotate_mod.ROOT_INDEX
UNVISITED = annotate_mod.UNVISITED
make_node_class = node_mod.make_node_class
MCTSCallbacks = tree_mod.MCTSCallbacks
make_tree_class = tree_mod.make_tree_class
muzero_policy = policy_mod.muzero_policy
select_action = action_selection_mod.select_action


def setup_tree(max_nodes: int = 64) -> Tuple[object, MCTSCallbacks, type]:
    embedding_cls = ttt.make_tictactoe_embedding_state()
    action_space = ttt.make_tictactoe_action_space()
    node_cls = make_node_class(embedding_cls, action_space)
    tree_cls = make_tree_class(node_cls, max_nodes, action_space.get_shape())

    callbacks = MCTSCallbacks(
        encode=ttt.encode_state,
        decode=ttt.decode_state,
        invalid_actions=lambda state: ttt.mask_invalid_actions(state.board),
        apply_action=lambda state, action: state.apply_action(action),
        is_terminal=lambda state: state.is_draw() or state.winner() != 0,
        transition_reward=ttt.transition_reward,
        value=ttt.evaluate_state,
    )
    tree = tree_cls.from_root(callbacks, ttt.TicTacToeState.empty())
    return tree, callbacks, tree_cls


def test_reset_masks_invalid_actions():
    _, callbacks, tree_cls = setup_tree(max_nodes=32)
    board = jnp.array([1, 0, -1, 0, 1, 0, -1, 0, 0], dtype=jnp.float32)
    state = ttt.TicTacToeState(board=board, current_player=-1)
    tree = tree_cls.from_root(callbacks, state)
    assert jnp.array_equal(tree.root_invalid_actions, ttt.mask_invalid_actions(board))


def test_iteration_updates_visits_and_children():
    tree, callbacks, _ = setup_tree(max_nodes=64)
    tree, traces = tree.run_search(callbacks, num_iterations=6)
    root = tree.nodes[ROOT_INDEX]
    assert int(root.visit_count) == 6
    assert int(tree.next_free_idx) > 1
    assert any(trace.expanded_action is not None for trace in traces)


def test_illegal_actions_remain_unvisited():
    _, callbacks, tree_cls = setup_tree(max_nodes=64)
    board = jnp.array([1, -1, 1, 0, -1, 0, 0, 0, 0], dtype=jnp.float32)
    state = ttt.TicTacToeState(board=board, current_player=1)
    tree = tree_cls.from_root(callbacks, state)
    tree, _ = tree.run_search(callbacks, num_iterations=4)
    root = tree.nodes[ROOT_INDEX]
    assert jnp.all(root.children_idx[:3] == UNVISITED)


def test_ranked_actions_match_visit_sort():
    tree, callbacks, _ = setup_tree(max_nodes=64)
    tree, _ = tree.run_search(callbacks, num_iterations=8)
    ranking = tree.ranked_actions()
    stats = tree.get_action_statistics()
    last_visits = float("inf")
    for action in ranking:
        visits = stats[action]["visits"]
        assert visits <= last_visits + 1e-6
        last_visits = visits


def test_muzero_policy_returns_valid_action():
    _, callbacks, tree_cls = setup_tree(max_nodes=64)
    rng = jax.random.PRNGKey(0)
    output = muzero_policy(
        tree_cls=tree_cls,
        callbacks=callbacks,
        root_state=ttt.TicTacToeState.empty(),
        rng_key=rng,
        num_simulations=4,
    )
    assert 0 <= output.action < ttt.NUM_CELLS
    assert output.action_weights.shape[0] == ttt.NUM_CELLS
    assert jnp.isclose(jnp.sum(output.action_weights), 1.0, atol=1e-6)


def test_select_action_defaults_to_gumbel():
    tree, callbacks, _ = setup_tree(max_nodes=64)
    tree, _ = tree.run_search(callbacks, num_iterations=6)
    action = select_action(
        tree=tree,
        node_index=ROOT_INDEX,
        depth=0,
        rng_key=jax.random.PRNGKey(123),
    )
    assert 0 <= action < ttt.NUM_CELLS


def test_select_action_puct_strategy():
    tree, callbacks, _ = setup_tree(max_nodes=64)
    tree, _ = tree.run_search(callbacks, num_iterations=6)
    action = select_action(
        tree=tree,
        node_index=ROOT_INDEX,
        depth=0,
        rng_key=jax.random.PRNGKey(999),
        strategy="puct",
    )
    assert 0 <= action < ttt.NUM_CELLS

