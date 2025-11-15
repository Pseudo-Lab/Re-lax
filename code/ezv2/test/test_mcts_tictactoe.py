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
search_visit_policy = policy_mod.search_visit_policy
select_action = action_selection_mod.select_action


def setup_tree(max_nodes: int = 64) -> Tuple[object, MCTSCallbacks, type]:
    simulation = ttt.get_simulation()
    embedding_cls = simulation.embedding_state_cls
    action_space = simulation.action_space
    metadata = action_space.metadata()
    assert metadata.kind == "discrete"
    assert metadata.size == action_space.get_size()
    assert metadata.shape == action_space.get_shape()
    node_cls = make_node_class(embedding_cls, action_space)
    tree_cls = make_tree_class(node_cls, max_nodes, action_space.get_shape())

    callbacks = simulation.make_callbacks()
    tree = tree_cls.from_root(callbacks, simulation.initial_state())
    return tree, callbacks, tree_cls


def test_action_space_metadata_reflects_branching():
    action_space = ttt.make_tictactoe_action_space()
    metadata = action_space.metadata()
    assert metadata.kind == "discrete"
    assert metadata.size == ttt.NUM_CELLS
    assert metadata.nvec == (ttt.NUM_CELLS,)


def test_action_mask_coercion_validates_shape():
    _, callbacks, tree_cls = setup_tree(max_nodes=32)
    state = ttt.TicTacToeState.empty()
    mask = ttt.mask_invalid_actions(state.board)
    coerced = tree_cls._coerce_action_mask(mask, context="test")
    assert coerced.shape == tree_cls._action_shape
    try:
        tree_cls._coerce_action_mask(mask[:-1], context="bad-shape")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for malformed action mask.")


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


def test_search_visit_policy_returns_valid_action():
    _, callbacks, tree_cls = setup_tree(max_nodes=64)
    rng = jax.random.PRNGKey(0)
    output = search_visit_policy(
        tree_cls=tree_cls,
        callbacks=callbacks,
        root_state=ttt.TicTacToeState.empty(),
        rng_key=rng,
        num_simulations=4,
    )
    assert 0 <= output.action < ttt.NUM_CELLS
    assert output.action_weights.shape[0] == ttt.NUM_CELLS
    assert jnp.isclose(jnp.sum(output.action_weights), 1.0, atol=1e-6)
    assert output.gumbel_extra is None


def test_search_visit_policy_can_capture_gumbel_noise():
    _, callbacks, tree_cls = setup_tree(max_nodes=64)
    rng = jax.random.PRNGKey(42)
    output = search_visit_policy(
        tree_cls=tree_cls,
        callbacks=callbacks,
        root_state=ttt.TicTacToeState.empty(),
        rng_key=rng,
        num_simulations=4,
        return_gumbel_noise=True,
    )
    assert output.gumbel_extra is not None
    assert output.gumbel_extra.root_gumbel.shape[0] == ttt.NUM_CELLS


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


def test_select_action_with_extras_returns_metadata():
    tree, callbacks, _ = setup_tree(max_nodes=64)
    tree, _ = tree.run_search(callbacks, num_iterations=6)
    result = select_action(
        tree=tree,
        node_index=ROOT_INDEX,
        depth=0,
        rng_key=jax.random.PRNGKey(555),
        with_extras=True,
    )
    assert result.action >= 0
    assert result.root_invalid_mask is not None
    if result.gumbel_extra is not None:
        assert result.gumbel_extra.root_gumbel.shape == result.root_invalid_mask.shape


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

