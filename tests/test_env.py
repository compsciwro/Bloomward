# ============================================================
# tests/test_env.py
#
# Test suite for BloomwardEnv.
# Run with:  python -m pytest tests/ -v
# ============================================================

import numpy as np
import pytest
import random

from bloomward_env.env import BloomwardEnv
from bloomward_env.board import HexBoard, hex_distance
from bloomward_env.rules import (
    detect_combos, spread_corruption, _all_triangles
)
from bloomward_env.constants import TILE_FERTILE, TILE_CORRUPT, TILE_SACRED


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def env():
    e = BloomwardEnv(render_mode=None)
    e.reset(seed=42)
    return e


# ============================================================
# 1. Gymnasium API compliance
# ============================================================

class TestGymnasiumAPI:

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset(seed=0)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_observation_in_obs_space_after_reset(self, env):
        obs, _ = env.reset(seed=1)
        assert env.observation_space.contains(obs)

    def test_step_returns_five_values(self, env):
        result = env.step(env.action_space.sample())
        assert len(result) == 5

    def test_step_observation_in_obs_space(self, env):
        obs, *_ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)

    def test_reward_is_float(self, env):
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert isinstance(reward, float)

    def test_terminated_truncated_are_bool(self, env):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_action_space_sample_valid_range(self, env):
        for _ in range(50):
            a = env.action_space.sample()
            assert 0 <= a < env._n_tiles


# ============================================================
# 2. Board construction
# ============================================================

class TestBoardConstruction:

    def test_tile_count(self):
        board = HexBoard(radius=3)
        assert board.num_tiles() == 37

    def test_exactly_one_sacred_core(self):
        board  = HexBoard(radius=3)
        sacred = [t for t in board.tiles if t.tile_type == TILE_SACRED]
        assert len(sacred) == 1
        assert sacred[0].q == 0 and sacred[0].r == 0

    def test_outer_ring_corrupt(self):
        board = HexBoard(radius=3)
        outer = [t for t in board.tiles
                 if hex_distance(0, 0, t.q, t.r) == board.radius]
        assert all(t.tile_type == TILE_CORRUPT for t in outer)

    def test_inner_ring_fertile(self):
        board = HexBoard(radius=3)
        inner = [t for t in board.tiles
                 if 0 < hex_distance(0, 0, t.q, t.r) < board.radius]
        assert all(t.tile_type == TILE_FERTILE for t in inner)


# ============================================================
# 3. Tile placement
# ============================================================

class TestPlacement:

    def test_valid_placement_non_negative_reward(self, env):
        placeable = env._board.placeable_indices()
        assert len(placeable) > 0
        _, reward, _, _, info = env.step(placeable[0])
        # +1 placement -1 turn penalty = 0.0 minimum
        assert reward >= 0
        assert not info["invalid_action"]

    def test_invalid_placement_negative_reward(self, env):
        sacred_idx = next(
            i for i, t in enumerate(env._board.tiles)
            if t.tile_type == TILE_SACRED
        )
        _, reward, _, _, info = env.step(sacred_idx)
        assert reward < 0
        assert info["invalid_action"]

    def test_double_placement_invalid(self, env):
        idx = env._board.placeable_indices()[0]
        env.step(idx)
        _, reward, _, _, info = env.step(idx)
        assert info["invalid_action"]
        assert reward < 0


# ============================================================
# 4. Combo detection
# ============================================================

class TestComboDetection:

    def test_no_combo_fresh_board(self):
        board = HexBoard(radius=3)
        assert detect_combos(board) == []

    def test_triangle_combo_detected(self):
        board = HexBoard(radius=3)
        for a, b, c in _all_triangles(board):
            if all(t.tile_type == TILE_FERTILE for t in [a, b, c]):
                a.flower = b.flower = c.flower = 1
                break
        combos = detect_combos(board)
        assert len(combos) >= 1
        assert combos[0]["flower"] == 1

    def test_no_combo_mismatched_flowers(self):
        board = HexBoard(radius=3)
        for a, b, c in _all_triangles(board):
            if all(t.tile_type == TILE_FERTILE for t in [a, b, c]):
                a.flower, b.flower, c.flower = 1, 2, 3
                break
        assert detect_combos(board) == []

    def test_combo_not_double_counted(self):
        """Same triangle should not activate twice."""
        board            = HexBoard(radius=3)
        activated_combos = set()
        for a, b, c in _all_triangles(board):
            if all(t.tile_type == TILE_FERTILE for t in [a, b, c]):
                a.flower = b.flower = c.flower = 2
                break
        first  = detect_combos(board, activated_combos=activated_combos)
        assert len(first) >= 1
        for combo in first:
            activated_combos.add(combo["key"])
        second = detect_combos(board, activated_combos=activated_combos)
        assert second == []


# ============================================================
# 5. Corruption spread
# ============================================================

class TestCorruption:

    def test_corruption_spreads_inward(self):
        board = HexBoard(radius=3)
        rng   = np.random.default_rng(0)
        before = board.count_corrupted()
        spread_corruption(board, n_tiles=2, np_random=rng)
        assert board.count_corrupted() > before

    def test_corruption_cannot_overwrite_flower(self):
        board = HexBoard(radius=3)
        rng   = np.random.default_rng(7)
        for tile in board.tiles:
            if tile.is_placeable():
                if any(n.corrupted
                       for n in board.get_neighbours_of(tile.q, tile.r)):
                    tile.flower = 1
        before = board.count_corrupted()
        spread_corruption(board, n_tiles=10, np_random=rng)
        assert board.count_corrupted() == before

    def test_sacred_core_not_corrupted_by_spread(self):
        board = HexBoard(radius=3)
        rng   = np.random.default_rng(99)
        for _ in range(30):
            spread_corruption(board, n_tiles=4, np_random=rng)
        sacred = board.coord_to_tile[(0, 0)]
        assert not sacred.corrupted


# ============================================================
# 6. Episode smoke tests
# ============================================================

class TestEpisode:

    def test_random_agent_smoke(self):
        env = BloomwardEnv(render_mode=None)
        obs, _ = env.reset(seed=123)
        assert env.observation_space.contains(obs)
        for _ in range(50):
            obs, _, terminated, truncated, _ = env.step(
                env.action_space.sample())
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break
        env.close()

    def test_seeding_reproducibility(self):
        env1 = BloomwardEnv(render_mode=None)
        obs1, _ = env1.reset(seed=77)
        env2 = BloomwardEnv(render_mode=None)
        obs2, _ = env2.reset(seed=77)
        np.testing.assert_array_equal(obs1["board"], obs2["board"])
        assert obs1["current_flower"] == obs2["current_flower"]

    def test_truncation_at_max_turns(self):
        env = BloomwardEnv(render_mode=None, max_turns=10)
        env.reset(seed=0)
        done = False
        for _ in range(20):
            placeable = env._board.placeable_indices()
            action    = placeable[0] if placeable else env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                done = True
                break
        assert done


# ============================================================
# 7. Render
# ============================================================

class TestRender:

    def test_ansi_render_returns_string(self):
        env = BloomwardEnv(render_mode="ansi")
        env.reset(seed=5)
        output = env.render()
        assert isinstance(output, str)
        assert "BLOOMWARD" in output
