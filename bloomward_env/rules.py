# ============================================================
# bloomward_env/rules.py
#
# Game rule functions:
#   - Triangle combo detection (with duplicate prevention)
#   - Spirit activation and effects
#   - Corruption spread
#   - Season helpers
#   - Win / loss / truncation checks
# ============================================================

from .board import HexBoard
from .constants import (
    TILE_FERTILE,
    FLOWER_SUNFLOWER, FLOWER_TULIP, FLOWER_BLOSSOM,
    SPIRIT_NAMES,
    SEASON_SPREAD_RATE, TURNS_PER_SEASON,
)


# ============================================================
# 1. Triangle combo detection
# ============================================================

def _triangles_containing(board: HexBoard, q, r):
    """
    Generate all unit-triangles that include tile (q, r).

    Only inspects the 6 neighbours of the pivot tile, so this is
    O(36) instead of O(n_tiles) and avoids a full board scan.
    """
    pivot = board.get_tile(q, r)
    if pivot is None:
        return
    nbrs = board.get_neighbours_of(q, r)
    nbr_set = {(n.q, n.r) for n in nbrs}

    for i in range(len(nbrs)):
        for j in range(i + 1, len(nbrs)):
            a, b = nbrs[i], nbrs[j]
            if (a.q, a.r) in {(n.q, n.r)
                               for n in board.get_neighbours_of(b.q, b.r)}:
                yield (pivot, a, b)


def detect_combos(board: HexBoard,
                  last_placed_coord=None,
                  activated_combos: set = None):
    """
    Scan the board for new triangle combos: three tiles of the
    same flower type that are mutually adjacent.

    Parameters
    ----------
    board              : HexBoard
    last_placed_coord  : (q, r) of the most recently placed flower.
                         Only triangles containing that tile are checked.
    activated_combos   : set of frozenset keys already activated.
                         Used to prevent double-counting the same
                         triangle across multiple turns.

    Returns
    -------
    List of combo dicts:
        { "flower": int, "tiles": [Tile, Tile, Tile], "key": frozenset }
    """
    if activated_combos is None:
        activated_combos = set()

    if last_placed_coord is None:
        return []

    combos  = []
    checked = set()

    for tile_a, tile_b, tile_c in _triangles_containing(board, *last_placed_coord):
        key = frozenset([(tile_a.q, tile_a.r),
                         (tile_b.q, tile_b.r),
                         (tile_c.q, tile_c.r)])

        if key in checked or key in activated_combos:
            continue
        checked.add(key)

        # All three must carry the same non-None flower
        f = tile_a.flower
        if f is None:
            continue
        if tile_b.flower == f and tile_c.flower == f:
            combos.append({
                "flower": f,
                "tiles":  [tile_a, tile_b, tile_c],
                "key":    key,
            })

    return combos


# ============================================================
# 2. Spirit effects
# ============================================================

def activate_spirit(board: HexBoard, combo: dict, np_random):
    """
    Resolve a spirit activation.

    Spirit of Renewal  (Sunflower) → cleanse one adjacent corrupted tile
    Spirit of Rain     (Tulip)     → signal to skip corruption this turn
                                     (handled in env.step via return value)
    Spirit of Blossom  (Blossom)   → protect adjacent fertile tiles

    Returns a result dict with details for the info dictionary.
    """
    flower      = combo["flower"]
    spirit_name = SPIRIT_NAMES.get(flower, "Unknown Spirit")
    result      = {"spirit": spirit_name, "flower": flower}

    combo_tiles = combo["tiles"]

    if flower == FLOWER_SUNFLOWER:
        # ── Spirit of Renewal: cleanse one corrupted neighbour ──────
        candidates = []
        for tile in combo_tiles:
            for nbr in board.get_neighbours_of(tile.q, tile.r):
                if nbr.corrupted and nbr not in candidates:
                    candidates.append(nbr)

        cleansed = None
        if candidates:
            # Use the seeded numpy RNG for reproducibility
            idx    = int(np_random.integers(0, len(candidates)))
            target = candidates[idx]
            target.corrupted  = False
            target.tile_type  = TILE_FERTILE   # restore as fertile
            cleansed = (target.q, target.r)

        result["cleansed_tile"] = cleansed

    elif flower == FLOWER_TULIP:
        # ── Spirit of Rain: signal to skip corruption this turn ─────
        # The actual skip logic is handled in env.step().
        # This flag is checked there.
        result["skip_corruption"] = True

    elif flower == FLOWER_BLOSSOM:
        # ── Spirit of Blossom: protect adjacent fertile tiles ───────
        protected = []
        for tile in combo_tiles:
            for nbr in board.get_neighbours_of(tile.q, tile.r):
                if (nbr.tile_type == TILE_FERTILE
                        and not nbr.corrupted
                        and not nbr.protected):
                    nbr.protected = True
                    protected.append((nbr.q, nbr.r))
        result["protected_tiles"] = protected

    return result


# ============================================================
# 3. Corruption spread
# ============================================================

def spread_corruption(board: HexBoard, n_tiles: int, np_random):
    """
    Spread corruption inward by up to n_tiles tiles.

    Rules:
      - Only empty, unprotected fertile tiles adjacent to a
        corrupted tile are eligible.
      - Cannot overwrite flowers.
      - Cannot overwrite the Sacred Core.
      - Protected tiles (Spirit of Blossom) are skipped and
        their protection is consumed on contact.

    Uses the seeded np_random for reproducibility.

    Returns a list of (q, r) coords that became newly corrupted.
    """
    eligible = []
    for tile in board.tiles:
        if tile.tile_type != TILE_FERTILE:
            continue
        if tile.corrupted or tile.flower is not None:
            continue
        # Must be adjacent to a corrupted tile
        has_corrupt_nbr = any(
            n.corrupted
            for n in board.get_neighbours_of(tile.q, tile.r)
        )
        if has_corrupt_nbr:
            eligible.append(tile)

    # Shuffle using the seeded RNG
    if eligible:
        indices = np_random.permutation(len(eligible))
        eligible = [eligible[i] for i in indices]

    newly_corrupted = []
    spread_count    = 0

    for tile in eligible:
        if spread_count >= n_tiles:
            break

        if tile.protected:
            # Protection consumed — tile is safe this time
            tile.protected = False
            continue

        tile.corrupted = True
        newly_corrupted.append((tile.q, tile.r))
        spread_count += 1

    return newly_corrupted


# ============================================================
# 4. Season helpers
# ============================================================

def get_season(turn: int) -> int:
    """Return the season index (0–3) for the given turn."""
    return (turn // TURNS_PER_SEASON) % 4


def spread_rate_for_season(season: int) -> int:
    """Number of tiles corruption spreads in this season."""
    return SEASON_SPREAD_RATE[season]


# ============================================================
# 5. Win / loss / truncation
# ============================================================

def check_terminal(board: HexBoard, spirit_count: int,
                   turn: int, max_turns: int):
    """
    Evaluate the episode terminal state.

    Returns
    -------
    (terminated: bool, truncated: bool, reason: str)

    reason values:
      "win"                      — agent won
      "loss_sacred_core"         — corruption reached the Sacred Core
      "loss_no_valid_moves"      — no placeable tiles left
      "truncated_max_turns"      — turn limit reached
      "ongoing"                  — episode continues
    """
    from .constants import WIN_SCORE_TARGET

    # Loss: corruption reached the Sacred Core
    if board.sacred_core_corrupted():
        return True, False, "loss_sacred_core"

    # Loss: no valid moves remain
    if len(board.placeable_indices()) == 0:
        return True, False, "loss_no_valid_moves"

    # Win: enough spirits AND corruption under control
    score = spirit_count * 10 - board.count_corrupted()
    if score >= WIN_SCORE_TARGET:
        return True, False, "win"

    # Truncation: max turns reached
    if turn >= max_turns:
        return False, True, "truncated_max_turns"

    return False, False, "ongoing"

# ============================================================
# Spirit targeting and application for player-chosen targets (Renewal and Blossom)
# ============================================================
 
def get_spirit_targets(board: HexBoard, combo: dict) -> list:
    """
    Return valid target tile INDICES for a spirit that requires player input.
 
    Spirit of Renewal (Sunflower) -> corrupted tiles adjacent to the combo tiles
    Spirit of Blossom (Blossom)   -> empty fertile tiles adjacent to the combo tiles
    Spirit of Rain    (Tulip)     -> empty list (auto-resolves, no targeting needed)
    """
    flower     = combo["flower"]
    candidates = set()
 
    if flower == FLOWER_SUNFLOWER:
        for tile in combo["tiles"]:
            for nbr in board.get_neighbours_of(tile.q, tile.r):
                if nbr.corrupted:
                    idx = board.index_of.get((nbr.q, nbr.r))
                    if idx is not None:
                        candidates.add(idx)
 
    elif flower == FLOWER_BLOSSOM:
        for tile in combo["tiles"]:
            for nbr in board.get_neighbours_of(tile.q, tile.r):
                if (nbr.tile_type == TILE_FERTILE
                        and not nbr.corrupted
                        and not nbr.protected
                        and nbr.flower is None):
                    idx = board.index_of.get((nbr.q, nbr.r))
                    if idx is not None:
                        candidates.add(idx)
 
    return list(candidates)
 
 
def apply_spirit_at_target(board: HexBoard, combo: dict, target_idx: int) -> dict:
    """
    Apply a Renewal or Blossom spirit to a player-chosen target tile.
    Returns a result dict in the same format as activate_spirit().
    """
    flower      = combo["flower"]
    spirit_name = SPIRIT_NAMES.get(flower, "Unknown Spirit")
    result      = {"spirit": spirit_name, "flower": flower}
 
    if flower == FLOWER_SUNFLOWER:
        target   = board.tile_at_index(target_idx)
        cleansed = None
        if target and target.corrupted:
            target.corrupted = False
            target.tile_type = TILE_FERTILE
            cleansed = (target.q, target.r)
        result["cleansed_tile"] = cleansed
 
    elif flower == FLOWER_BLOSSOM:
        target    = board.tile_at_index(target_idx)
        protected = []
        if target and target.tile_type == TILE_FERTILE and not target.corrupted:
            if not target.protected:
                target.protected = True
                protected.append((target.q, target.r))
            for nbr in board.get_neighbours_of(target.q, target.r):
                if (nbr.tile_type == TILE_FERTILE
                        and not nbr.corrupted
                        and not nbr.protected):
                    nbr.protected = True
                    protected.append((nbr.q, nbr.r))
        result["protected_tiles"] = protected
 
    return result
