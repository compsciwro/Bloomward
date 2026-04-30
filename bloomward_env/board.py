# ============================================================
# bloomward_env/board.py
#
# Hexagonal board using axial (q, r) coordinates.
#
# Axial coordinate primer:
#   Center tile is (0, 0).
#   Six neighbours of (q, r) use the direction vectors below.
#   Hex distance = max(|dq|, |dr|, |dq+dr|).
#
# Board zones (radius = 3, 37 tiles total):
#   distance 0          → Sacred Core (1 tile)
#   distance 1–2        → Fertile Soil (18 tiles)
#   distance == radius  → Corrupt Soil (18 tiles)
# ============================================================

from .constants import (
    BOARD_RADIUS,
    TILE_FERTILE, TILE_CORRUPT, TILE_SACRED,
    ENC_EMPTY, ENC_CORRUPT, ENC_SACRED,
)

# Six neighbour directions for a flat-top hex grid
HEX_DIRECTIONS = [
    (1, 0), (1, -1), (0, -1),
    (-1, 0), (-1, 1), (0, 1),
]


def hex_distance(q1, r1, q2, r2):
    """Chebyshev distance between two axial hex coordinates."""
    return max(abs(q1 - q2), abs(r1 - r2), abs((q1 + r1) - (q2 + r2)))


def get_neighbours(q, r):
    """Return (q, r) tuples for all six potential neighbours."""
    return [(q + dq, r + dr) for dq, dr in HEX_DIRECTIONS]


class Tile:
    """
    One hex tile on the board.

    Attributes
    ----------
    q, r        : axial coordinates
    tile_type   : TILE_FERTILE | TILE_CORRUPT | TILE_SACRED
    flower      : None, or int flower code (1=sunflower, 2=tulip, 3=blossom)
    corrupted   : bool — True once corruption has taken this tile
    protected   : bool — True when Spirit of Blossom shields this tile
    """

    def __init__(self, q, r, tile_type):
        self.q         = q
        self.r         = r
        self.tile_type = tile_type
        self.flower    = None
        self.corrupted = (tile_type == TILE_CORRUPT)
        self.protected = False   # set by Spirit of Blossom

    def is_placeable(self):
        """True if an agent can place a flower here."""
        return (
            self.tile_type == TILE_FERTILE
            and not self.corrupted
            and self.flower is None
        )

    def encode(self):
        """
        Integer encoding used in the observation board array:
          0 = empty fertile
          1/2/3 = flower type
          4 = corrupted
          5 = sacred core
        """
        if self.tile_type == TILE_SACRED:
            return ENC_SACRED
        if self.corrupted:
            return ENC_CORRUPT
        if self.flower is not None:
            return self.flower   # 1, 2, or 3
        return ENC_EMPTY

    def __repr__(self):
        return (f"Tile({self.q},{self.r} "
                f"type={self.tile_type} "
                f"flower={self.flower} "
                f"corrupt={self.corrupted})")


class HexBoard:
    """
    Full hexagonal board.

    Tiles are stored in two structures:
      coord_to_tile : dict  {(q, r): Tile}   — for neighbour lookups
      tiles         : list  [Tile, ...]       — for stable index mapping

    The index of a tile in self.tiles is its action-space index.
    This mapping is fixed for the lifetime of the board object.
    """

    def __init__(self, radius=BOARD_RADIUS):
        self.radius        = radius
        self.coord_to_tile = {}
        self.tiles         = []
        self.index_of      = {}   # (q, r) → int index
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self):
        """Generate all tiles within the hex grid."""
        self.coord_to_tile.clear()
        self.tiles.clear()
        self.index_of.clear()

        for q in range(-self.radius, self.radius + 1):
            r_min = max(-self.radius, -q - self.radius)
            r_max = min(self.radius,  -q + self.radius)
            for r in range(r_min, r_max + 1):
                dist = hex_distance(0, 0, q, r)

                if dist == 0:
                    tile_type = TILE_SACRED
                elif dist == self.radius:
                    tile_type = TILE_CORRUPT
                else:
                    tile_type = TILE_FERTILE

                tile = Tile(q, r, tile_type)
                idx  = len(self.tiles)
                self.tiles.append(tile)
                self.coord_to_tile[(q, r)] = tile
                self.index_of[(q, r)]      = idx

    def reset(self):
        """Rebuild the board to its starting state."""
        self._build()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def num_tiles(self):
        return len(self.tiles)

    def tile_at_index(self, idx):
        if 0 <= idx < len(self.tiles):
            return self.tiles[idx]
        return None

    def get_tile(self, q, r):
        return self.coord_to_tile.get((q, r))

    def get_neighbours_of(self, q, r):
        """Return existing Tile objects adjacent to (q, r)."""
        return [
            self.coord_to_tile[(nq, nr)]
            for nq, nr in get_neighbours(q, r)
            if (nq, nr) in self.coord_to_tile
        ]

    # ------------------------------------------------------------------
    # Board-state queries
    # ------------------------------------------------------------------

    def count_corrupted(self):
        """Number of currently corrupted tiles."""
        return sum(1 for t in self.tiles if t.corrupted)

    def placeable_indices(self):
        """List of tile indices where a flower can legally be placed."""
        return [i for i, t in enumerate(self.tiles) if t.is_placeable()]

    def sacred_core_corrupted(self):
        """True if the Sacred Core tile itself has been corrupted."""
        sacred = self.coord_to_tile.get((0, 0))
        return sacred is not None and sacred.corrupted

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def to_array(self):
        """Return a list of encoded tile values (length = num_tiles)."""
        return [t.encode() for t in self.tiles]

    # ------------------------------------------------------------------
    # Text render
    # ------------------------------------------------------------------

    def render_text(self):
        """
        Simple ASCII board.
        Legend: O=sacred  S=sunflower  T=tulip  B=blossom
                X=corrupt  .=empty
        """
        symbol = {0: ".", 1: "S", 2: "T", 3: "B", 4: "X", 5: "O"}
        lines  = []
        for r in range(-self.radius, self.radius + 1):
            indent = " " * abs(r)
            row = [
                symbol[self.coord_to_tile[(q, r)].encode()]
                for q in range(-self.radius, self.radius + 1)
                if (q, r) in self.coord_to_tile
            ]
            lines.append(indent + " ".join(row))
        return "\n".join(lines)
