# ============================================================
# bloomward_env/constants.py
#
# All tunable constants for Bloomward in one place.
# Change reward values, spread rates, or win conditions here
# without touching game logic files.
# ============================================================

# --------------- Tile types --------------------------------
TILE_FERTILE = "fertile"
TILE_CORRUPT = "corrupt"
TILE_SACRED  = "sacred"

# --------------- Tile encoding (observation array) ---------
# These integers appear in the "board" observation array.
ENC_EMPTY     = 0   # empty fertile tile
ENC_SUNFLOWER = 1
ENC_TULIP     = 2
ENC_BLOSSOM   = 3
ENC_CORRUPT   = 4
ENC_SACRED    = 5

# --------------- Flower types ------------------------------
FLOWER_SUNFLOWER = 1
FLOWER_TULIP     = 2
FLOWER_BLOSSOM   = 3
FLOWER_NAMES     = {1: "Sunflower", 2: "Tulip", 3: "Blossom"}

# Weighted draw probabilities — must sum to 1.0
FLOWER_TYPES   = [FLOWER_SUNFLOWER, FLOWER_TULIP, FLOWER_BLOSSOM]
FLOWER_WEIGHTS = [0.50,             0.30,          0.20]

# --------------- Spirits -----------------------------------
SPIRIT_NAMES = {
    FLOWER_SUNFLOWER: "Spirit of Renewal",
    FLOWER_TULIP:     "Spirit of Rain",
    FLOWER_BLOSSOM:   "Spirit of Blossom",
}

# --------------- Seasons -----------------------------------
SEASON_SPRING = 0
SEASON_RAIN   = 1
SEASON_AUTUMN = 2
SEASON_WINTER = 3
SEASON_NAMES  = {0: "Spring", 1: "Rain", 2: "Autumn", 3: "Winter"}

# How many corrupt tiles spread per activation, by season
SEASON_SPREAD_RATE = {
    SEASON_SPRING: 1,
    SEASON_RAIN:   1,
    SEASON_AUTUMN: 2,
    SEASON_WINTER: 2,
}

# Season advances every N turns
TURNS_PER_SEASON = 4

# Corruption spreads every N turns
SPREAD_EVERY_N_TURNS = 8

# --------------- Board -------------------------------------
BOARD_RADIUS = 4  # radius 4 → 61   tiles total (3r²+3r+1)

# --------------- Win / loss --------------------------------
WIN_SCORE_TARGET = 120
MAX_TURNS        = 100 # episode truncation limit

# --------------- Rewards (edit here to tune) ---------------
REWARD_VALID_PLACEMENT = 2
REWARD_INVALID_ACTION  = -2
REWARD_COMBO           = 50
REWARD_NEAR_COMBO = 3
REWARD_WIN             = 200
REWARD_LOSS            = -50
REWARD_TURN_PENALTY    = -0.1
