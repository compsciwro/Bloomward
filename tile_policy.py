# -*- coding: utf-8 -*-
"""
Bloomward - tile_policy.py
The v2 agent's network, used with MaskablePPO.

The standard network looks at the whole board at once and has to work out,
for each of the 61 tiles separately, what makes that tile good. This one
gives every tile the SAME small network:

  1. each tile's features (plus the whole-board features) -> a tile summary
  2. each tile also sees the average summary of its six neighbours
  3. one shared layer turns each tile's summary into that tile's score

The tile scores are the action preferences; MaskablePPO masks the illegal
ones and samples as usual. What it learns about a "good tile" in one place
applies everywhere on the board. Which tiles are good is still learned from
reward; nothing here is a rule about how to play.
"""

import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy


class TileExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, adjacency, emb=48):
        n = adjacency.shape[0]
        super().__init__(observation_space, features_dim=n * emb)
        self.n, self.emb = n, emb
        self.tile_in = observation_space["tiles"].shape[0] // n
        self.glob_in = observation_space["globals"].shape[0]
        a = th.tensor(adjacency, dtype=th.float32)
        self.register_buffer("adj", a / a.sum(1, keepdim=True).clamp(min=1))
        self.embed = nn.Sequential(nn.Linear(self.tile_in + self.glob_in, 96), nn.ReLU(),
                                   nn.Linear(96, emb), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(2 * emb, emb), nn.ReLU())

    def forward(self, obs):
        b = obs["tiles"].shape[0]
        tiles = obs["tiles"].view(b, self.n, self.tile_in)
        glob = obs["globals"].unsqueeze(1).expand(b, self.n, self.glob_in)
        h = self.embed(th.cat([tiles, glob], -1))               # step 1: (batch, 61, emb)
        neighbours = th.einsum("ij,bje->bie", self.adj, h)       # step 2: neighbour average
        h = self.mix(th.cat([h, neighbours], -1))
        return h.reshape(b, -1)


class PerTileScore(nn.Module):
    """Step 3: the same layers score every tile."""
    def __init__(self, n, emb):
        super().__init__()
        self.n, self.emb = n, emb
        self.score = nn.Sequential(nn.Linear(emb, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, latent):
        return self.score(latent.view(latent.shape[0], self.n, self.emb)).squeeze(-1)


class PooledValue(nn.Module):
    """Value estimate from a summary of all tiles (their average and their maximum)."""
    def __init__(self, n, emb):
        super().__init__()
        self.n, self.emb = n, emb
        self.v = nn.Sequential(nn.Linear(2 * emb, 64), nn.ReLU())

    def forward(self, latent):
        h = latent.view(latent.shape[0], self.n, self.emb)
        return self.v(th.cat([h.mean(1), h.max(1).values], -1))


class TilePolicy(MaskableMultiInputActorCriticPolicy):
    def __init__(self, *args, adjacency=None, emb=48, **kwargs):
        self._adjacency, self._emb = adjacency, emb
        kwargs["features_extractor_class"] = TileExtractor
        kwargs["features_extractor_kwargs"] = dict(adjacency=adjacency, emb=emb)
        kwargs["net_arch"] = dict(pi=[], vf=[])
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        n = self._adjacency.shape[0]
        self.action_net = PerTileScore(n, self._emb)
        self.mlp_extractor.value_net = PooledValue(n, self._emb)
        self.mlp_extractor.latent_dim_vf = 64
        self.value_net = nn.Linear(64, 1)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)
