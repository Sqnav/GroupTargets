"""Causal world state conditions fusion after one visual-prior radar query.

History never changes the current visual mean/std or candidate addresses.
Its prior enters the fused evidence token before the single final depth head.
All memory is explicit and keyed by episode and public visual identities.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Mapping

import torch
from torch import nn

from .single_frame import RadarVisionLocalizer, current_features

@dataclass(frozen=True)
class MemberState:
    latent: torch.Tensor
    timestamp_s: float
    frame_index: int


@dataclass(frozen=True)
class WorldModelMemory:
    episode_id: str
    members: Mapping[Hashable, MemberState]

    def detached(self):
        """Explicit truncated-BPTT boundary chosen by the future trainer."""
        return WorldModelMemory(self.episode_id, {
            key: MemberState(value.latent.detach(), value.timestamp_s, value.frame_index)
            for key, value in self.members.items()
        })


class WorldFusionLocalizer(RadarVisionLocalizer):
    def __init__(self, config=None):
        super().__init__(config)
        self.group = "world_fusion"
        settings = self.config.get("newworld", {})
        if not isinstance(settings, dict):
            raise ValueError("newworld must be a configuration mapping")
        hidden = int(settings.get("hidden_size", 64))
        horizons = tuple(settings.get("rollout_horizons", (1, 2, 4)))
        if hidden < 8 or not horizons or any(type(x) is not int or x < 1 for x in horizons):
            raise ValueError("Positive rollout horizons and hidden_size >= 8 are required")
        if len(set(horizons)) != len(horizons):
            raise ValueError("Rollout horizons must be unique")
        self.hidden_size = hidden
        self.rollout_horizons = horizons
        self.max_memory_age_s = float(settings.get("max_memory_age_s", 10.))
        if not math.isfinite(self.max_memory_age_s) or self.max_memory_age_s <= 0:
            raise ValueError("max_memory_age_s must be positive and finite")
        self.transition = nn.GRUCell(1, hidden)
        self.observation_encoder = nn.Sequential(nn.Linear(1024+3+3+7+1, hidden), nn.SiLU(),
                                               nn.Linear(hidden, hidden))
        self.posterior = nn.GRUCell(hidden, hidden)
        self.position_decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        self.memory_bridge = nn.Linear(hidden, self.core.hidden_size, bias=False)
        # Births receive zero injection. The bridge conditions the fused token,
        # leaving the visual mean/std and all candidate addresses unchanged.
        nn.init.zeros_(self.memory_bridge.weight)

    def initialize_from_single_frame(self, single_frame):
        """Copy the current one-query single-frame architecture; reset the bridge."""
        if (not isinstance(single_frame, RadarVisionLocalizer) or single_frame.visual
                or single_frame.group != "single_frame"):
            raise TypeError("A group 3 RadarVisionLocalizer is required")
        self.core.load_state_dict(single_frame.core.state_dict(), strict=True)
        nn.init.zeros_(self.memory_bridge.weight)

    def _identity(self, features):
        episode = features.get("episode_id")
        frame = features.get("frame_index")
        timestamp = features.get("timestamp_s")
        ids = tuple(features.get("member_ids", ()))
        if not isinstance(episode, str) or not episode:
            raise ValueError("Explicit episode_id is required for world memory")
        if type(frame) is not int or frame < 0:
            raise ValueError("A nonnegative integer frame_index is required")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError("A finite public timestamp_s is required")
        if len(ids) != len(features["ray_directions"]):
            raise ValueError("member_ids must follow current detection order")
        assigned = [key for key in ids if key is not None]
        if any(isinstance(key, bool) or not isinstance(key, (str, int)) for key in assigned) or len(set(assigned)) != len(assigned):
            raise ValueError("Unique string/integer public visual member IDs or unassociated None are required")
        return episode, frame, float(timestamp), ids

    def _step(self, latent, dt):
        return self.transition(torch.log1p(dt).reshape(-1, 1), latent)

    def _rollout(self, posterior, dt, scale):
        """Open-loop future states: no future observation is an argument."""
        values = {}
        state = posterior
        for step in range(1, max(self.rollout_horizons)+1):
            state = self._step(state, dt)
            if step in self.rollout_horizons:
                values[step] = self.position_decoder(state)*scale
        return torch.stack([values[step] for step in self.rollout_horizons], dim=1)

    def forward(self, features, memory=None):
        episode, frame, timestamp, ids = self._identity(features)
        if memory is not None and not isinstance(memory, WorldModelMemory):
            raise TypeError("memory must be WorldModelMemory or None")
        parameter = self.memory_bridge.weight
        device, dtype = parameter.device, parameter.dtype
        n = len(ids)
        previous = memory.members if memory is not None and memory.episode_id == episode else {}
        priors, seen, durations = [], [], []
        for member in ids:
            past = previous.get(member) if member is not None else None
            if past is not None and (frame <= past.frame_index or timestamp <= past.timestamp_s):
                raise ValueError("World memory must be strictly causal within each episode/member")
            available = past is not None and timestamp-past.timestamp_s <= self.max_memory_age_s
            dt = timestamp-past.timestamp_s if available else 1.
            durations.append(dt)
            if available:
                latent = past.latent.to(device=device, dtype=dtype)
                if latent.shape != (self.hidden_size,) or not bool(torch.isfinite(latent).all()):
                    raise ValueError("Invalid world memory latent")
                priors.append(self._step(latent[None], parameter.new_tensor([dt]))[0])
            else:
                priors.append(parameter.new_zeros(self.hidden_size))
            seen.append(available)
        prior = torch.stack(priors) if priors else parameter.new_empty((0, self.hidden_size))
        available = torch.tensor(seen, dtype=torch.bool, device=device)
        duration = parameter.new_tensor(durations)
        injection = self.memory_bridge(prior)*available[:, None]
        output = self.core(current_features(features), fusion_prior=injection)
        scale = float(features["range_scale_m"])
        tensor = lambda key: torch.as_tensor(features[key], device=device, dtype=dtype)
        observation = torch.cat((tensor("detector_roi_features").detach(), tensor("ray_origins")/scale,
                                 tensor("ray_directions"), tensor("visual_scale"),
                                 output["depths"][:, None]/scale), dim=-1)
        posterior = self.posterior(self.observation_encoder(observation), prior)
        valid = output["valid_mask"]
        # Keep absent members only for a short public-time timeout; invalid
        # geometry does not overwrite a usable posterior with dummy evidence.
        updated = {key: value for key, value in previous.items()
                   if 0. <= timestamp-value.timestamp_s <= self.max_memory_age_s}
        for index, member in enumerate(ids):
            if member is not None and bool(valid[index]):
                updated[member] = MemberState(posterior[index], timestamp, frame)
        output["state"] = WorldModelMemory(episode, updated)
        output["aux"].update({
            "group": self.group, "architecture": self.architecture, "radar_used": True,
            "depth_head_count": 1, "prior_latent": prior,
            "posterior_latent": posterior, "prior_available": available & valid,
            "prior_positions": self.position_decoder(prior)*scale,
            "rollout_positions": self._rollout(posterior, duration, scale),
            "rollout_horizons": self.rollout_horizons, "memory_injection": injection,
            "member_ids": ids,
        })
        return output
