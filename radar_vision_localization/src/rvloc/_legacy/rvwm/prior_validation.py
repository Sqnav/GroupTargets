"""Deterministic per-frame seed shared by training validation and deployment."""
import hashlib


def frame_seed(base_seed, episode_id, frame_idx):
    digest = hashlib.sha256(f'{int(base_seed)}:{episode_id}:{int(frame_idx)}:deployment_prior_v1'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') % (2**31)
