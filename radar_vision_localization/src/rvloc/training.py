"""Explicit TRAIN-only neural training with causal, truncated recurrence.

This module performs no work on import. Labels enter only ``_targets`` and the
loss, never visual association, model features or world-state initialization.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from itertools import chain
import json
import math
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
import torch

from .config import ExperimentConfig, external_path, resolve_method
from .data import load_manifest, load_labels
from .models import build_model, localization_loss, save_checkpoint
from .models.single_frame import FORBIDDEN_FIELDS
from .supervision import match_direct_ray_labels


def _settings(config):
    value = dict(config.raw.get("training", {}))
    value.setdefault("epochs", 80)
    value.setdefault("learning_rate", .0003)
    value.setdefault("weight_decay", .0001)
    value.setdefault("seed", 20261118)
    value.setdefault("bptt_steps", value.get("truncation_frames", 4))
    value.setdefault("grad_clip", value.get("gradient_clip", 5.))
    value.setdefault("warmup_epochs", 0)
    value.setdefault("device", "cpu")
    value.setdefault("prepared_only", False)
    value.setdefault("cache_observations", False)
    value.setdefault("log_every_updates", 50)
    for key in ("prepared_only", "cache_observations"):
        if type(value[key]) is not bool:
            raise ValueError(f"training.{key} must be boolean")
    if type(value["log_every_updates"]) is not int or value["log_every_updates"] < 1:
        raise ValueError("training.log_every_updates must be a positive integer")
    for key in ("epochs", "bptt_steps"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("seed", "warmup_epochs"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"training.{key} must be a nonnegative integer")
    for key in ("learning_rate", "weight_decay", "grad_clip"):
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f"training.{key} must be finite")
        if number < 0 or key != "weight_decay" and number == 0:
            raise ValueError(f"training.{key} must be positive (weight_decay may be zero)")
    return value


def _association_factory(config, manifest):
    from .runtime import load_calibration
    from .tracking import VisualAssociator
    calibration = load_calibration(config, manifest, required=True)
    if not isinstance(calibration.get("visual_params"), dict):
        raise ValueError("TRAIN-frozen visual association parameters are required")
    factory = lambda: VisualAssociator(params=calibration["visual_params"])
    factory.calibration_sha256 = calibration.get("_sha256")
    return factory


def _associate(associator, observation):
    features = observation.features
    value = associator.step(observation.frame_idx, observation.physical_time_s,
                            features["bbox_xyxy"], features["detector_roi_features"],
                            features["detection_scores"])
    # The public adapter returns IDs. Supporting the legacy diagnostic wrapper
    # here also makes it possible to inspect one unchanged association backend.
    if isinstance(value, dict):
        if "member_ids" in value:
            value = value["member_ids"]
        elif "outputs" in value:
            value = [item["track_id"] for item in value["outputs"]]
        else:
            raise ValueError("Visual association must return detection-aligned member IDs")
    return list(value)


def _targets(features, positions, model_config, prediction):
    """Offline per-frame matching; annotation member IDs are never accessed."""
    matched = match_direct_ray_labels(features, positions, model_config)
    n = len(features["ray_directions"])
    device, dtype = prediction["depths"].device, prediction["depths"].dtype
    depths = torch.zeros(n, device=device, dtype=dtype)
    points = torch.zeros((n, 3), device=device, dtype=dtype)
    valid = torch.zeros(n, device=device, dtype=torch.bool)
    rows = torch.as_tensor(matched["rows"], device=device)
    if len(rows):
        depths[rows] = torch.as_tensor(matched["depths_m"], device=device, dtype=dtype)
        points[rows] = torch.as_tensor(np.asarray(positions)[matched["cols"]], device=device, dtype=dtype)
        valid[rows] = True
    return {"depths": depths, "member_positions": points, "valid_mask": valid}, matched


def _code_hashes():
    # Include the vendored geometry/RA/RE implementation as well as adapters.
    files = sorted(Path(__file__).parent.rglob("*.py"))
    return {str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def train(config: ExperimentConfig, group, *, manifest=None, model=None, output_dir=None,
          association_factory=None):
    """Train arm 2/3/7 with separate result and weight run directories.

    ``manifest``/``model``/``association_factory`` can be injected for synthetic
    CPU verification. Production group 7 requires a TRAIN-frozen association
    artifact through ``runtime.load_calibration``. Current position supervision
    trains the transitioned prior; multi-horizon forward rollouts remain
    available through the model/loss API and are not claimed as future-label
    supervision by this trainer.
    """
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be ExperimentConfig")
    method_id, method = resolve_method(group)
    if method_id not in (2, 3, 7):
        raise ValueError("Only neural arms 2, 3 and 7 are independently trained")
    settings = _settings(config)
    if manifest is None:
        if config.manifest is None:
            raise ValueError("Training requires a ready, explicit data manifest")
        manifest = load_manifest(config.manifest)
    if settings["cache_observations"] and method["input"] == "rgb_radar" and not settings["prepared_only"]:
        raise ValueError("Caching fusion observations requires authenticated prepared-only replay, not raw IQ")
    observation_options = {"modality": method["input"]}
    if settings["prepared_only"] and method["input"] == "rgb_radar":
        observation_options["prepared_only"] = True
    first_iterator = iter(manifest.observations("train", **observation_options))
    try:
        first_observation = next(first_iterator)
    except StopIteration as error:
        raise ValueError("TRAIN contains no observations; refusing an empty training run") from error
    if method_id == 7 and association_factory is None:
        association_factory = _association_factory(config, manifest)
    torch.manual_seed(settings["seed"])
    model_options = dict(config.raw.get("model", {}))
    if "world" in config.raw:
        model_options["newworld"] = dict(config.raw["world"])
    if model is None:
        model = build_model(method["model"], model_options)
    elif getattr(model, "group", None) != method["model"]:
        raise ValueError("Injected model does not match requested experiment arm")
    model.to(settings["device"]).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    if output_dir is None:
        label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid4().hex[:8]
        destination = config.output_root/f"train-{model.group}-{label}"
    else:
        destination = external_path(output_dir, field="training output_dir", base=config.path.parent)
        if destination == config.output_root or not destination.is_relative_to(config.output_root):
            raise ValueError("training output_dir must be a run directory inside configured output_root")
    relative_run = destination.relative_to(config.output_root)
    weights_destination = config.weights_root/relative_run
    # Check both namespaces before creating either: a custom run name cannot
    # overwrite an existing weight run merely because its result folder is absent.
    if destination.exists() or weights_destination.exists():
        raise FileExistsError("Training result or weight run already exists; choose a new run directory")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        weights_destination.mkdir(parents=True, exist_ok=False)
    except BaseException:
        destination.rmdir()  # Empty directory created by this call only.
        raise
    run_id=relative_run.as_posix()
    identity = {"config_sha256": config.sha256, "manifest_sha256": manifest.sha256,
                "architecture": "single_query_depth_v3", "depth_refinement_steps": 0,
                "local_query_enabled": method_id != 2, "query_rounds":0 if method_id == 2 else 1,
                "supervision_domain": "public_camera_ray_support",
                "code_sha256": _code_hashes(), "experiment_config": config.raw,
                "training_split": "train", "method_id": method_id,
                "center_contract": config.raw["center_contract"],
                "observer_position_reference": "camera_optical_center",
                "body_navigation_is_model_input": False,
                "calibration_sha256": getattr(association_factory, "calibration_sha256", None),
                "association": "TRAIN-frozen visual only" if method_id == 7 else "none",
                "run_id":run_id,"output_dir":str(destination),"weights_dir":str(weights_destination)}
    history = []
    updates = 0
    started = time.monotonic()
    first_sources = None
    observation_cache = []
    estimated_frames = None
    if hasattr(manifest, "splits") and isinstance(manifest.episodes, dict):
        estimated_frames = sum(len(manifest.episodes[x]) for x in manifest.splits["train"])

    def log(message, *, status="training", **progress):
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[{timestamp}] TRAIN G{method_id} {message}", flush=True)
        payload = dict(schema="rvloc.training_progress.v1", status=status, method_id=method_id,
                       updated_at_utc=timestamp, device=settings["device"],
                       epochs=settings["epochs"], optimizer_updates=updates,
                       elapsed_s=time.monotonic()-started, **progress)
        temporary = destination/"progress.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
        temporary.replace(destination/"progress.json")

    log(f"START device={settings['device']} epochs={settings['epochs']} "
        f"prepared_only={settings['prepared_only']} cpu_sensor_cache={settings['cache_observations']} "
        f"frames_per_epoch={estimated_frames}", epoch=0)
    for epoch in range(settings["epochs"]):
        epoch_started = time.monotonic()
        log(f"EPOCH {epoch+1}/{settings['epochs']} START", epoch=epoch+1, frames=0)
        warmup = min(1., (epoch+1)/max(1, settings["warmup_epochs"]))
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = settings["learning_rate"]*warmup
        observations = chain((first_observation,), first_iterator) if epoch == 0 else \
            observation_cache if settings["cache_observations"] else \
            manifest.observations("train", **observation_options)
        episode = None
        finished_episodes = set()
        previous_frame = previous_time = None
        memory = associator = None
        pending = []
        pending_frames = 0
        frame_count = matched_frames = matched_members = unassigned_members = 0
        totals = {}
        sources = hashlib.sha256()

        def flush():
            nonlocal updates, pending_frames, memory
            if pending:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.stack(pending).mean()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite TRAIN loss; checkpoint was not committed")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"],
                                                     error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                totals["gradient_norm_sum"] = totals.get("gradient_norm_sum", 0.)+float(norm)
                if updates % settings["log_every_updates"] == 0:
                    elapsed = time.monotonic()-started
                    completed = epoch*(estimated_frames or 0)+frame_count
                    eta = (settings["epochs"]*estimated_frames-completed)*elapsed/max(1,completed) if estimated_frames else None
                    log(f"epoch={epoch+1}/{settings['epochs']} frame={frame_count}/{estimated_frames} "
                        f"update={updates} loss={float(loss.detach()):.6f} "
                        f"lr={optimizer.param_groups[0]['lr']:.6g} elapsed_s={elapsed:.1f} "
                        f"eta_s={eta:.1f}" if eta is not None else
                        f"epoch={epoch+1}/{settings['epochs']} frame={frame_count} update={updates} "
                        f"loss={float(loss.detach()):.6f} elapsed_s={elapsed:.1f}",
                        epoch=epoch+1, frames=frame_count, last_loss=float(loss.detach()), eta_s=eta)
            pending.clear()
            pending_frames = 0
            if memory is not None:
                memory = memory.detached()

        for observation in observations:
            if epoch == 0 and settings["cache_observations"]:
                if getattr(observation, "iq", None) is not None:
                    raise ValueError("CPU sensor replay must not retain raw IQ")
                observation_cache.append(observation)
            if observation.episode_id != episode:
                flush()
                if observation.episode_id in finished_episodes:
                    raise ValueError("TRAIN must keep all frames of each episode contiguous")
                if episode is not None:
                    finished_episodes.add(episode)
                episode = observation.episode_id
                memory = None
                associator = association_factory() if method_id == 7 else None
                previous_frame = previous_time = None
            if previous_frame is not None and (observation.frame_idx <= previous_frame or
                                               observation.physical_time_s <= previous_time):
                raise ValueError("TRAIN frames and public time must increase strictly within an episode")
            previous_frame, previous_time = observation.frame_idx, observation.physical_time_s
            features = dict(observation.features)
            forbidden = FORBIDDEN_FIELDS.intersection(features)
            if forbidden:
                raise ValueError("Observation contains labels or forbidden inference payloads")
            if method_id == 7:
                member_ids = _associate(associator, observation)
                unassigned_members += sum(member is None for member in member_ids)
                features.update(episode_id=episode, member_ids=member_ids,
                                frame_index=observation.frame_idx, timestamp_s=observation.physical_time_s)
            prediction = model(features, memory)
            memory = prediction["state"]
            # The first label access happens only after the current prediction.
            labels = load_labels(manifest, observation)
            targets, matched = _targets(features, labels.positions, model.config, prediction)
            count = len(matched["rows"])
            if count:
                loss, terms = localization_loss(prediction, targets,
                    huber_delta=float(model.config.get("depth_huber_delta_m", 1.)),
                    visual_prior_weight=float(model.config.get('visual_prior_nll_weight',.1)),
                    mse_weight=float(model.config.get("depth_loss_mse_weight", .05)))
                pending.append(loss)
                matched_frames += 1
                matched_members += count
                for key, value in terms.items():
                    totals[key] = totals.get(key, 0.)+float(value.detach())
            pending_frames += 1
            frame_count += 1
            sources.update(json.dumps({"episode": episode, "frame": observation.frame_idx,
                                       "source_hashes": getattr(observation, "source_hashes", None)},
                                      sort_keys=True, allow_nan=False).encode())
            if pending_frames >= settings["bptt_steps"]:
                flush()
        flush()
        if frame_count == 0 or matched_frames == 0:
            raise ValueError("TRAIN has no matched supervision; refusing a successful empty checkpoint")
        source_digest = sources.hexdigest()
        if first_sources is not None and source_digest != first_sources:
            raise ValueError("TRAIN observation identity changed between epochs")
        first_sources = source_digest
        row = {"epoch": epoch+1, "frames": frame_count, "matched_frames": matched_frames,
               "matched_members": matched_members, "unassigned_visual_members": unassigned_members,
               "optimizer_updates": updates, "learning_rate": optimizer.param_groups[0]["lr"],
               "loss": {key: value/matched_frames for key, value in totals.items()
                        if key != "gradient_norm_sum"}}
        history.append(row)
        row["epoch_elapsed_s"] = time.monotonic()-epoch_started
        model.core.trained = True
        metadata = dict(identity, completed_epochs=epoch+1, optimizer_updates=updates,
                        observation_identity_sha256=source_digest, training_settings=settings,
                        history=history, inference_uses_gt_ids=False,
                        dynamics_supervision="current prior position" if method_id == 7 else None)
        save_checkpoint(model, weights_destination/f"epoch_{epoch+1:04d}.pt", optimizer=optimizer, metadata=metadata)
        (destination/"history.json").write_text(json.dumps(history,indent=2,allow_nan=False)+"\n")
        log(f"EPOCH {epoch+1}/{settings['epochs']} COMPLETE frames={frame_count} "
            f"updates={updates} mean_loss={row['loss'].get('total')} "
            f"epoch_s={row['epoch_elapsed_s']:.1f} checkpoint={weights_destination/f'epoch_{epoch+1:04d}.pt'}",
            epoch=epoch+1, frames=frame_count, mean_loss=row["loss"].get("total"))
    save_checkpoint(model, weights_destination/"model.pt", optimizer=optimizer, metadata=metadata)
    receipt = dict(schema="rvloc.training.v3", status="trained", method_id=method_id, group=model.group,
                   architecture="single_query_depth_v3", depth_refinement_steps=0,
                   local_query_enabled=method_id != 2, query_rounds=0 if method_id == 2 else 1,
                   run_id=run_id,output_dir=str(destination),weights_dir=str(weights_destination),
                   checkpoint=str(weights_destination/"model.pt"),
                   epochs=settings["epochs"], optimizer_updates=updates, history=history,
                   training_identity=identity, observation_identity_sha256=first_sources,
                   elapsed_s=time.monotonic()-started, bptt_steps=settings["bptt_steps"],
                   labels_only_in_loss=True, inference_uses_gt_ids=False,
                   dynamics_supervision="one-step prior position and detached posterior latent" if method_id == 7 else None,
                   future_rollout_supervision=False)
    (destination/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    log(f"COMPLETE epochs={settings['epochs']} updates={updates} checkpoint={receipt['checkpoint']}",
        status="completed", epoch=settings["epochs"], frames=frame_count)
    return receipt


__all__ = ["train"]
