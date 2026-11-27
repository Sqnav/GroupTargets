"""Offline, immutable ROI-to-GT labels and prediction-independent evaluation.

This tool is deliberately outside rvloc/src: running training is unchanged.
GT identity is supervision/evaluation only, never an online association input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from rvloc.config import ExperimentConfig, external_path
from rvloc.data import load_manifest, load_labels
from rvloc.evaluation import metrics, episode_bootstrap
from rvloc.runtime import file_sha256, write_json, jsonable
from rvloc.supervision import match_direct_ray_labels

SCHEMA = 'rvloc.fixed_roi_gt_index.v1'
FRAME_SCHEMA = 'rvloc.fixed_roi_gt_frame.v1'


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def bbox_assignment_audit(detected_xyxy, gt_ids, gt_xywh):
    """Maximise positive IoU, permitting unmatched rows; diagnostic only.

    No threshold is selected from DEV. Any positive overlap is admissible;
    ambiguities/duplicates are retained as flags, not certified true identity.
    """
    boxes = np.asarray(detected_xyxy, float).reshape(-1, 4)
    raw = np.asarray(gt_xywh, float).reshape(-1, 4)
    _check(len(raw) == len(gt_ids), 'GT box/ID length mismatch')
    _check(np.isfinite(boxes).all() and np.isfinite(raw).all(), 'Nonfinite boxes')
    gt = raw.copy(); gt[:, 2:] += gt[:, :2]
    area_a = np.prod(np.maximum(0., boxes[:, 2:] - boxes[:, :2]), -1)
    area_b = np.prod(np.maximum(0., raw[:, 2:]), -1)
    overlap = np.prod(np.maximum(0., np.minimum(boxes[:, None, 2:], gt[None, :, 2:])
                               - np.maximum(boxes[:, None, :2], gt[None, :, :2])), -1)
    union = area_a[:, None] + area_b[None] - overlap
    iou = overlap / np.maximum(union, np.finfo(float).tiny)
    positive = iou > 0.
    n, m = iou.shape
    assignment = []
    if n and m:
        cost = np.column_stack((np.where(positive, 1.-iou, 2.), np.ones((n, n))))
        rr, cc = linear_sum_assignment(cost)
        assignment = [dict(detection_index=int(r), gt_member_id=gt_ids[int(c)],
                           gt_index=int(c), iou=float(iou[r, c]),
                           detection_overlaps_gt_count=int(positive[r].sum()),
                           gt_overlaps_detection_count=int(positive[:, c].sum()))
                      for r, c in zip(rr, cc) if c < m and positive[r, c]]
    return dict(policy='maximum positive IoU sum with dummy unmatched rows; audit only',
                source='saved instance-segmentation-derived GT bboxes; masks not required',
                assignments=assignment,
                multiple_gt_overlap_detections=np.flatnonzero(positive.sum(1) > 1).tolist(),
                multiple_detection_overlap_gt=[gt_ids[i] for i in np.flatnonzero(positive.sum(0) > 1)],
                unmatched_detection_indices=sorted(set(range(n)) - {r['detection_index'] for r in assignment}),
                unmatched_gt_ids=[x for x in gt_ids if x not in {r['gt_member_id'] for r in assignment}],
                not_an_identity_oracle=True)


def build_frame(observation, labels, frame_record, *, model_config, gt_capture=None, capture_identity=None):
    """The frozen assignment is exactly the current trainer's geometry rule."""
    features = observation.features
    matching = match_direct_ray_labels(features, labels.positions, model_config)
    pairs = [dict(detection_index=int(r), gt_index=int(c), gt_member_id=labels.member_ids[int(c)],
                  gt_position_m=labels.positions[int(c)].tolist(), target_depth_m=float(depth),
                  angle_error_deg=float(angle))
             for r, c, depth, angle in zip(matching['rows'], matching['cols'], matching['depths_m'],
                                            matching['diagnostics']['matched_angle_errors_deg'])]
    out = dict(schema=FRAME_SCHEMA, episode_id=observation.episode_id,
        frame_idx=observation.frame_idx, physical_time_s=observation.physical_time_s,
        normal_sha256=frame_record.normal_sha256, labels_sha256=frame_record.labels_sha256,
        visual_sha256=observation.source_hashes['visual'], detection_count=len(features['bbox_xyxy']),
        bbox_xyxy=features['bbox_xyxy'], ray_origins=features['ray_origins'],
        ray_directions=features['ray_directions'], ray_depth_bounds=features['ray_depth_bounds'],
        gt_member_ids=list(labels.member_ids), gt_positions_m=labels.positions,
        training_equivalent_assignment=pairs, training_assignment_diagnostics=matching['diagnostics'],
        unmatched_detection_indices=sorted(set(range(len(features['bbox_xyxy']))) - set(matching['rows'].tolist())),
        supervision_only=True, normal_inference_input=False, capture_identity=capture_identity)
    if gt_capture is not None:
        group = gt_capture['group']
        _check(set(group['ids']) == set(labels.member_ids), 'Capture/label member IDs differ')
        audit = bbox_assignment_audit(features['bbox_xyxy'], group['ids'], group['bboxes_xywh'])
        alternative = {r['detection_index']: r['gt_member_id'] for r in audit['assignments']}
        shared = [p for p in pairs if p['detection_index'] in alternative]
        audit['shared_detection_count'] = len(shared)
        audit['shared_assignment_disagreements'] = [dict(detection_index=p['detection_index'],
             training_gt_member_id=p['gt_member_id'], bbox_gt_member_id=alternative[p['detection_index']])
             for p in shared if p['gt_member_id'] != alternative[p['detection_index']]]
        out['instance_bbox_assignment_audit'] = audit
    return jsonable(out)


def _capture_episode(manifest, episode_id):
    first = manifest.episodes[episode_id][0]
    normal_file = manifest._resolve(first.normal_path)
    export_path = normal_file.parent.parent/'export_receipt.json'
    export = json.loads(export_path.read_text())
    _check(export['episode_id'] == episode_id, 'Export episode differs')
    gen = export['generation_receipt']
    _check(file_sha256(gen['path']) == gen['sha256'], 'Generation receipt SHA differs')
    generation = json.loads(Path(gen['path']).read_text())
    _check(generation == gen['content'], 'Inline generation receipt differs')
    # Explicitly constrained branch path; no enumeration of historical TEST.
    capture = manifest.data_root/'dataset_construction/camera_center_branch_20261126/data'/episode_id/'frames.jsonl'
    capture_sha = file_sha256(capture)
    _check(capture_sha == generation['identity']['capture_sha256'], 'Original capture SHA differs')
    rows = [json.loads(line) for line in capture.read_text().splitlines() if line.strip()]
    by_index = {row['frame_idx']: row for row in rows}
    _check(len(by_index) == len(rows) == len(manifest.episodes[episode_id]), 'Capture frame coverage differs')
    proofs = {}
    for ref in export['frames']:
        proof = json.loads(manifest._read(ref['path'], ref['sha256']))
        _check(proof['episode_id'] == episode_id, 'Frame source proof episode differs')
        _check(proof['cache_internal_receipt']['feature_builder_identity']['generation_receipt_sha256'] == gen['sha256'],
               'Frame proof does not bind this capture generation')
        proofs[proof['frame_idx']] = proof
    identity = dict(path=str(capture), sha256=capture_sha,
                    export_receipt_path=str(export_path), export_receipt_sha256=file_sha256(export_path),
                    generation_receipt_sha256=gen['sha256'])
    return by_index, proofs, identity


def prepare(config_path, output_dir):
    config = ExperimentConfig.load(config_path)
    _check(config.manifest is not None, 'Explicit ready manifest required')
    manifest = load_manifest(config.manifest)
    output = external_path(output_dir, field='fixed annotation output')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol = dict(schema=SCHEMA, manifest_path=str(manifest.path), manifest_sha256=manifest.sha256,
        config_path=str(config.path), config_sha256=config.sha256,
        assignment_source_sha256=file_sha256(inspect.getfile(match_direct_ray_labels)),
        tool_sha256=file_sha256(__file__), model_config=config.raw.get('model', {}),
        frozen_before_predictions=True, supervision_only=True, normal_inference_input=False,
        inference_uses_gt=False, historical_test_opened=False,
        identity_namespace=['episode_id', 'gt_member_id'],
        policy='training-equivalent public-ray angular assignment, independent of model predictions',
        bbox_audit_policy='maximum positive IoU sum; diagnostic only; does not change training labels',
        frames=[], summaries={})
    write_json(output/'protocol.json', {k:v for k,v in protocol.items() if k != 'frames'})
    captures = {}
    for split in ('train', 'dev'):
        counts = Counter()
        for observation in manifest.observations(split, modality='rgb'):
            ep = observation.episode_id
            if ep not in captures:
                captures.clear()  # Only current episode capture/proofs in memory.
                captures[ep] = _capture_episode(manifest, ep)
            cap, proofs, identity = captures[ep]
            record = next(r for r in manifest.episodes[ep] if r.frame_idx == observation.frame_idx)
            proof = proofs[record.frame_idx]
            _check((proof['normal_sha256'], proof['labels_sha256'], proof['visual_sha256']) ==
                   (record.normal_sha256, record.labels_sha256, observation.source_hashes['visual']),
                   'Source capture proof does not bind the current normal/labels/visual')
            labels = load_labels(manifest, observation)
            frame = build_frame(observation, labels, record, model_config=config.raw.get('model', {}),
                                gt_capture=cap[record.frame_idx], capture_identity=identity)
            path = output/'frames'/ep/f'frame_{record.frame_idx:05d}.json'
            write_json(path, frame)
            protocol['frames'].append(dict(episode_id=ep, frame_idx=record.frame_idx, split=split,
                                           path=path.relative_to(output).as_posix(), sha256=file_sha256(path)))
            audit = frame['instance_bbox_assignment_audit']
            counts.update(frames=1, detections=frame['detection_count'], gt_members=len(labels.member_ids),
                          frozen_pairs=len(frame['training_equivalent_assignment']),
                          bbox_pairs=len(audit['assignments']), bbox_shared=audit['shared_detection_count'],
                          bbox_disagreements=len(audit['shared_assignment_disagreements']),
                          multiple_gt_overlap_detections=len(audit['multiple_gt_overlap_detections']),
                          multiple_detection_overlap_gt=len(audit['multiple_detection_overlap_gt']),
                          unmatched_detections=len(frame['unmatched_detection_indices']))
            if counts['frames'] % 250 == 0:
                print(json.dumps(dict(stage='fixed_gt_prepare',split=split,**counts)), flush=True)
        protocol['summaries'][split] = dict(counts)
    protocol['wall_s'] = time.perf_counter()-started
    path = write_json(output/'fixed_roi_gt_index.json', protocol)
    write_json(output/'summary.json', dict(status='prepared', index_path=str(path), index_sha256=file_sha256(path),
                                         summaries=protocol['summaries'], wall_s=protocol['wall_s']))
    return dict(status='prepared', index_path=str(path), index_sha256=file_sha256(path), summaries=protocol['summaries'])


def score_fixed_frame(prediction, frozen):
    """Direct row-to-fixed-GT errors: no prediction-dependent assignment."""
    _check(frozen['schema'] == FRAME_SCHEMA, 'Unknown fixed frame schema')
    _check((prediction['episode_id'], prediction['frame_idx'], prediction['physical_time_s']) ==
           (frozen['episode_id'], frozen['frame_idx'], frozen['physical_time_s']), 'Frame/time differs')
    source = prediction.get('source_hashes', {})
    _check(source.get('normal') == frozen['normal_sha256'] and source.get('visual') == frozen['visual_sha256'],
           'Prediction sensor identity differs from frozen ROI identity')
    n = frozen['detection_count']
    points = np.asarray(prediction['positions'], float).reshape(-1, 3)
    valid = np.asarray(prediction['valid'])
    boxes = np.asarray(prediction['bbox_xyxy'], float).reshape(-1, 4)
    _check(points.shape == (n, 3) and valid.shape == (n,) and valid.dtype == np.bool_, 'Detection-aligned output shape differs')
    _check(np.array_equal(boxes, np.asarray(frozen['bbox_xyxy'], float).reshape(-1, 4)), 'Detection bbox order/content differs')
    _check(np.isfinite(points[valid]).all(), 'Valid output contains nonfinite positions')
    assignments = frozen['training_equivalent_assignment']
    by_gt = {row['gt_member_id']:row for row in assignments}
    _check(len(by_gt) == len(assignments) and len({p['detection_index'] for p in assignments}) == len(assignments),
           'Frozen assignment is not one-to-one')
    rows = []
    for member, truth in zip(frozen['gt_member_ids'], frozen['gt_positions_m']):
        pair = by_gt.get(member)
        index = pair['detection_index'] if pair else None
        error = float(np.linalg.norm(points[index]-truth)) if index is not None and valid[index] else None
        rows.append(dict(episode_id=frozen['episode_id'], frame_idx=frozen['frame_idx'], gt_member_id=member,
                         detection_index=index, error_m=error,
                         prediction_m=points[index].tolist() if error is not None else None,
                         gt_position_m=truth))
    selected = {p['detection_index'] for p in assignments}
    diagnostics = dict(detections=n, frozen_pairs=len(assignments),
        unmatched_output_count=n-len(assignments),
        invalid_outputs=int((~valid).sum()),
        valid_unmatched_outputs=sum(bool(valid[i]) for i in range(n) if i not in selected),
        assigned_invalid_outputs=sum(not bool(valid[i]) for i in selected))
    return rows, diagnostics


def evaluate(index_path, predictions_path, output_dir):
    index_path = Path(index_path).resolve()
    index = json.loads(index_path.read_text())
    _check(index['schema'] == SCHEMA and index['supervision_only'] is True, 'Invalid frozen index')
    pred_path = Path(predictions_path).resolve()
    prediction = json.loads(pred_path.read_text())
    _check(prediction.get('schema') == 'rvloc.predictions.v1' and prediction.get('normal_inference') is True
           and prediction.get('labels_opened') is False, 'Persisted sensor-only predictions required')
    _check(prediction['method_id'] in (2,3,5,6,7), 'ROI fixed evaluation applies only to visual groups 2/3/5/6/7')
    _check(prediction['manifest_sha256'] == index['manifest_sha256'], 'Manifest identity differs')
    split = prediction['split']
    _check(split in ('train','dev'), 'Historical TEST is reserved')
    frames = {(p['episode_id'],p['frame_idx']):p for p in prediction['frames']}
    expected = {(r['episode_id'],r['frame_idx']):r for r in index['frames'] if r['split']==split}
    _check(len(frames) == len(prediction['frames']) and frames.keys() == expected.keys(), 'Missing/duplicate/extra predicted frames')
    output = external_path(output_dir, field='fixed evaluation output')
    output.mkdir(parents=True,exist_ok=False)
    rows = []; totals = Counter()
    for key, ref in expected.items():
        path = (index_path.parent/ref['path']).resolve()
        _check(path.is_relative_to(index_path.parent), 'Frozen frame path escaped index directory')
        _check(file_sha256(path) == ref['sha256'], 'Frozen frame SHA differs')
        frozen = json.loads(path.read_text())
        current, diagnostic = score_fixed_frame(frames[key], frozen)
        rows.extend(current);totals.update(diagnostic)
    summary = metrics(rows)
    summary.update(totals)
    summary['matching'] = 'fixed ROI-to-GT training correspondence; never reassigned using predicted positions'
    summary['not_tracking_identity_accuracy'] = True
    summary['uncertainty'] = episode_bootstrap(rows,draws=2000,seed=20261126)
    summary['conditional_rmse_requires_all_gt_success_and_missing_counts'] = True
    result = dict(schema='rvloc.fixed_roi_evaluation.v1',method_id=prediction['method_id'],split=split,
        manifest_sha256=index['manifest_sha256'], frozen_index_sha256=file_sha256(index_path),
        predictions_sha256=file_sha256(pred_path), normal_inference_changed=False,
        summary=summary, rows=rows,
        episodes={ep:metrics([r for r in rows if r['episode_id']==ep]) for ep in sorted({r['episode_id'] for r in rows})})
    path = write_json(output/'evaluation.json', result)
    write_json(output/'localization_metrics.json',summary)
    return dict(status='evaluated',evaluation=str(path),summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare');prep.add_argument('--config',required=True);prep.add_argument('--output-dir',required=True)
    score = commands.add_parser('evaluate');score.add_argument('--index',required=True);score.add_argument('--predictions',required=True);score.add_argument('--output-dir',required=True)
    args = parser.parse_args()
    result = prepare(args.config,args.output_dir) if args.command=='prepare' else evaluate(args.index,args.predictions,args.output_dir)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
