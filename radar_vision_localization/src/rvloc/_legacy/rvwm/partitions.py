"""One immutable source partition shared by preprocessing and fusion.

The detector's frozen manifest is canonical and is reused without reformatting.
Historical configurations without ``split_manifest`` retain their explicitly
recorded legacy intersection for reproducibility. New runs must bind this file.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

POLICY='unified_episode_manifest_v1'
FAMILY_POLICY='scene_family_70_15_15_v1'
EPISODE_SAMPLING='uniform_episode_v1'
ROLES=('train','val','test')


def capture_family(metadata,episode):
    """The family is provenance, never an inference feature or class label."""
    family=metadata.get('scene_family_id')
    if family is None:
        if metadata.get('dataset_split_policy')==FAMILY_POLICY:
            raise ValueError('Family-split source is missing scene_family_id: '+episode)
        return episode
    if not isinstance(family,str) or re.fullmatch(r'[A-Za-z0-9_.:-]{1,256}',family) is None:
        raise ValueError('Invalid scene_family_id: '+episode)
    for source in (metadata.get('scenario_spec',{}),metadata.get('planner_metadata',{})):
        if source.get('scene_family_id',family)!=family:
            raise ValueError('Frozen scene family differs between capture and plan: '+episode)
    return family


def family_membership(inventory,seed=20261120):
    """Deterministic 70/15/15 family counts; a family is never divided."""
    groups={}
    for row in inventory:
        family=row.get('scene_family_id')
        if not isinstance(family,str) or not family:
            raise ValueError('Explicit scene_family_id required for every family-split episode')
        groups.setdefault(family,[]).append(row['episode'])
    if len(groups)<3:raise ValueError('Family partition needs at least three independent scene families')
    ordered=sorted(groups,key=lambda family:(hashlib.sha256(f'{int(seed)}:{family}'.encode()).digest(),family))
    expected=[len(groups)*fraction for fraction in (.7,.15,.15)]
    counts=[int(value) for value in expected]
    for i in sorted(range(3),key=lambda i:(-(expected[i]-counts[i]),i))[:len(groups)-sum(counts)]:counts[i]+=1
    for i in range(3):
        if not counts[i]:
            donor=max(range(3),key=lambda j:counts[j]);counts[donor]-=1;counts[i]=1
    membership={};families={};start=0
    for role,count in zip(ROLES,counts):
        families[role]=sorted(ordered[start:start+count]);start+=count
        membership[role]=sorted(episode for family in families[role] for episode in groups[family])
    return membership,dict(policy=FAMILY_POLICY,seed=int(seed),fractions=dict(train=.7,val=.15,test=.15),
        grouping='frozen scene_family_id; variants share the same role',families=families,
        family_counts={role:len(values) for role,values in families.items()},
        ratio_unit='independent scene families; episode/frame counts may differ for unequal family sizes')


def _read(path):
    path=Path(path).expanduser().resolve()
    raw=path.read_bytes()
    return path,json.loads(raw),hashlib.sha256(raw).hexdigest()


def canonical_manifest(config):
    explicit=config.get('split_manifest')
    if not explicit:raise ValueError('Unified partition requires split_manifest')
    path,value,digest=_read(explicit)
    expected=config.get('split_manifest_sha256')
    if expected and digest!=expected:raise ValueError('Canonical split manifest SHA256 changed')
    visual=config.get('visual_split_manifest')
    if visual:
        _,_,visual_digest=_read(visual)
        if visual_digest!=digest:raise ValueError('Detector and fusion split manifests differ in exact content/SHA256')
    if config.get('split_policy',POLICY)!=POLICY:raise ValueError('Unknown active split policy')
    return path,value,digest


def unified_splits(dataset,config):
    path,manifest,digest=canonical_manifest(config)
    split=manifest.get('splits',{})
    if set(split)!=set(ROLES):raise ValueError('Manifest must define train/val/test')
    result={}
    for role in ROLES:
        rows=split[role].get('episode_ids') if isinstance(split[role],dict) else split[role]
        if not isinstance(rows,list) or not rows or any(not isinstance(x,str) for x in rows):
            raise ValueError('Nonempty explicit episode list required: '+role)
        if len(rows)!=len(set(rows)):raise ValueError('Duplicate episode inside partition: '+role)
        if isinstance(split[role],dict) and split[role].get('episode_count',len(rows))!=len(rows):
            raise ValueError('Partition episode_count is inconsistent: '+role)
        result[role]=sorted(rows)
    all_ids=[e for role in ROLES for e in result[role]]
    if len(set(all_ids))!=len(all_ids):raise ValueError('Episode appears in multiple partitions')
    available=set(dataset.episodes)
    if set(all_ids)!=available:
        missing=sorted(set(all_ids)-available);unassigned=sorted(available-set(all_ids))
        raise ValueError('Canonical partition/data coverage mismatch; missing='+str(missing[:5])+' unassigned='+str(unassigned[:5]))
    inventory=manifest.get('episodes')
    if not isinstance(inventory,list) or any(not isinstance(r,dict) or 'episode' not in r for r in inventory):
        raise ValueError('Canonical detector episode inventory is required')
    indexed={row['episode']:row for row in inventory}
    if len(indexed)!=len(inventory) or set(indexed)!=available:
        raise ValueError('Canonical episode inventory is duplicated or incomplete')
    source_roles={};family_roles={}
    family_policy=manifest.get('request',{}).get('split_policy')==FAMILY_POLICY
    if family_policy and manifest.get('request',{}).get('training_sampling_policy')!=EPISODE_SAMPLING:
        raise ValueError('Scene-family manifest must freeze uniform-episode training sampling')
    for role in ROLES:
        for episode in result[role]:
            source=dataset.group_key(episode)
            expected=indexed[episode].get('frames_sha256')
            if not isinstance(expected,str) or not expected or not source or source!=expected:
                raise ValueError('Canonical source capture hash mismatch: '+episode)
            if getattr(dataset,'verify_sources',False):
                capture=Path(dataset.root)/'data'/episode/'frames.jsonl'
                if not capture.is_file() or hashlib.sha256(capture.read_bytes()).hexdigest()!=source:
                    raise ValueError('Canonical source frame content changed: '+episode)
            if source in source_roles and source_roles[source]!=role:
                raise ValueError('Same source capture crosses partitions: '+episode)
            source_roles[source]=role
            family=indexed[episode].get('scene_family_id',episode)
            if family_policy:
                if ('scene_family_id' not in indexed[episode]
                        or indexed[episode].get('dataset_split_policy')!=FAMILY_POLICY
                        or indexed[episode].get('training_sampling_policy')!=EPISODE_SAMPLING):
                    raise ValueError('Missing scene family/sampling inventory: '+episode)
                if hasattr(dataset,'scene_family_id') and dataset.scene_family_id(episode)!=family:
                    raise ValueError('Source scene family changed: '+episode)
                if getattr(dataset,'verify_sources',False):
                    metadata_path=Path(dataset.root)/'data'/episode/'metadata.json'
                    raw=metadata_path.read_bytes();metadata=json.loads(raw)
                    if (hashlib.sha256(raw).hexdigest()!=indexed[episode].get('metadata_sha256')
                            or capture_family(metadata,episode)!=family
                            or metadata.get('dataset_split_policy')!=FAMILY_POLICY
                            or metadata.get('training_sampling_policy')!=EPISODE_SAMPLING):
                        raise ValueError('Frozen source family/metadata changed: '+episode)
            if family in family_roles and family_roles[family]!=role:
                raise ValueError('Same scene family crosses partitions: '+episode)
            family_roles[family]=role
    if family_policy:
        receipt=manifest.get('family_partition',{})
        expected,verified=family_membership(inventory,receipt.get('seed',20261120))
        if result!=expected or receipt!=verified:
            raise ValueError('Scene-family split differs from its frozen deterministic membership')
    forced=manifest.get('request',{}).get('forced_validation_episodes',[])
    if not set(forced)<=set(result['val']):raise ValueError('Previously inspected forced-validation episodes moved out of validation')
    # Membership is independent of a caller's seed. Never silently reshuffle.
    return result


def active_splits(dataset,config):
    """Final eligible IDs: canonical for current runs, legacy for old configs."""
    if config.get('split_manifest'):return unified_splits(dataset,config)
    actual=dataset.splits(int(config.get('seed',20261114)))
    visual=config.get('visual_split_manifest')
    if visual:
        _,manifest,_=_read(visual)
        actual={role:sorted(set(actual[role])&set(manifest['splits'][role]['episode_ids'])) for role in ROLES}
    return {role:sorted(actual[role]) for role in ROLES}


def partition_receipt(config):
    if not config.get('split_manifest'):
        return dict(policy='legacy_hash_detector_intersection',historical_configuration=True)
    path,manifest,digest=canonical_manifest(config)
    result=dict(policy=POLICY,path=str(path),sha256=digest,
        detector_uses_same_manifest=True,
        counts={role:len(manifest['splits'][role]['episode_ids']) for role in ROLES},
        membership_seed_semantics='canonical explicit membership, never recomputed from training RNG seed',
        split_source_policy=manifest.get('request',{}).get('split_policy'),
        exposure_notice='Test membership is not proof of an unused holdout; keep prior-use audit separately.')
    if manifest.get('request',{}).get('split_policy')==FAMILY_POLICY:
        result.update(family_partition=manifest['family_partition'],
            training_sampling_policy=manifest['request']['training_sampling_policy'])
    return result


def checkpoint_partition_receipt(metadata,dataset,config):
    """Verify actual checkpoint lineage without relabelling subset training."""
    split=active_splits(dataset,config)
    trained=metadata.get('train_episodes',metadata.get('training_episodes',metadata.get('episodes',[])))
    validation=metadata.get('validation_episodes',[])
    if len(trained)!=len(set(trained)) or len(validation)!=len(set(validation)):
        raise ValueError('Checkpoint episode lineage contains duplicates')
    if not set(trained)<=set(split['train']):raise ValueError('Checkpoint training overlaps active validation/test partition')
    if not set(validation)<=set(split['val']):raise ValueError('Checkpoint model-selection episodes are outside active validation partition')
    return dict(active_partition=partition_receipt(config),actual_training_episodes=len(trained),
        actual_validation_episodes=len(validation),assigned_training_episodes=len(split['train']),
        trained_on_entire_assigned_partition=bool(trained) and set(trained)==set(split['train']),
        actual_training_ids=sorted(trained),actual_validation_ids=sorted(validation),
        status='recorded_training_subset' if set(trained)!=set(split['train']) else 'entire_training_partition',
        previous_training_not_retroactively_relabelled=True)
