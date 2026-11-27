import numpy as np
import pytest
from gnn_research.audit_graph_coverage import AuditError,recover_counts


def test_recovery_weights_shared_nodes_by_their_actual_edges():
    counts=recover_counts(dict(evidence_candidate_degree=np.array([1.,9.]),
        evidence_distinct_member_degree=np.array([1.,2.]),edge_count=np.array([4,6]),
        candidate_degree=np.array([3,7]),unique_evidence_nodes=np.array([1,1])))
    assert counts['shared_edge_coverage']==.9
    assert counts['shared_edge_count']==9 and counts['edge_count']==10


def test_missing_numerator_arrays_is_not_observed_zero():
    with pytest.raises(AuditError,match='Missing'):recover_counts({'edge_count':np.array([5])})


def test_inconsistent_raw_and_reported_edge_totals_are_rejected():
    with pytest.raises(AuditError,match='edge_count'):recover_counts(dict(
        evidence_candidate_degree=np.array([2]),evidence_distinct_member_degree=np.array([2]),edge_count=np.array([3])))


def test_no_edges_remains_unmeasured_not_zero_ratio():
    result=recover_counts(dict(evidence_candidate_degree=np.array([]),
        evidence_distinct_member_degree=np.array([]),edge_count=np.array([0])))
    assert result['shared_edge_count']==0 and result['shared_edge_coverage'] is None
