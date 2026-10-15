import numpy as np
import pandas as pd
import pytest

from ecgai.data import (SUPERCLASSES, assert_no_patient_overlap, assert_splits_disjoint, build_label_matrix,
                        codes_to_superclasses, split_indices)

SCP_MAP = {"NORM": "NORM", "IMI": "MI", "ASMI": "MI", "NDT": "STTC", "LAFB": "CD", "LVH": "HYP"}
DATA_CFG = {"train_folds": [1, 2, 3, 4, 5, 6, 7, 8], "val_folds": [9], "test_folds": [10]}


def test_codes_map_to_superclasses_in_fixed_order():
    assert codes_to_superclasses({"LVH": 100, "IMI": 50, "SR": 0}, SCP_MAP) == ["MI", "HYP"]


def test_non_diagnostic_codes_are_ignored():
    assert codes_to_superclasses({"SR": 0, "PACE": 100}, SCP_MAP) == []


def test_min_likelihood_filter():
    assert codes_to_superclasses({"IMI": 15, "NORM": 100}, SCP_MAP, min_likelihood=50) == ["NORM"]


def test_label_matrix_drops_unlabelled_records():
    df = pd.DataFrame({"scp_codes": [{"NORM": 100}, {"SR": 0}, {"IMI": 100, "NDT": 50}]})
    Y, keep = build_label_matrix(df, SCP_MAP)
    assert keep.tolist() == [True, False, True]
    assert Y.shape == (3, len(SUPERCLASSES))
    assert Y[2, SUPERCLASSES.index("MI")] == 1 and Y[2, SUPERCLASSES.index("STTC")] == 1


def test_official_fold_split():
    folds = np.array([1, 5, 8, 9, 10, 10, 3])
    s = split_indices(folds, DATA_CFG)
    assert s["train"].tolist() == [0, 1, 2, 6]
    assert s["val"].tolist() == [3]
    assert s["test"].tolist() == [4, 5]
    assert_splits_disjoint(s)


def test_patient_overlap_is_detected():
    s = {"train": np.array([0, 1]), "val": np.array([2]), "test": np.array([3])}
    assert_no_patient_overlap(np.array([10, 11, 12, 13]), s)
    with pytest.raises(AssertionError):
        assert_no_patient_overlap(np.array([10, 11, 12, 10]), s)
