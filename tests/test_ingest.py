import numpy as np

from stonetwin.stages.ingest import flag_blurriest


def test_flags_lowest_scores_by_rank():
    scores = np.array([50.0, 10.0, 30.0, 5.0, 40.0, 20.0, 60.0, 70.0, 80.0, 90.0])
    flags = flag_blurriest(scores, 20)  # 20 % of 10 -> 2 images
    assert flags.sum() == 2
    assert set(np.flatnonzero(flags)) == {1, 3}


def test_zero_percentile_flags_nothing():
    assert not flag_blurriest(np.arange(5.0), 0).any()


def test_rounding_of_reject_count():
    assert flag_blurriest(np.arange(42.0), 5).sum() == 2  # 2.1 -> 2
    assert flag_blurriest(np.arange(155.0), 5).sum() == 8  # 7.75 -> 8
