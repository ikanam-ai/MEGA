from __future__ import annotations

import pandas as pd

from analysis.reproduce_publication_stats import assign_clusters


def test_cluster_labels_follow_arousal_order() -> None:
    profiles = pd.DataFrame(
        [
            {"model_id": "low_a", "valence": 0.70, "arousal": 0.20, "dominance": 0.55, "generic_rate": 0.60},
            {"model_id": "low_b", "valence": 0.71, "arousal": 0.22, "dominance": 0.56, "generic_rate": 0.58},
            {"model_id": "mid_a", "valence": 0.69, "arousal": 0.45, "dominance": 0.66, "generic_rate": 0.15},
            {"model_id": "mid_b", "valence": 0.70, "arousal": 0.47, "dominance": 0.65, "generic_rate": 0.14},
            {"model_id": "high_a", "valence": 0.72, "arousal": 0.75, "dominance": 0.78, "generic_rate": 0.01},
            {"model_id": "high_b", "valence": 0.71, "arousal": 0.77, "dominance": 0.80, "generic_rate": 0.02},
        ]
    )
    clustered, tree = assign_clusters(profiles)
    cluster_means = clustered.groupby("cluster")["arousal"].mean()
    assert list(cluster_means.index) == [1, 2, 3]
    assert cluster_means.is_monotonic_increasing
    assert tree.shape == (5, 4)
