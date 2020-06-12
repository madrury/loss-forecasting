import numpy as np

def diff_within_groups(x, groups):
    group_boundaries = np.diff(groups).astype(bool)
    return np.diff(x)[~group_boundaries]

def count_elements_in_groups(groups):
    return np.diff(np.concatenate([
        [0],
        # Look for indexes of boundaries between groups.
        np.where(np.diff(groups) == 1)[0] + 1,
        [len(groups)]
    ]))

def count_elements_in_group_differences(group_counts):
    return np.concatenate([[0], np.cumsum(group_counts - 1)])