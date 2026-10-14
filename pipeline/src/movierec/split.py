"""Temporal leave-last-out split.

For every user the most recent rating is held out and everything before it is
training data. Ties on timestamp (MovieLens users often rate in bursts) are
broken by movieId so the split is deterministic.

What this does not prevent: other users' ratings that happen after a given
user's held-out timestamp stay in training. That is the usual leave-last-out
trade-off and is noted in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame


def leave_last_out(ratings: pd.DataFrame, min_history: int = 2) -> Split:
    """Hold out each user's last interaction.

    Users with fewer than ``min_history`` ratings stay entirely in training,
    because holding out their only rating would leave nothing to query with.
    """
    ordered = ratings.sort_values(["userId", "timestamp", "movieId"], kind="stable")
    sizes = ordered.groupby("userId")["movieId"].transform("size")
    is_last = ~ordered.duplicated("userId", keep="last")
    held_out = is_last & (sizes >= min_history)
    return Split(
        train=ordered[~held_out].reset_index(drop=True),
        test=ordered[held_out].reset_index(drop=True),
    )
