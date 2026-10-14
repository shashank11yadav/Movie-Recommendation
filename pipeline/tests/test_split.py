from __future__ import annotations

import pandas as pd

from movierec.split import leave_last_out


def test_one_held_out_rating_per_user(ratings: pd.DataFrame) -> None:
    split = leave_last_out(ratings)
    assert split.test["userId"].is_unique
    assert set(split.test["userId"]) == set(ratings["userId"])
    assert len(split.train) + len(split.test) == len(ratings)


def test_no_leakage(ratings: pd.DataFrame) -> None:
    split = leave_last_out(ratings)
    train_pairs = set(zip(split.train["userId"], split.train["movieId"], strict=True))
    for row in split.test.itertuples():
        assert (row.userId, row.movieId) not in train_pairs
        user_train = split.train[split.train["userId"] == row.userId]
        assert (user_train["timestamp"] <= row.timestamp).all()


def test_holds_out_the_latest_rating(ratings: pd.DataFrame) -> None:
    test = leave_last_out(ratings).test.set_index("userId")["movieId"]
    assert test.to_dict() == {1: 10, 2: 3, 3: 7, 4: 10}


def test_timestamp_ties_break_on_movie_id() -> None:
    tied = pd.DataFrame(
        {"userId": [1, 1, 1], "movieId": [9, 4, 6], "rating": [4.0] * 3, "timestamp": [5, 5, 5]}
    )
    for _ in range(3):
        shuffled = tied.sample(frac=1.0, random_state=_)
        assert leave_last_out(shuffled).test["movieId"].tolist() == [9]


def test_users_with_a_single_rating_stay_in_train() -> None:
    lone = pd.DataFrame({"userId": [1], "movieId": [1], "rating": [4.0], "timestamp": [1]})
    split = leave_last_out(lone)
    assert split.test.empty
    assert len(split.train) == 1


def test_nested_split_for_validation_is_also_leak_free(ratings: pd.DataFrame) -> None:
    outer = leave_last_out(ratings)
    inner = leave_last_out(outer.train)
    outer_pairs = set(zip(outer.test["userId"], outer.test["movieId"], strict=True))
    inner_train_pairs = set(zip(inner.train["userId"], inner.train["movieId"], strict=True))
    assert not outer_pairs & inner_train_pairs
