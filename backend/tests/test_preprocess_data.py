import pandas as pd
import pytest

from preprocess_data import (
    FEATURE_COLUMNS,
    build_target,
    clean_zero_as_missing,
    combine_name_to_first_last,
    match_combine_to_career,
    normalize_name,
)


class TestCombineNameToFirstLast:
    def test_converts_last_comma_first(self):
        assert combine_name_to_first_last("Washington, PJ") == "PJ Washington"

    def test_leaves_name_without_comma_unchanged(self):
        assert combine_name_to_first_last("LeBron James") == "LeBron James"

    def test_strips_surrounding_whitespace(self):
        assert combine_name_to_first_last("  Doe,  John  ") == "John Doe"


class TestNormalizeName:
    def test_lowercases(self):
        assert normalize_name("LeBron James") == "lebron james"

    def test_strips_accents(self):
        assert normalize_name("Théo Maledon") == "theo maledon"

    def test_strips_punctuation(self):
        assert normalize_name("P.J. Washington") == "pj washington"

    def test_collapses_whitespace(self):
        assert normalize_name("John   Doe") == "john doe"


class TestMatchCombineToCareer:
    def test_matches_same_draft_year(self):
        combine = pd.DataFrame({"match_key": ["john doe"], "YEAR": [2010]})
        career = pd.DataFrame({"match_key": ["john doe"], "DraftYr": [2010]})
        pairs = match_combine_to_career(combine, career)
        assert pairs.to_dict("records") == [{"combine_idx": 0, "career_idx": 0}]

    def test_matches_within_one_year_tolerance(self):
        # An early-entrant who attended the combine, then returned to
        # school and was actually drafted the following year.
        combine = pd.DataFrame({"match_key": ["john doe"], "YEAR": [2010]})
        career = pd.DataFrame({"match_key": ["john doe"], "DraftYr": [2011]})
        pairs = match_combine_to_career(combine, career)
        assert len(pairs) == 1

    def test_does_not_match_beyond_one_year(self):
        combine = pd.DataFrame({"match_key": ["john doe"], "YEAR": [2010]})
        career = pd.DataFrame({"match_key": ["john doe"], "DraftYr": [2013]})
        pairs = match_combine_to_career(combine, career)
        assert len(pairs) == 0

    def test_name_with_no_career_match_is_dropped(self):
        combine = pd.DataFrame({"match_key": ["nobody knows me"], "YEAR": [2010]})
        career = pd.DataFrame({"match_key": ["john doe"], "DraftYr": [2010]})
        pairs = match_combine_to_career(combine, career)
        assert len(pairs) == 0

    def test_disambiguates_two_different_people_with_the_same_name(self):
        # Two real "Dee Brown"s: one drafted 1990, one drafted 2006. Each
        # combine appearance should match the correctly-dated career row,
        # not get cross-matched to the wrong person.
        combine = pd.DataFrame(
            {"match_key": ["dee brown", "dee brown"], "YEAR": [1990, 2006]}
        )
        career = pd.DataFrame(
            {"match_key": ["dee brown", "dee brown"], "DraftYr": [1990, 2006]}
        )
        pairs = match_combine_to_career(combine, career).sort_values("combine_idx")
        records = pairs.to_dict("records")
        assert records == [
            {"combine_idx": 0, "career_idx": 0},
            {"combine_idx": 1, "career_idx": 1},
        ]

    def test_each_side_matched_at_most_once(self):
        # A combine row should never be matched to two career rows, and
        # vice versa, even if multiple candidates pass the year filter.
        combine = pd.DataFrame({"match_key": ["john doe"], "YEAR": [2010]})
        career = pd.DataFrame(
            {"match_key": ["john doe", "john doe"], "DraftYr": [2010, 2011]}
        )
        pairs = match_combine_to_career(combine, career)
        assert len(pairs) == 1
        # The closer year (2010, diff=0) should win over 2011 (diff=1).
        assert pairs.iloc[0]["career_idx"] == 0


class TestCleanZeroAsMissing:
    def test_zero_in_feature_column_becomes_nan(self):
        # Regression test: Kris Dunn's 2016 combine row has a literal 0.0
        # for WGT and BMI in the raw CSV -- a missing measurement encoded
        # as 0 instead of "NA". That must not be treated as a real 0 lb,
        # 0 BMI prospect.
        row = {col: 100.0 for col in FEATURE_COLUMNS}
        row["WGT"] = 0.0
        row["BMI"] = 0.0
        df = pd.DataFrame([row])

        cleaned = clean_zero_as_missing(df)

        assert pd.isna(cleaned.loc[0, "WGT"])
        assert pd.isna(cleaned.loc[0, "BMI"])
        # Untouched feature columns keep their real (non-zero) values.
        assert cleaned.loc[0, "HGT"] == 100.0

    def test_nonzero_values_are_left_alone(self):
        row = {col: 100.0 for col in FEATURE_COLUMNS}
        df = pd.DataFrame([row])

        cleaned = clean_zero_as_missing(df)

        for col in FEATURE_COLUMNS:
            assert cleaned.loc[0, col] == 100.0


class TestBuildTarget:
    @pytest.fixture
    def career(self):
        # 100 evenly spaced VORP values gives exact quantile boundaries:
        # q25=25.75, q70=70.3, q90=90.1
        return pd.DataFrame({"VORP": list(range(1, 101))})

    def test_quantile_boundaries(self, career):
        categories = build_target(career)
        assert categories.iloc[0] == "Bust"          # VORP = 1
        assert categories.iloc[24] == "Bust"          # VORP = 25 (<= 25.75)
        assert categories.iloc[25] == "Role Player"   # VORP = 26
        assert categories.iloc[69] == "Role Player"   # VORP = 70 (<= 70.3)
        assert categories.iloc[70] == "Starter"       # VORP = 71
        assert categories.iloc[89] == "Starter"       # VORP = 90 (<= 90.1)
        assert categories.iloc[90] == "Star"          # VORP = 91
        assert categories.iloc[99] == "Star"          # VORP = 100

    def test_bucket_sizes_match_target_quantiles(self, career):
        counts = build_target(career).value_counts()
        assert counts["Bust"] == 25
        assert counts["Role Player"] == 45
        assert counts["Starter"] == 20
        assert counts["Star"] == 10
