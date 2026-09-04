"""
Preprocess NBA combine + career data.

Core rule this script enforces: the final dataset only contains players
who appear in BOTH the combine data and the career data. A combine
prospect with no career row has no outcome to predict, and a career
player with no combine row has no features to predict from -- neither
is usable for training, and career-only players must not leak into the
data used for analysis/comparison.

Matching combine prospects to career rows is not a plain name join:
- Combine names are "Last, First"; career names are "First Last".
- Some names contain accents (e.g. "Théo Maledon") that are spelled
  without accents in the other file.
- A handful of names are shared by two different real players drafted
  in different years (e.g. two "Dee Brown"s, 1990 and 2006). We
  disambiguate using the fact that the NBA combine happens the same
  spring as the draft, so combine YEAR should equal (or be one year
  before, for early entrants who returned to school) the player's
  DraftYr. Matches more than a year apart are treated as different
  people, not merged.
"""

import re
import unicodedata

import numpy as np
import pandas as pd

RAW_COMBINE_PATH = "data/raw/nba-prospect-combine-data.csv"
RAW_CAREER_PATH = "data/raw/nba-allplayers-careeraverage-data-trimmed.csv"

COMBINE_OUT_PATH = "data/processed/combine_processed.csv"
CAREER_OUT_PATH = "data/processed/career_processed.csv"
MERGED_OUT_PATH = "data/processed/player_data.csv"

FEATURE_COLUMNS = [
    "HGT", "WGT", "BMI", "WNGSPN", "STNDRCH", "BAR",
    "STNDVERT", "LANE", "SPRINT",
]

# Career columns used to define the outcome label.
OUTCOME_COLUMNS = ["VORP", "WS", "WS/48", "BPM", "PPG", "RPG", "APG", "MPG", "G", "Yrs"]


def combine_name_to_first_last(name: str) -> str:
    """Convert combine's "Last, First" format to "First Last"."""
    if "," in name:
        last, first = name.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return name.strip()


def normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace, for matching only."""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^a-z0-9 ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def clean_zero_as_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLUMNS:
        df.loc[df[col] == 0, col] = np.nan
    return df


def load_combine() -> pd.DataFrame:
    df = pd.read_csv(RAW_COMBINE_PATH)
    df["PLAYER"] = df["PLAYER"].apply(combine_name_to_first_last)
    df["match_key"] = df["PLAYER"].apply(normalize_name)

    # BAR arrives as a percentage string ("105.5%"); the rest are numeric
    # but use the literal string "NA" for missing values.
    df["BAR"] = df["BAR"].astype(str).str.rstrip("%")
    for col in df.columns:
        if col not in ("PLAYER", "POS", "match_key"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return clean_zero_as_missing(df)


def load_career() -> pd.DataFrame:
    df = pd.read_csv(RAW_CAREER_PATH)
    df["Player"] = df["Player"].str.strip()
    df["match_key"] = df["Player"].apply(normalize_name)
    return df


def match_combine_to_career(combine: pd.DataFrame, career: pd.DataFrame) -> pd.DataFrame:
    """Return (combine_idx, career_idx) pairs for a clean 1:1 match."""
    combine = combine.reset_index().rename(columns={"index": "combine_idx"})
    career = career.reset_index().rename(columns={"index": "career_idx"})

    candidates = pd.merge(
        combine[["combine_idx", "match_key", "YEAR"]],
        career[["career_idx", "match_key", "DraftYr"]],
        on="match_key",
        how="inner",
    )
    candidates["year_diff"] = (candidates["YEAR"] - candidates["DraftYr"]).abs()
    candidates = candidates[candidates["year_diff"] <= 1].sort_values("year_diff")

    # Greedily keep the closest-year match, then enforce 1:1 on both sides.
    candidates = candidates.drop_duplicates("career_idx", keep="first")
    candidates = candidates.drop_duplicates("combine_idx", keep="first")
    return candidates[["combine_idx", "career_idx"]]


def build_target(career: pd.DataFrame) -> pd.Series:
    q25, q70, q90 = career["VORP"].quantile([0.25, 0.70, 0.90])

    def categorize(vorp: float) -> str:
        if vorp <= q25:
            return "Bust"
        if vorp <= q70:
            return "Role Player"
        if vorp <= q90:
            return "Starter"
        return "Star"

    return career["VORP"].apply(categorize)


def main():
    combine = load_combine()
    career = load_career()

    pairs = match_combine_to_career(combine, career)

    matched_combine = combine.loc[pairs["combine_idx"]].reset_index(drop=True)
    matched_career = career.loc[pairs["career_idx"]].reset_index(drop=True)

    # Drop players missing the outcome metric used for the label -- no
    # career games logged means there is nothing to predict.
    keep = matched_career["VORP"].notna()
    matched_combine, matched_career = matched_combine[keep].reset_index(drop=True), matched_career[keep].reset_index(drop=True)

    matched_career["category"] = build_target(matched_career)

    # Save the trimmed, intersection-only versions of each raw file.
    matched_combine.drop(columns=["match_key"]).to_csv(COMBINE_OUT_PATH, index=False)
    matched_career.drop(columns=["match_key"]).to_csv(CAREER_OUT_PATH, index=False)

    # Median-impute the small remaining gaps in the feature columns.
    features = matched_combine[FEATURE_COLUMNS].copy()
    features = features.fillna(features.median(numeric_only=True))

    merged = pd.concat(
        [
            matched_combine[["YEAR", "PLAYER", "POS"]],
            features,
            matched_career[["DraftYr", "category"] + [c for c in OUTCOME_COLUMNS if c != "G"] + ["G"]],
        ],
        axis=1,
    )
    merged.to_csv(MERGED_OUT_PATH, index=False)

    print(f"Combine rows (raw): {len(combine)}")
    print(f"Career rows (raw): {len(career)}")
    print(f"Matched players (combine ∩ career): {len(merged)}")
    print("\nOutcome category distribution:")
    print(merged["category"].value_counts())
    print(f"\nWrote:\n  {COMBINE_OUT_PATH}\n  {CAREER_OUT_PATH}\n  {MERGED_OUT_PATH}")


if __name__ == "__main__":
    main()
