"""
Generate report/REPORT.md and its figures from the trained model + processed
data. Run after preprocess_data.py and model_training.py.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_PATH = "data/processed/player_data.csv"
METRICS_PATH = "data/processed/metrics.json"
FIG_DIR = "report/figures"
REPORT_PATH = "report/REPORT.md"

FEATURE_COLUMNS = [
    "HGT", "WGT", "BMI", "WNGSPN", "STNDRCH", "BAR",
    "STNDVERT", "LANE", "SPRINT",
]
FEATURE_LABELS = {
    "HGT": "Height", "WGT": "Weight", "BMI": "BMI", "WNGSPN": "Wingspan",
    "STNDRCH": "Standing Reach", "BAR": "Body Adiposity Ratio",
    "STNDVERT": "Standing Vertical", "LANE": "Lane Agility", "SPRINT": "Sprint",
}
CATEGORY_ORDER = ["Bust", "Role Player", "Starter", "Star"]
CATEGORY_COLORS = {"Bust": "#c0392b", "Role Player": "#7f8c8d", "Starter": "#2980b9", "Star": "#f39c12"}


def correlation_table(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    rows = []
    for f in FEATURE_COLUMNS:
        r, p = spearmanr(df[f], df[outcome])
        rows.append({"feature": FEATURE_LABELS[f], "r": r, "p": p})
    return pd.DataFrame(rows).sort_values("r", key=lambda s: s.abs(), ascending=False)


def plot_model_comparison(metrics: dict):
    comparison = metrics["model_comparison"]
    names = [m["name"] for m in comparison]
    cv_acc = [m["cv_accuracy"] for m in comparison]
    test_acc = [m["test_accuracy"] for m in comparison]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, cv_acc, width, label="Tuned 5-fold CV accuracy", color="#2980b9")
    ax.bar(x + width / 2, test_acc, width, label="Held-out test accuracy", color="#5da8d6")
    ax.axhline(metrics["baseline_accuracy"], color="#7f8c8d", linestyle="--", linewidth=1.5,
               label=f"Baseline ({metrics['baseline_accuracy']:.0%}, always guess majority class)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Three tuned model families, none beat the baseline")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/model_comparison.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(metrics: dict):
    importances = metrics["feature_importances"]
    labels = [FEATURE_LABELS[k] for k in importances.keys()]
    values = list(importances.values())
    colors = ["#2980b9" if v >= 0 else "#c0392b" for v in values]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Permutation importance (accuracy drop when shuffled)")
    ax.set_title(f"What the winning model ({metrics['winning_model']}) leans on")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/feature_importance.png", dpi=150)
    plt.close(fig)


def plot_correlation(corr_vorp: pd.DataFrame, corr_ppg: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, corr, title in [
        (axes[0], corr_vorp, "vs. career value (VORP)"),
        (axes[1], corr_ppg, "vs. scoring rate (PPG)"),
    ]:
        colors = ["#c0392b" if r < 0 else "#2980b9" for r in corr["r"]]
        ax.barh(corr["feature"][::-1], corr["r"][::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-0.2, 0.2)
        ax.set_title(title)
        ax.set_xlabel("Spearman correlation")
    fig.suptitle("Combine measurables barely correlate with career outcomes")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/correlations.png", dpi=150)
    plt.close(fig)


def plot_feature_by_category(df: pd.DataFrame):
    key_features = ["HGT", "WNGSPN", "STNDVERT", "SPRINT"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    for ax, f in zip(axes, key_features):
        data = [df.loc[df["category"] == c, f].dropna() for c in CATEGORY_ORDER]
        bp = ax.boxplot(data, tick_labels=CATEGORY_ORDER, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], CATEGORY_ORDER):
            patch.set_facecolor(CATEGORY_COLORS[c])
            patch.set_alpha(0.7)
        ax.set_title(FEATURE_LABELS[f])
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Distribution of combine measurables by career outcome — the boxes barely move")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/feature_by_category.png", dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    corr_vorp = correlation_table(df, "VORP")
    corr_ppg = correlation_table(df, "PPG")

    plot_model_comparison(metrics)
    plot_feature_importance(metrics)
    plot_correlation(corr_vorp, corr_ppg)
    plot_feature_by_category(df)

    n = len(df)
    cat_counts = df["category"].value_counts().reindex(CATEGORY_ORDER)
    strongest_vorp = corr_vorp.iloc[0]
    strongest_ppg = corr_ppg.iloc[0]
    n_significant_vorp = (corr_vorp["p"] < 0.05).sum()
    n_significant_ppg = (corr_ppg["p"] < 0.05).sum()

    lines = []
    lines.append("# Do NBA Combine Scores Predict Career Success?\n")
    lines.append(
        "This report evaluates whether pre-draft NBA Combine measurables "
        "(height, weight, wingspan, vertical leap, agility, sprint speed, etc.) "
        "predict how a player's career actually turns out.\n"
    )

    lines.append("## Data\n")
    lines.append(
        f"- Raw combine records: 1,678 prospects (2000–2025).\n"
        f"- Raw career-average records: 1,868 drafted players (1990–2021).\n"
        f"- **Matched players (appear in both files): {n}.** Only these players are used "
        "anywhere in this analysis — a combine prospect with no career record has no "
        "outcome to learn from, and a career player with no combine record has no "
        "measurables to predict from, so neither belongs in the dataset. See "
        "`backend/preprocess_data.py` for the name-matching logic (accent normalization "
        "+ draft-year tolerance to resolve players who share a name).\n"
    )
    lines.append(
        "- Outcome label (`category`): each player's career **VORP** (Value Over "
        "Replacement Player, a standard cumulative advanced stat combining quality of "
        "play and playing time) is bucketed into quartile/decile-based tiers: "
        "**Bust** (bottom 25%), **Role Player** (25th–70th percentile), **Starter** "
        "(70th–90th), **Star** (top 10%). Cutoffs come from the data's own distribution, "
        "not hand-picked thresholds.\n"
    )
    lines.append("| Category | Players |\n|---|---|\n")
    for c in CATEGORY_ORDER:
        lines.append(f"| {c} | {cat_counts[c]} |\n")

    lines.append("\n## Can a model predict the outcome tier from combine measurables?\n")
    lines.append(
        "Three different model families were trained on the same 9 combine measurables "
        "(height, weight, BMI, wingspan, standing reach, body adiposity ratio, standing "
        "vertical, lane agility, sprint) to predict the four-tier outcome category: "
        "**Random Forest**, **Logistic Regression**, and **Gradient Boosting**. Each was "
        "hyperparameter-tuned with `GridSearchCV` (5-fold cross-validation) rather than "
        "run with default settings, specifically so a weak result couldn't be waved away "
        "as \"wrong model\" or \"needed more tuning.\" The best-tuned model of each family "
        f"was then evaluated on a held-out test set of {metrics['n_test']} players "
        f"(trained on {metrics['n_train']}).\n"
    )
    majority_class = cat_counts.idxmax()
    lines.append(f"\n![Model comparison]({FIG_DIR.replace('report/', '')}/model_comparison.png)\n")
    lines.append("\n| Model | Tuned CV accuracy | Test accuracy |\n|---|---|---|\n")
    for m in metrics["model_comparison"]:
        lines.append(f"| {m['name']} | {m['cv_accuracy']:.1%} | {m['test_accuracy']:.1%} |\n")
    lines.append(
        f"\n**Baseline accuracy (always guess \"{majority_class}\", the most common outcome): "
        f"{metrics['baseline_accuracy']:.1%}**\n"
    )
    lines.append(
        f"\n**None of the three tuned model families beat the baseline.** The best of "
        f"them ({metrics['winning_model']}) reached {metrics['cv_accuracy_mean']:.1%} ± "
        f"{metrics['cv_accuracy_std']:.1%} cross-validated accuracy — statistically "
        "indistinguishable from just guessing the most common outcome every time — and "
        f"only {metrics['test_accuracy']:.1%} on the untouched test set. This isn't a "
        "quirk of one algorithm or under-tuned hyperparameters: a linear model, a bagged "
        "tree ensemble, and a boosted tree ensemble all land in the same place, after "
        "each was given a real grid search to find its best settings.\n"
    )
    lines.append(f"\n![Feature importance]({FIG_DIR.replace('report/', '')}/feature_importance.png)\n")
    lines.append(
        "\nPermutation importance (how much accuracy drops when a feature's values are "
        "randomly shuffled) tells the same story: several measurables have *negative* "
        "importance, meaning the model does slightly **better** when that column is "
        "scrambled into noise. A feature the model is actually learning from should never "
        "hurt to keep.\n"
    )

    lines.append("\n## Do individual measurables correlate with career outcomes?\n")
    lines.append(
        f"Spearman correlation between each measurable and (a) career VORP and (b) "
        f"career PPG, across all {n} matched players:\n"
    )
    lines.append(f"\n![Correlations]({FIG_DIR.replace('report/', '')}/correlations.png)\n")
    lines.append(
        f"\n- **Vs. career value (VORP): {n_significant_vorp} of 9 measurables reach "
        f"statistical significance (p < 0.05).** The strongest is "
        f"{strongest_vorp['feature']} at r = {strongest_vorp['r']:+.3f} — a correlation "
        "so close to zero it has no practical use. Combine measurables essentially do not "
        "predict how good a player's career turns out to be.\n"
    )
    lines.append(
        f"- **Vs. scoring rate (PPG): {n_significant_ppg} of 9 measurables reach "
        f"statistical significance**, but the effect sizes are still small "
        f"(strongest: {strongest_ppg['feature']}, r = {strongest_ppg['r']:+.3f}). The "
        "sign is informative, though: height, wingspan, standing reach and lane-agility "
        "time all correlate *negatively* with PPG. This isn't measurables predicting "
        "talent — it's measurables predicting **position**. Taller, longer players are "
        "more often centers/forwards who score less per game than guards, regardless of "
        "how good they are.\n"
    )
    lines.append(f"\n![Feature by category]({FIG_DIR.replace('report/', '')}/feature_by_category.png)\n")
    lines.append(
        "\nThe box plots make the same point visually: the distribution of height, "
        "wingspan, vertical leap and sprint time is nearly identical whether a player "
        "ended up a Bust or a Star. The boxes barely move across categories.\n"
    )

    lines.append("\n## Conclusion\n")
    lines.append(
        "**No — on their own, NBA Combine measurables do not meaningfully predict "
        "whether a prospect becomes a star, a solid role player, or a bust.** Three "
        "different tuned model families cannot beat the trivial strategy of guessing "
        "the most common outcome for every player, and none of the individual "
        "measurables show a correlation with career value worth acting on. What the "
        "combine *does* capture is closer to **body type and position** than **talent "
        "or ceiling** — a longer, bigger player is more likely to be used as a big man "
        "who scores less per game, but that says nothing about whether he'll be good at "
        "that job. Draft evaluators lean on the combine for a reason (durability, "
        "positional fit, medical flags), but as a stand-alone predictor of career "
        "outcome, it has essentially no signal in this dataset.\n"
    )
    lines.append(
        "\n*Caveat:* this analysis uses only 9 physical/athletic testing measurables — "
        "the null result has now been checked against three model families and a "
        "hyperparameter search, but not against a richer feature set. It does not "
        "include college production, age, or scouting grades, any of which would likely "
        "predict career outcome far better than the combine alone.\n"
    )

    with open(REPORT_PATH, "w") as f:
        f.writelines(lines)

    print(f"Wrote {REPORT_PATH} and figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
