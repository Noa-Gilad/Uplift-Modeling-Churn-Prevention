"""
Utility functions for uplift churn modeling: EDA, feature engineering, and model helpers.

This module is intended to be imported by the uplift_churn_modeling notebook.
All logic is documented in docstrings; see the notebook for end-to-end workflow.

Data paths: not defined here. Callers (notebooks) use BASE_DIR / 'files' for data,
e.g. files/train/, files/test/, files/wellco_client_brief.txt.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    from causalml.inference.meta import BaseSRegressor, BaseTRegressor
except Exception:
    BaseSRegressor = BaseTRegressor = None  # type: ignore[misc, assignment]
try:
    from causalml.metrics import qini_score as causalml_qini_score
    from causalml.metrics import get_qini as causalml_get_qini
except Exception:
    causalml_qini_score = None  # type: ignore[misc, assignment]
    causalml_get_qini = None  # type: ignore[misc, assignment]
try:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
except Exception:
    LGBMRegressor = XGBRegressor = None  # type: ignore[misc, assignment]

try:
    import shap  # SHAP interpretability for tree-based uplift models
except Exception:
    shap = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as ST

try:
    from IPython.display import display
except ImportError:
    display = print  # noqa: A001

# ---------------------------------------------------------------------------
# Constants (defaults; notebook may override or pass as arguments)
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD: float = 0.2
EMBED_MODEL_NAME: str = "all-MiniLM-L6-v2"
FOCUS_ICD_CODES: list[str] = ["E11.9", "I10", "Z71.3"]
RANDOM_STATE: int = 42
DOW_NAMES: dict[int, str] = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------


def set_axes_clear(ax, x_axis_at_zero: bool = False) -> None:
    """Set clear axes: y-axis at x=0 (left spine). If x_axis_at_zero, x-axis is at y=0 (hide bottom spine; caller must add axhline(0)); else show bottom spine as x-axis."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if x_axis_at_zero:
        ax.spines["bottom"].set_visible(False)
    else:
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(1.2)


def print_table_overview(name: str, df: pd.DataFrame) -> None:
    """Print structure, dtypes, numeric describe, date ranges, and head for a single table.

    Parameters
    ----------
    name : str
        Display name of the table.
    df : pandas.DataFrame
        The table to summarize.

    Returns
    -------
    None
    """
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print("\n--- dtypes ---")
    print(df.dtypes.to_string())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["member_id"]]
    if len(numeric_cols) > 0:
        print("\n--- describe (numeric columns only) ---")
        print(df[numeric_cols].describe().to_string())

    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    if date_cols:
        print("\n--- date ranges ---")
        for col in date_cols:
            print(f"  {col}: {df[col].min()} to {df[col].max()}")

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        print("\n--- object columns (unique counts) ---")
        for col in object_cols:
            print(f"  {col}: {df[col].nunique()} unique values")
    print("\n--- head(2) ---")
    display(df.head(2))


def missingness_and_member_coverage(
    all_tables: dict[str, pd.DataFrame],
    churn_labels: pd.DataFrame,
    web_visits: pd.DataFrame,
    app_usage: pd.DataFrame,
    claims: pd.DataFrame,
    test_members: pd.DataFrame,
    test_web_visits: pd.DataFrame,
    test_app_usage: pd.DataFrame,
    test_claims: pd.DataFrame,
    show_plot: bool = True,
) -> None:
    """Report column-level nulls and member coverage across activity sources (web, app, claims).

    Part A: For each table in all_tables, report any column with nulls.
    Part B: For TRAIN (churn_labels) and TEST (test_members), count members with zero rows
    in each source; for TRAIN only, print cross-source missingness patterns and optionally
    show a bar chart of % absent per source.

    Parameters
    ----------
    all_tables : dict[str, pd.DataFrame]
        Map of table name -> DataFrame (all 8 tables) for Part A null check.
    churn_labels : pandas.DataFrame
        Train base set (member_id); used as TRAIN base for Part B.
    web_visits : pandas.DataFrame
        Train web events (member_id).
    app_usage : pandas.DataFrame
        Train app events (member_id).
    claims : pandas.DataFrame
        Train claims (member_id).
    test_members : pandas.DataFrame
        Test base set (member_id); used as TEST base for Part B.
    test_web_visits : pandas.DataFrame
        Test web events (member_id).
    test_app_usage : pandas.DataFrame
        Test app events (member_id).
    test_claims : pandas.DataFrame
        Test claims (member_id).
    show_plot : bool, optional
        If True, show bar chart of % members absent per source for train (default True).

    Returns
    -------
    None
    """
    # Part A: Column-level null check
    print("=" * 60)
    print("  Part A — Column-level null check")
    print("=" * 60)

    null_rows: list[dict] = []
    for name, df in all_tables.items():
        nulls = df.isnull().sum()
        for col, cnt in nulls.items():
            if cnt > 0:
                null_rows.append({"table": name, "column": col, "null_count": int(cnt)})

    if null_rows:
        null_df = pd.DataFrame(null_rows)
        print(null_df.to_string(index=False))
    else:
        print("\n✓ No null values found in any column of any table.\n")

    # Part B: Member coverage across activity sources
    print("=" * 60)
    print("  Part B — Member coverage across sources")
    print("=" * 60)

    train_src = {"web_visits": web_visits, "app_usage": app_usage, "claims": claims}
    test_src = {
        "web_visits": test_web_visits,
        "app_usage": test_app_usage,
        "claims": test_claims,
    }

    for split_name, base_df, src_tables in [
        ("TRAIN", churn_labels, train_src),
        ("TEST", test_members, test_src),
    ]:
        base_ids = set(base_df["member_id"].unique())
        n_base = len(base_ids)
        print(f"\n--- {split_name} (base members: {n_base}) ---")

        source_sets: dict[str, set] = {}
        coverage_rows: list[dict] = []
        for src_name, src_df in src_tables.items():
            present = set(src_df["member_id"].unique())
            source_sets[src_name] = present
            missing = len(base_ids - present)
            coverage_rows.append(
                {
                    "source": src_name,
                    "members_present": len(present & base_ids),
                    "members_absent": missing,
                    "absent_pct": missing / n_base * 100,
                }
            )

        cov = pd.DataFrame(coverage_rows)
        print(cov.to_string(index=False))

        # Cross-source pattern and bar chart (train only)
        if split_name == "TRAIN":
            has_web = source_sets["web_visits"] & base_ids
            has_app = source_sets["app_usage"] & base_ids
            has_claims = source_sets["claims"] & base_ids
            no_web = base_ids - has_web
            no_app = base_ids - has_app
            no_claims = base_ids - has_claims

            patterns = {
                "missing web only": len(no_web - no_app - no_claims),
                "missing app only": len(no_app - no_web - no_claims),
                "missing claims only": len(no_claims - no_web - no_app),
                "missing web+app": len(no_web & no_app - no_claims),
                "missing web+claims": len(no_web & no_claims - no_app),
                "missing app+claims": len(no_app & no_claims - no_web),
                "missing all 3": len(no_web & no_app & no_claims),
                "present in all": len(has_web & has_app & has_claims),
            }
            print("\nCross-source missingness patterns (train):")
            for pat, cnt in patterns.items():
                print(f"  {pat}: {cnt} ({cnt / n_base * 100:.2f}%)")

            if show_plot:
                fig, ax = plt.subplots(figsize=(7, 4))
                bars = ax.bar(
                    cov["source"], cov["absent_pct"], color=sns.color_palette()[:3]
                )
                ax.set_ylabel("% with zero activity")
                ax.set_xlabel("Source table")
                ax.set_title("Members absent from each activity source (train)")
                for bar, row in zip(bars, cov.itertuples()):
                    ax.annotate(
                        f"{row.members_absent}\n({row.absent_pct:.2f}%)",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
                set_axes_clear(ax, x_axis_at_zero=False)
                fig.subplots_adjust(left=0.22, right=0.96, top=0.92, bottom=0.12)
                plt.show()


def missingness_mechanism_analysis(
    churn_labels: pd.DataFrame,
    web_visits: pd.DataFrame,
    app_usage: pd.DataFrame,
    claims: pd.DataFrame,
    show_plot: bool = True,
) -> None:
    """Assess whether absence from activity sources correlates with churn/outreach (MCAR vs MAR/MNAR).

    Builds has_web / has_app / has_claims on the train base, runs Chi-square tests for
    churn and outreach by presence/absence per source, prints cross-source contingency,
    and optionally shows a grouped bar chart of churn rate (has activity vs no activity per source).

    Parameters
    ----------
    churn_labels : pandas.DataFrame
        Train labels with columns member_id, churn, outreach.
    web_visits : pandas.DataFrame
        Train web events (member_id).
    app_usage : pandas.DataFrame
        Train app events (member_id).
    claims : pandas.DataFrame
        Train claims (member_id).
    show_plot : bool, optional
        If True, show grouped bar chart of churn rate by source and presence (default True).

    Returns
    -------
    None
    """
    # Build has_X flags on train base
    train_ids = churn_labels[["member_id", "churn", "outreach"]].copy()
    web_ids = set(web_visits["member_id"].unique())
    app_ids = set(app_usage["member_id"].unique())
    claims_ids = set(claims["member_id"].unique())
    train_ids["has_web"] = train_ids["member_id"].isin(web_ids).astype(int)
    train_ids["has_app"] = train_ids["member_id"].isin(app_ids).astype(int)
    train_ids["has_claims"] = train_ids["member_id"].isin(claims_ids).astype(int)

    # Chi-square tests: churn and outreach rate by presence/absence per source
    results: list[dict] = []
    for source in ["has_web", "has_app", "has_claims"]:
        for target in ["churn", "outreach"]:
            ct = pd.crosstab(train_ids[source], train_ids[target])
            chi2, p, dof, expected = chi2_contingency(ct)
            rate_absent = train_ids.loc[train_ids[source] == 0, target].mean()
            rate_present = train_ids.loc[train_ids[source] == 1, target].mean()
            results.append(
                {
                    "source_flag": source,
                    "target": target,
                    "rate_absent (0)": round(rate_absent, 4)
                    if pd.notna(rate_absent)
                    else "N/A",
                    "rate_present (1)": round(rate_present, 4),
                    "chi2": round(chi2, 2),
                    "p_value": f"{p:.4g}",
                }
            )

    results_df = pd.DataFrame(results)
    print("=" * 70)
    print("  Chi-square tests: churn/outreach rate by presence/absence")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Cross-source contingency (train)
    print("\n" + "=" * 70)
    print("  Cross-source contingency (train)")
    print("=" * 70)
    cross = (
        train_ids.groupby(["has_web", "has_app", "has_claims"])
        .size()
        .reset_index(name="count")
    )
    print(cross.to_string(index=False))

    # Grouped bar chart: churn rate present vs absent per source
    if not show_plot:
        return

    chart_data: list[dict] = []
    for source in ["has_web", "has_app", "has_claims"]:
        for val, label in [(1, "present"), (0, "absent")]:
            subset = train_ids[train_ids[source] == val]
            if len(subset) > 0:
                chart_data.append(
                    {
                        "source": source.replace("has_", ""),
                        "group": label,
                        "churn_rate": subset["churn"].mean(),
                        "n": len(subset),
                    }
                )

    chart_df = pd.DataFrame(chart_data)
    if len(chart_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sources = chart_df["source"].unique()
    x = np.arange(len(sources))
    width = 0.35
    present = chart_df[chart_df["group"] == "present"].set_index("source")
    absent = chart_df[chart_df["group"] == "absent"].set_index("source")
    bars1 = ax.bar(
        x - width / 2,
        [present.loc[s, "churn_rate"] if s in present.index else 0 for s in sources],
        width,
        label="Has activity",
        color=sns.color_palette()[0],
    )
    bars2 = ax.bar(
        x + width / 2,
        [absent.loc[s, "churn_rate"] if s in absent.index else 0 for s in sources],
        width,
        label="No activity",
        color=sns.color_palette()[3],
    )
    ax.set_ylabel("Churn rate")
    ax.set_xlabel("Activity source")
    ax.set_title("Churn rate: members with vs without activity per source")
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    set_axes_clear(ax, x_axis_at_zero=False)
    plt.tight_layout()
    plt.show()


def _chi2_pvalue_safe(crosstab: pd.DataFrame) -> float | None:
    """Run chi-square test on a 2x2 (or larger) crosstab; return p-value or None if invalid.

    Parameters
    ----------
    crosstab : pandas.DataFrame
        Contingency table (e.g. is_null x churn).

    Returns
    -------
    float or None
        P-value from chi2_contingency, or None if table too small or has zero cells.
    """
    if crosstab.size < 4 or crosstab.min().min() < 1:
        return None
    try:
        _, p, _, _ = chi2_contingency(crosstab)
        return float(p)
    except Exception:
        return None


def missingness_mechanism_aggregated_table(
    member_agg: pd.DataFrame,
    churn_labels: pd.DataFrame,
    alpha: float = 0.05,
    print_result: bool = True,
) -> pd.DataFrame:
    """Assess missingness mechanism per column of an aggregated member table (MCAR vs MAR).

    For each column that has nulls, runs chi-square tests of "is null" vs churn and vs
    outreach. Labels mechanism as MAR when missingness is associated with that outcome,
    or MCAR when there is no significant association.

    Parameters
    ----------
    member_agg : pandas.DataFrame
        Per-member aggregate table (one row per member); must include member_id.
    churn_labels : pandas.DataFrame
        Labels with columns member_id, churn, outreach.
    alpha : float, optional
        Significance level for chi-square (default 0.05).
    print_result : bool, optional
        If True, print the results table and interpretation (default True).

    Returns
    -------
    pandas.DataFrame
        Columns: column, null_count, churn_p, outreach_p, mechanism.
    """
    agg_with_labels = member_agg.merge(
        churn_labels[["member_id", "churn", "outreach"]], on="member_id", how="inner"
    )
    results: list[dict] = []
    for col in member_agg.columns:
        null_count = member_agg[col].isna().sum()
        if null_count == 0:
            results.append({
                "column": col,
                "null_count": 0,
                "churn_p": "—",
                "outreach_p": "—",
                "mechanism": "no nulls",
            })
            continue
        is_null = agg_with_labels[col].isna().astype(int)
        p_churn = _chi2_pvalue_safe(pd.crosstab(is_null, agg_with_labels["churn"]))
        p_outreach = _chi2_pvalue_safe(pd.crosstab(is_null, agg_with_labels["outreach"]))
        sig_churn = p_churn is not None and p_churn < alpha
        sig_outreach = p_outreach is not None and p_outreach < alpha
        if sig_churn and sig_outreach:
            mechanism = "MAR (churn & outreach)"
        elif sig_churn:
            mechanism = "MAR (churn)"
        elif sig_outreach:
            mechanism = "MAR (outreach)"
        else:
            mechanism = "MCAR (no sig. association)"
        results.append({
            "column": col,
            "null_count": int(null_count),
            "churn_p": round(p_churn, 4) if p_churn is not None else "—",
            "outreach_p": round(p_outreach, 4) if p_outreach is not None else "—",
            "mechanism": mechanism,
        })
    res_df = pd.DataFrame(results).sort_values("null_count", ascending=False).reset_index(drop=True)
    if print_result:
        print("=" * 80)
        print("  Per-member table: missingness mechanism by column (null vs churn / outreach)")
        print("=" * 80)
        print(res_df.to_string(index=False))
        print("\nInterpretation: MAR = missingness associated with observed variable; MCAR = no significant association.")
    return res_df


def plot_balance(
    data: pd.DataFrame,
    x: str,
    title: str,
    xlabel: str,
    ylabel: str = "Count",
    y: str | None = None,
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """Draw a single balance plot: countplot (if y is None) or barplot (if y is set).

    Used for labels/treatment balance: outreach counts, churn counts, or churn rate by group.
    Same styling (title, axis labels, figsize, tight_layout) for consistency.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot (e.g. churn_labels or summary_labels.reset_index()).
    x : str
        Column name for the x-axis (e.g. 'outreach', 'churn').
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str, optional
        Y-axis label (default 'Count'). Use e.g. 'Churn rate' for barplot.
    y : str or None, optional
        If None, use sns.countplot(data, x=x). If set, use sns.barplot(data, x=x, y=y).
    figsize : tuple of (int, int), optional
        Figure size (default (8, 5)).

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    if y is None:
        sns.countplot(data=data, x=x)
    else:
        sns.barplot(data=data, x=x, y=y)
    ax = plt.gca()
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tick_params(labelsize=11)
    set_axes_clear(ax, x_axis_at_zero=False)
    plt.tight_layout()
    plt.show()


def feat_distribution_summary(
    data: pd.DataFrame,
    feat: str,
    bins: int = 50,
    color: str | None = None,
) -> None:
    """Print quantile summary and show a log-scaled histogram for one numeric feature.

    Reusable for distribution sanity checks: engagement (web_visits_count, app_sessions_count,
    url_nunique), claims (claims_count, icd_nunique), or any other zero-filled count-like column.
    Plots log(1 + values) to handle skew.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with at least the column named by feat (zero-filled or numeric).
    feat : str
        Column name to summarize and plot.
    bins : int, optional
        Number of histogram bins (default 50).
    color : str or None, optional
        Bar color for the histogram. If None, default matplotlib color is used.

    Returns
    -------
    None
    """
    print(f"\n{feat}:")
    print(data[feat].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]).to_string())
    fig, ax = plt.subplots(figsize=(5, 4))
    kwargs: dict = {"bins": bins, "edgecolor": "black", "alpha": 0.7}
    if color is not None:
        kwargs["color"] = color
    ax.hist(np.log1p(data[feat]), **kwargs)
    ax.set_xlabel(f"log(1 + {feat})", fontsize=11)
    ax.set_ylabel("Number of members", fontsize=11)
    ax.set_title(feat, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    set_axes_clear(ax, x_axis_at_zero=False)
    plt.tight_layout()
    plt.show()


def feature_diagnostics(
    df: pd.DataFrame,
    feature_cols: list[str],
    title_suffix: str = "",
) -> None:
    """Print feature summary statistics and zeros/missing percentages for selected columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the feature columns (e.g. train_features).
    feature_cols : list of str
        Column names to summarize.
    title_suffix : str, optional
        Suffix for section headers (e.g. '(train)').

    Returns
    -------
    None
    """
    subset = df[feature_cols]
    print("=" * 70)
    print(f"FEATURE SUMMARY STATISTICS {title_suffix}".strip())
    print("=" * 70)
    print(subset.describe().T.to_string())
    print("\n" + "=" * 70)
    print(f"ZEROS AND MISSING VALUES {title_suffix}".strip())
    print("=" * 70)
    n = len(df)
    for col in feature_cols:
        pct_zero = (df[col] == 0).sum() / n * 100
        pct_miss = df[col].isna().sum() / n * 100
        print(f"  {col:<35s}  zeros: {pct_zero:6.2f}%   missing: {pct_miss:6.2f}%")


def plot_feature_histograms(
    df: pd.DataFrame,
    feature_cols: list[str],
    xlabels: dict[str, str] | None = None,
    figsize: tuple[int, int] = (18, 8),
    bins: int = 40,
    ncols: int = 4,
    suptitle: str | None = None,
) -> None:
    """Plot a grid of histograms (one per feature) with median line and optional custom x-labels.

    Each subplot: x = feature value, y = count of members; red dashed vertical line at median.
    Unused subplots (when len(feature_cols) < nrows * ncols) are hidden.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the feature columns (e.g. train_features).
    feature_cols : list of str
        Column names to plot, in order.
    xlabels : dict of str -> str or None, optional
        Map feature column name -> x-axis label. If None, column name is used.
    figsize : tuple of (int, int), optional
        Figure size (default (18, 8)).
    bins : int, optional
        Number of histogram bins per subplot (default 40).
    ncols : int, optional
        Number of columns in the grid (default 4).
    suptitle : str or None, optional
        Figure suptitle. If None, no suptitle.

    Returns
    -------
    None
    """
    if xlabels is None:
        xlabels = {}
    n = len(feature_cols)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=bins, edgecolor="white", alpha=0.8)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabels.get(col, col), fontsize=10)
        ax.set_ylabel("Number of members", fontsize=10)
        med = data.median()
        ax.axvline(
            med, color="red", linestyle="--", linewidth=1, label=f"median={med:.1f}"
        )
        ax.legend(fontsize=7)
        set_axes_clear(ax, x_axis_at_zero=False)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    if suptitle:
        plt.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Bivariate / variable distribution EDA (KDE for numeric, bar for categorical)
# ---------------------------------------------------------------------------

def _is_numeric_series(s: pd.Series) -> bool:
    """True if series is numeric or datetime (datetime plotted as KDE of ordinal)."""
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)


def _plot_numeric_kde(series: pd.Series, ax, title: str) -> None:
    """Plot kernel density for a numeric or datetime series on the given axes."""
    data = series.dropna()
    if pd.api.types.is_datetime64_any_dtype(series):
        data = data.astype("int64")
    if data.empty or data.nunique() < 2:
        ax.text(0.5, 0.5, "Insufficient variation", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    sns.kdeplot(data=data, ax=ax, fill=True)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Density")


def _plot_categorical_bars(
    series: pd.Series,
    ax,
    title: str,
    max_labels: int,
    high_card_threshold: int,
) -> None:
    """Bar plot of value_counts(); x-axis labels vertical. High-cardinality: top N + Other, no raw names."""
    vc = series.value_counts()
    n_unique = len(vc)
    if n_unique > high_card_threshold:
        top_n = min(15, n_unique)
        top = vc.head(top_n)
        other_count = vc.iloc[top_n:].sum()
        if other_count > 0:
            top = pd.concat([top, pd.Series({"Other": other_count})])
        labels = [str(i + 1) for i in range(len(top))]
        if "Other" in top.index:
            labels[-1] = "Other"
        counts = top.values
        ax.bar(range(len(counts)), counts, edgecolor="white", alpha=0.8)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_xlabel(
            f"Top {top_n} categories (1..{top_n}) + Other — names omitted (>{high_card_threshold} levels)"
        )
    else:
        plot_vc = vc.head(max_labels)
        ax.bar(range(len(plot_vc)), plot_vc.values, edgecolor="white", alpha=0.8)
        ax.set_xticks(range(len(plot_vc)))
        ax.set_xticklabels(plot_vc.index.astype(str), rotation=90, ha="right")
        if n_unique > max_labels:
            ax.set_xlabel(f"Top {max_labels} of {n_unique} categories")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Count")


def plot_variable_distributions(
    tables: dict[str, pd.DataFrame],
    *,
    skip_cols: set[str] | None = None,
    max_cat_labels: int = 20,
    high_cardinality_threshold: int = 50,
    ncols: int = 3,
    figsize_per_plot: tuple[int, int] = (5, 4),
    omit_table_name_in_titles: bool = False,
) -> None:
    """Plot per-variable distributions: KDE for numeric/datetime, bar for categorical.

    For each table, plots one subplot per column (excluding skip_cols). Numeric and
    datetime columns use kernel density; categorical use bar plots with vertical x-axis
    labels. High-cardinality categoricals (e.g. url) are shown as top N + Other without
    raw names on the x-axis.

    Parameters
    ----------
    tables : dict of str -> pandas.DataFrame
        Map table name -> DataFrame. Columns to plot are inferred (all except skip_cols).
    skip_cols : set of str or None, optional
        Column names to exclude from every table (e.g. member_id). Default {"member_id"}.
    max_cat_labels : int, optional
        Max category labels to show on x-axis for non–high-cardinality categoricals (default 20).
    high_cardinality_threshold : int, optional
        Above this many unique values, categorical is shown as top 15 + Other only (default 50).
    ncols : int, optional
        Number of subplot columns per figure (default 3).
    figsize_per_plot : tuple of (int, int), optional
        (width, height) per subplot; total figsize = (ncols * w, nrows * h) (default (5, 4)).
    omit_table_name_in_titles : bool, optional
        If True, subplot titles show only column name and suptitle is "Variable distributions" (default False).

    Returns
    -------
    None
    """
    if skip_cols is None:
        skip_cols = {"member_id"}
    for table_name, df in tables.items():
        cols = [c for c in df.columns if c not in skip_cols]
        if not cols:
            continue
        title_prefix = "" if omit_table_name_in_titles else f"{table_name}: "
        suptitle = "Variable distributions" if omit_table_name_in_titles else f"Variable distributions — {table_name}"
        numeric_cols = [c for c in cols if _is_numeric_series(df[c])]
        cat_cols = [c for c in cols if c not in numeric_cols]
        n_plots = len(numeric_cols) + len(cat_cols)
        nrows = (n_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * figsize_per_plot[0], nrows * figsize_per_plot[1]),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).flatten()
        idx = 0
        for col in numeric_cols:
            _plot_numeric_kde(df[col], axes[idx], f"{title_prefix}{col}")
            idx += 1
        for col in cat_cols:
            _plot_categorical_bars(
                df[col], axes[idx], f"{title_prefix}{col}",
                max_labels=max_cat_labels,
                high_card_threshold=high_cardinality_threshold,
            )
            idx += 1
        for j in range(idx, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
        plt.show()


def plot_correlation_diagnostics(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.8,
    title_suffix: str = "",
    figsize: tuple[int, int] | None = None,
) -> None:
    """Plot feature correlation heatmap and print pairs with |r| >= threshold.

    Figure size and annotation font scale with number of features so the heatmap
    stays readable (no overlapping labels or cell values).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the feature columns (e.g. train_features).
    feature_cols : list of str
        Column names to include in the correlation matrix.
    threshold : float, optional
        Minimum |correlation| to flag (default 0.8).
    title_suffix : str, optional
        Suffix for the plot title (e.g. '(train set)').
    figsize : tuple of (int, int) or None, optional
        Figure size. If None, scales with number of features (default None).

    Returns
    -------
    None
    """
    # Use only rows with no NaN in any feature; then drop constant columns so .corr() has no NaN (no empty cells).
    df_clean = df[feature_cols].dropna()
    variances = df_clean.var()
    cols_nonconst = [c for c in feature_cols if variances[c] > 0]
    excluded = [c for c in feature_cols if c not in cols_nonconst]
    if excluded:
        print(f"Excluding constant columns (zero variance in complete-case subset): {excluded}")
    feature_cols_use = cols_nonconst if cols_nonconst else feature_cols
    corr = df_clean[feature_cols_use].corr()
    # For display only: treat near-zero correlations as 0 so we show "0.00" instead of "-0.00" / "0.00" (zero is zero).
    corr_display = corr.copy()
    corr_display[np.abs(corr_display) < 0.005] = 0.0
    n = len(feature_cols_use)
    if figsize is None:
        # Scale figure so each cell is large enough; minimum (10, 8).
        side = max(10, int(n * 0.55))
        figsize = (side, side)
    plt.figure(figsize=figsize)
    # Annotation font: large enough to read (especially on near-zero light cells), scales down only for very many features.
    annot_font = max(8, 13 - n // 2)
    ax = sns.heatmap(
        corr_display,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        annot_kws={"size": annot_font},
        cbar_kws={"shrink": 0.8},
    )
    title = f"Feature Correlation Matrix {title_suffix}".strip()
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    # X-axis labels vertical; y-axis horizontal.
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    set_axes_clear(ax, x_axis_at_zero=False)
    plt.tight_layout()
    plt.show()
    print(f"\nPairs with |correlation| >= {threshold}:")
    flagged: list[tuple[str, str, float]] = []
    for i in range(len(feature_cols_use)):
        for j in range(i + 1, len(feature_cols_use)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                flagged.append((feature_cols_use[i], feature_cols_use[j], r))
                print(f"  {feature_cols_use[i]}  ↔  {feature_cols_use[j]}  :  r = {r:.3f}")
    if not flagged:
        print("  (none)")


def _churn_rate_by_decile(series: pd.Series, churn: pd.Series, n_deciles: int = 10) -> pd.Series:
    """Compute mean churn (outcome) per decile of a feature. Drops NaN in series; uses qcut with duplicates='drop'.

    Parameters
    ----------
    series : pandas.Series
        Feature values.
    churn : pandas.Series
        Binary outcome (0/1), same index as series.
    n_deciles : int, optional
        Number of bins (default 10).

    Returns
    -------
    pandas.Series
        Index = decile interval labels, values = mean churn in that decile.
    """
    use = pd.DataFrame({"x": series, "y": churn}).dropna(subset=["x"])
    if use.empty:
        return pd.Series(dtype=float)
    try:
        use["_bin"] = pd.qcut(use["x"], q=n_deciles, duplicates="drop")
    except Exception:
        use["_bin"] = pd.qcut(use["x"].rank(method="first"), q=n_deciles, duplicates="drop")
    return use.groupby("_bin", observed=True)["y"].mean()


def _churn_rate_by_level(series: pd.Series, churn: pd.Series) -> pd.Series:
    """Compute mean churn per distinct level of a (typically binary) feature.

    Parameters
    ----------
    series : pandas.Series
        Feature values (e.g. 0/1).
    churn : pandas.Series
        Binary outcome (0/1), same index as series.

    Returns
    -------
    pandas.Series
        Index = feature levels, values = mean churn in that level.
    """
    use = pd.DataFrame({"x": series, "y": churn}).dropna(subset=["x"])
    if use.empty:
        return pd.Series(dtype=float)
    return use.groupby("x", observed=True)["y"].mean().sort_index()


def _rate_axis_limits(
    values: pd.Series | np.ndarray,
    min_range: float = 0.08,
    padding_frac: float = 0.15,
) -> tuple[float, float]:
    """Compute y-axis limits for rate (0-1) bar charts so small differences are visible.

    Parameters
    ----------
    values : pandas.Series or numpy array
        Rate values in [0, 1].
    min_range : float, optional
        Minimum y-axis span (default 0.08) so narrow ranges are visible.
    padding_frac : float, optional
        Fraction of data range added as padding (default 0.15).

    Returns
    -------
    tuple of (float, float)
        (y_min, y_max) clipped to [0, 1].
    """
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    r = hi - lo
    if r < 1e-9:
        mid = lo
        half = min_range / 2
        return max(0.0, mid - half), min(1.0, mid + half)
    pad = max(r * padding_frac, 0.02)
    if r < min_range:
        pad = max(pad, (min_range - r) / 2)
    y_min = max(0.0, lo - pad)
    y_max = min(1.0, hi + pad)
    if y_max - y_min < min_range:
        mid = (y_min + y_max) / 2
        y_min = max(0.0, mid - min_range / 2)
        y_max = min(1.0, mid + min_range / 2)
    return y_min, y_max


def plot_bivariate_churn_grid(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "churn",
    n_deciles: int = 10,
    figsize: tuple[float, float] = (12, 5),
) -> None:
    """Plot bivariate vs churn: one figure per feature for readability.

    For each feature: left = churn rate by decile (or by level if binary); right = box plot by churn.
    Binary features (≤2 unique values) show only the left plot (two bars). Bar plot y-axes are scaled
    to the data range (with minimum range) so small rate differences are visible.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain feature_cols and target_col (e.g. churn 0/1).
    feature_cols : list of str
        Column names to plot.
    target_col : str, optional
        Binary outcome column (default "churn").
    n_deciles : int, optional
        Number of bins for non-binary features (default 10).
    figsize : tuple of (float, float), optional
        (width, height) per single-feature figure (default (12, 5)).

    Returns
    -------
    None
    """
    # Palette for box plot: use hue=target_col so palette is applied; keys must match data dtype.
    palette = {"0": "lightsteelblue", "1": "coral"}

    for col in feature_cols:
        use = df[[col, target_col]].dropna()
        if use.empty:
            fig, (ax_dec, ax_box) = plt.subplots(1, 2, figsize=figsize)
            ax_dec.set_title(f"{col} (no data)")
            ax_box.set_title(f"{col} (no data)")
            plt.tight_layout()
            plt.show()
            continue

        n_unique = use[col].nunique()
        is_binary = n_unique <= 2

        # Binary: single plot (churn rate by level only). Non-binary: two panels (decile + box).
        if is_binary:
            fig, ax_dec = plt.subplots(1, 1, figsize=(6, figsize[1]))
            ax_box = None
        else:
            fig, (ax_dec, ax_box) = plt.subplots(1, 2, figsize=figsize)

        # Left: churn rate by decile or by level (binary); y-axis scaled to data so differences are visible.
        if is_binary:
            rate_by_level = _churn_rate_by_level(use[col], use[target_col])
            if rate_by_level.empty:
                ax_dec.set_title(f"{col} (no data)")
            else:
                labels = [str(i) for i in rate_by_level.index]
                x_pos = range(len(rate_by_level))
                ax_dec.bar(x_pos, rate_by_level.values, color="steelblue", edgecolor="black", alpha=0.85)
                ax_dec.set_xticks(x_pos)
                ax_dec.set_xticklabels(labels, fontsize=11)
                y_lo, y_hi = _rate_axis_limits(rate_by_level.values)
                ax_dec.set_ylim(y_lo, y_hi)
                ax_dec.set_ylabel("Churn rate", fontsize=10)
                ax_dec.set_title(f"{col}: churn rate by level (binary)", fontsize=11)
        else:
            rate_by_dec = _churn_rate_by_decile(use[col], use[target_col], n_deciles=n_deciles)
            if rate_by_dec.empty:
                ax_dec.set_title(f"{col} (no bins)")
            else:
                n_bars = len(rate_by_dec)
                x_pos = range(n_bars)
                ax_dec.bar(x_pos, rate_by_dec.values, color="steelblue", edgecolor="black", alpha=0.85)
                ax_dec.set_xticks(x_pos)
                ax_dec.set_xticklabels([f"D{i+1}" for i in range(n_bars)], fontsize=9)
                y_lo, y_hi = _rate_axis_limits(rate_by_dec.values)
                ax_dec.set_ylim(y_lo, y_hi)
                ax_dec.set_ylabel("Churn rate", fontsize=10)
                ax_dec.set_title(f"{col}: churn rate by decile", fontsize=11)

        # Right: box plot by churn (only for non-binary)
        if ax_box is not None:
            use_box = use.copy()
            use_box[target_col] = use_box[target_col].astype(str)
            sns.boxplot(
                data=use_box, x=target_col, y=col, hue=target_col, legend=False, ax=ax_box,
                palette=palette,
            )
            ax_box.set_title(f"{col}: distribution by churn", fontsize=11)
            ax_box.set_xlabel(target_col, fontsize=10)
            ax_box.set_ylabel(col, fontsize=10)
            fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.12, wspace=0.28)
        else:
            fig.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)
        plt.show()


def build_claims_labels(
    claims: pd.DataFrame,
    churn_labels: pd.DataFrame,
    focus_icd_codes: list[str] | None = None,
) -> pd.DataFrame:
    """Build claims-and-labels dataframe: member_id, churn, outreach, claims_count, icd_nunique, has_focus_icd, focus_icd_count.

    Aggregates claims by member (count, unique ICDs, focus-ICD flag and count), merges with
    churn_labels, and zero-fills missing values. Used for claims distribution sanity checks
    and uplift-by-claims-strata analyses.

    Parameters
    ----------
    claims : pandas.DataFrame
        Claims table with member_id and icd_code.
    churn_labels : pandas.DataFrame
        Labels with member_id, churn, outreach.
    focus_icd_codes : list of str or None, optional
        ICD codes to treat as focus (e.g. WellCo clinical focus). If None, uses FOCUS_ICD_CODES.

    Returns
    -------
    pandas.DataFrame
        One row per member: member_id, churn, outreach, claims_count, icd_nunique, has_focus_icd, focus_icd_count (all filled, no NaN).
    """
    if focus_icd_codes is None:
        focus_icd_codes = FOCUS_ICD_CODES
    claims_per = claims.groupby("member_id").size().rename("claims_count").reset_index()
    icd_nun = (
        claims.groupby("member_id")["icd_code"]
        .nunique()
        .rename("icd_nunique")
        .reset_index()
    )
    claims_f = claims.copy()
    claims_f["is_focus"] = claims_f["icd_code"].isin(focus_icd_codes)
    focus_any = (
        claims_f.groupby("member_id")["is_focus"]
        .any()
        .rename("has_focus_icd")
        .reset_index()
    )
    focus_count = (
        claims_f[claims_f["is_focus"]]
        .groupby("member_id")["icd_code"]
        .nunique()
        .rename("focus_icd_count")
        .reset_index()
    )
    cl = (
        churn_labels[["member_id", "churn", "outreach"]]
        .merge(claims_per, on="member_id", how="left")
        .merge(icd_nun, on="member_id", how="left")
        .merge(focus_any, on="member_id", how="left")
        .merge(focus_count, on="member_id", how="left")
    )
    cl["claims_count"] = cl["claims_count"].fillna(0)
    cl["icd_nunique"] = cl["icd_nunique"].fillna(0)
    cl["has_focus_icd"] = cl["has_focus_icd"].fillna(False).astype(int)
    cl["focus_icd_count"] = cl["focus_icd_count"].fillna(0).astype(int)
    return cl


def print_focus_icd_stats(cl: pd.DataFrame) -> None:
    """Print focus-ICD prevalence and focus_icd_count distribution for a claims-labels dataframe.

    Assumes cl was produced by build_claims_labels (columns has_focus_icd, focus_icd_count).

    Parameters
    ----------
    cl : pandas.DataFrame
        Claims-labels dataframe from build_claims_labels.

    Returns
    -------
    None
    """
    print(f"\nFocus-ICD prevalence: {cl['has_focus_icd'].mean():.3f}")
    print("\nFocus-ICD count distribution:")
    print(cl["focus_icd_count"].value_counts().sort_index().to_string())


def count_events_before_signup(
    events_df: pd.DataFrame,
    date_col: str,
    labels_df: pd.DataFrame,
) -> int:
    """Count event rows where event date is before the member's signup_date (potential leakage).

    Parameters
    ----------
    events_df : pandas.DataFrame
        Event table with member_id and date_col.
    date_col : str
        Name of the date column (e.g. 'timestamp' or 'diagnosis_date').
    labels_df : pandas.DataFrame
        Table with member_id and signup_date.

    Returns
    -------
    int
        Number of event rows where event date < signup_date.
    """
    merged = events_df[["member_id", date_col]].merge(
        labels_df[["member_id", "signup_date"]], on="member_id", how="left"
    )
    return (merged[date_col] < merged["signup_date"]).sum()


def time_bin(h: int) -> str:
    """Map hour (0-23) to a time-of-day label for aggregation.

    Parameters
    ----------
    h : int
        Hour of day (0-23).

    Returns
    -------
    str
        One of 'Early Morning', 'Morning', 'Afternoon', 'Evening'.
    """
    if h < 6:
        return "Early Morning"
    if h < 12:
        return "Morning"
    if h < 18:
        return "Afternoon"
    return "Evening"


def compute_uplift(
    labels_df: pd.DataFrame, member_ids: pd.Series | np.ndarray
) -> tuple[float, int, int]:
    """Return (uplift, n_treated, n_control) for a set of member IDs.

    Parameters
    ----------
    labels_df : pandas.DataFrame
        Must contain columns member_id, churn, outreach.
    member_ids : array-like
        Member IDs to compute uplift for.

    Returns
    -------
    tuple of (float, int, int)
        uplift, n_treated, n_control. uplift is np.nan if no treated or control.
    """
    df = labels_df[labels_df["member_id"].isin(member_ids)]
    tr = df[df["outreach"] == 1]["churn"]
    co = df[df["outreach"] == 0]["churn"]
    uplift = tr.mean() - co.mean() if len(tr) > 0 and len(co) > 0 else np.nan
    return float(uplift) if not np.isnan(uplift) else np.nan, len(tr), len(co)


def plot_uplift_bars(
    bin_names: list[str],
    uplifts: list[float],
    title: str,
    xlabel: str,
) -> None:
    """Simple bar plot: one bar per bin, y = uplift, horizontal zero line.

    Parameters
    ----------
    bin_names : list of str
        Labels for each bar.
    uplifts : list of float
        Uplift value per bin.
    title : str
        Plot title.
    xlabel : str
        X-axis label.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        range(len(bin_names)), uplifts, color="steelblue", edgecolor="black", alpha=0.85
    )
    # Prominent horizontal line at y=0 (x-axis) so negative uplift is clearly below it
    ax.axhline(0, color="black", linestyle="-", linewidth=1.5)
    ax.set_xticks(range(len(bin_names)))
    ax.set_xticklabels(bin_names, rotation=20, ha="right")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Uplift (churn-rate difference)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="both", which="major", length=5, width=1, labelsize=10)
    set_axes_clear(ax, x_axis_at_zero=True)
    plt.tight_layout()
    plt.show()


def uplift_by_groups(
    events: pd.DataFrame,
    labels_df: pd.DataFrame,
    group_col: str,
    group_values: list,
    plot_labels: list[str] | None = None,
    title: str = "",
    xlabel: str = "",
) -> None:
    """Compute uplift per group, plot bars, and print a summary table.

    For each value in group_values, filters events by group_col == value, gets member IDs,
    computes uplift via compute_uplift(labels_df, ids), then plots all uplifts and prints
    a table (Group, Uplift, n_treated, n_control). Use for time-of-day, day-of-week,
    weekend vs weekday, or any other categorical split.

    Parameters
    ----------
    events : pandas.DataFrame
        Event table with member_id and the column named by group_col.
    labels_df : pandas.DataFrame
        Labels with member_id, churn, outreach (e.g. churn_labels or subset).
    group_col : str
        Column in events to filter on (e.g. 'time_of_day', 'dow_name', 'is_weekend').
    group_values : list
        Ordered list of values to iterate over (e.g. tod_order, dow_order, [False, True]).
    plot_labels : list of str or None, optional
        Labels for plot and table. If None, use str(v) for each value in group_values.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.

    Returns
    -------
    None
    """
    display_labels = (
        plot_labels if plot_labels is not None else [str(v) for v in group_values]
    )
    uplifts: list[float] = []
    n_ts: list[int] = []
    n_cs: list[int] = []
    for val in group_values:
        ids = events.loc[events[group_col] == val, "member_id"].unique()
        u, nt, nc = compute_uplift(labels_df, ids)
        uplifts.append(u)
        n_ts.append(nt)
        n_cs.append(nc)
    plot_uplift_bars(display_labels, uplifts, title=title, xlabel=xlabel)
    print(f"{'Group':<25} {'Uplift':>8} {'n_treated':>10} {'n_control':>10}")
    for name, u, nt, nc in zip(display_labels, uplifts, n_ts, n_cs):
        print(f"{name:<25} {u:>8.4f} {nt:>10} {nc:>10}")


def build_recency_tenure(
    members_df: pd.DataFrame,
    web_df: pd.DataFrame,
    app_df: pd.DataFrame,
    claims_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Build member-level recency and tenure features.

    Parameters
    ----------
    members_df : DataFrame with at least member_id and signup_date.
    web_df : DataFrame with member_id and timestamp.
    app_df : DataFrame with member_id and timestamp.
    claims_df : DataFrame with member_id and diagnosis_date.

    Returns
    -------
    tuple of (DataFrame, pd.Timestamp)
        DataFrame indexed by member_id with columns:
        days_since_last_web, days_since_last_app, days_since_last_claim,
        days_since_last_activity, tenure_days, recent_any_7d.
        Second element is the reference date used.
    """
    ref_date = max(
        pd.to_datetime(web_df["timestamp"], errors="coerce").dropna().max(),
        pd.to_datetime(app_df["timestamp"], errors="coerce").dropna().max(),
        pd.to_datetime(claims_df["diagnosis_date"], errors="coerce").dropna().max(),
    )
    last_web = web_df.groupby("member_id")["timestamp"].max()
    last_app = app_df.groupby("member_id")["timestamp"].max()
    last_claim = claims_df.groupby("member_id")["diagnosis_date"].max()

    out = members_df[["member_id", "signup_date"]].copy()
    out["days_since_last_web"] = (
        out["member_id"].map(last_web).pipe(lambda s: (ref_date - s).dt.days)
    )
    out["days_since_last_app"] = (
        out["member_id"].map(last_app).pipe(lambda s: (ref_date - s).dt.days)
    )
    out["days_since_last_claim"] = (
        out["member_id"].map(last_claim).pipe(lambda s: (ref_date - s).dt.days)
    )
    last_any = (
        pd.concat(
            [last_web.rename("ts"), last_app.rename("ts"), last_claim.rename("ts")]
        )
        .groupby(level=0)
        .max()
    )
    out["days_since_last_activity"] = (
        out["member_id"].map(last_any).pipe(lambda s: (ref_date - s).dt.days)
    )
    out["tenure_days"] = (
        ref_date - pd.to_datetime(out["signup_date"], errors="coerce")
    ).dt.days
    out["recent_any_7d"] = (
        out["days_since_last_activity"].notna() & (out["days_since_last_activity"] <= 7)
    ).astype(int)
    out = out.set_index("member_id").drop(columns=["signup_date"])
    return out, ref_date


def build_member_aggregates(
    members_df: pd.DataFrame,
    app_df: pd.DataFrame,
    web_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    ref_date: pd.Timestamp | None = None,
    focus_icd_codes: list[str] | None = None,
    *,
    wellco_embedding: np.ndarray | None = None,
    embed_model: "ST" | None = None,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> pd.DataFrame:
    """Build per-member aggregates for EDA: all-visit and WellCo-relevant web, app, claims, tenure.

    One row per member from members_df. Count columns (n_sessions, n_web_visits, n_claims, etc.)
    are set to 0 when the member has no rows in the source table (no event = zero count, not missing).
    Only days_since_* and similar recency columns can be NaN when there is no event (true missing).
    Includes days_since_signup; when wellco_embedding and embed_model are provided, also filters
    web visits by WellCo relevance and adds wellco_web_visits_count and days_since_last_wellco_web.

    Parameters
    ----------
    members_df : pd.DataFrame
        Must have member_id; signup_date required for days_since_signup.
    app_df, web_df, claims_df : pd.DataFrame
        Raw event tables (app: member_id, timestamp; web: member_id, timestamp, title, description; claims: member_id, diagnosis_date, icd_code).
    ref_date : pd.Timestamp or None, optional
        Reference date for recency. If None, derived from max of event tables.
    focus_icd_codes : list of str or None, optional
        ICD codes to count as focus. If None, uses FOCUS_ICD_CODES.
    wellco_embedding : np.ndarray or None, optional
        Pre-computed WellCo brief embedding (1, dim). If None, WellCo web columns are omitted.
    embed_model : SentenceTransformer or None, optional
        Pre-loaded embedding model. Required with wellco_embedding for WellCo web features.
    similarity_threshold : float, optional
        Cosine-similarity cutoff for WellCo-relevant visits (default SIMILARITY_THRESHOLD).

    Returns
    -------
    pd.DataFrame
        One row per member with columns: member_id, days_since_signup, n_sessions,
        has_app_usage, days_since_last_session, n_web_visits, has_web_visits,
        days_since_last_visit, n_distinct_titles, wellco_web_visits_count,
        days_since_last_wellco_web (if WellCo args provided), n_claims, icd_distinct_count,
        has_claims, days_since_last_claim, n_focus_icd_claims, has_focus_icd.
    """
    if ref_date is None:
        ref_date = ref_date_from_tables(web_df, app_df, claims_df)
    if focus_icd_codes is None:
        focus_icd_codes = FOCUS_ICD_CODES

    members = members_df[["member_id"]].drop_duplicates()

    # Days since signup (tenure)
    if "signup_date" in members_df.columns:
        signup = members_df[["member_id", "signup_date"]].drop_duplicates("member_id")
        members = members.merge(signup, on="member_id", how="left")
        members["days_since_signup"] = (
            ref_date - pd.to_datetime(members["signup_date"], errors="coerce")
        ).dt.days
        members = members.drop(columns=["signup_date"])
    else:
        members["days_since_signup"] = np.nan

    # App: count, has_any, last timestamp
    app_agg = app_df.groupby("member_id").agg(
        n_sessions=("timestamp", "count"),
        last_app=("timestamp", "max"),
    )
    members = members.merge(app_agg, on="member_id", how="left")
    # No row in app_df means count = 0, not missing (unlike days_since_*, which is truly unknown).
    members["n_sessions"] = members["n_sessions"].fillna(0).astype(int)
    members["has_app_usage"] = members["n_sessions"].gt(0).astype(int)
    members["days_since_last_session"] = (
        ref_date - members["last_app"]
    ).dt.days.where(members["last_app"].notna())
    members = members.drop(columns=["last_app"])

    # Web (all visits): count, has_any, last timestamp, n_distinct_titles
    web_agg = web_df.groupby("member_id").agg(
        n_web_visits=("timestamp", "count"),
        last_web=("timestamp", "max"),
    )
    if "title" in web_df.columns:
        web_agg["n_distinct_titles"] = web_df.groupby("member_id")["title"].nunique()
    members = members.merge(web_agg, on="member_id", how="left")
    # No row in web_df means count = 0, not missing.
    members["n_web_visits"] = members["n_web_visits"].fillna(0).astype(int)
    members["has_web_visits"] = members["n_web_visits"].gt(0).astype(int)
    members["days_since_last_visit"] = (
        ref_date - members["last_web"]
    ).dt.days.where(members["last_web"].notna())
    members = members.drop(columns=["last_web"])
    if "n_distinct_titles" not in members.columns:
        members["n_distinct_titles"] = 0
    else:
        members["n_distinct_titles"] = members["n_distinct_titles"].fillna(0).astype(int)

    # WellCo-relevant web: embed + filter, then wellco_web_visits_count, days_since_last_wellco_web
    if wellco_embedding is not None and embed_model is not None:
        web_relevant = filter_wellco_relevant_visits(
            web_df,
            wellco_embedding=wellco_embedding,
            embed_model=embed_model,
            similarity_threshold=similarity_threshold,
            ref_date=ref_date,
        )
        if web_relevant.empty:
            members["wellco_web_visits_count"] = 0
            members["days_since_last_wellco_web"] = np.nan
        else:
            wdf = web_relevant[web_relevant["timestamp"] <= ref_date]
            if wdf.empty:
                members["wellco_web_visits_count"] = 0
                members["days_since_last_wellco_web"] = np.nan
            else:
                wellco_agg = (
                    wdf.groupby("member_id")
                    .agg(
                        wellco_web_visits_count=("timestamp", "count"),
                        _last_wellco=("timestamp", "max"),
                    )
                    .reset_index()
                )
                wellco_agg["days_since_last_wellco_web"] = (
                    ref_date - wellco_agg["_last_wellco"]
                ).dt.days
                wellco_agg = wellco_agg.drop(columns=["_last_wellco"])
                members = members.merge(wellco_agg, on="member_id", how="left")
                members["wellco_web_visits_count"] = (
                    members["wellco_web_visits_count"].fillna(0).astype(int)
                )

    # Claims: count, icd_distinct_count, has_any, last diagnosis_date, focus ICD counts
    claims_agg = claims_df.groupby("member_id").agg(
        n_claims=("diagnosis_date", "count"),
        last_claim=("diagnosis_date", "max"),
    )
    if "icd_code" in claims_df.columns:
        claims_agg["icd_distinct_count"] = claims_df.groupby("member_id")["icd_code"].nunique()
        focus = claims_df[claims_df["icd_code"].isin(focus_icd_codes)]
        focus_agg = focus.groupby("member_id").size().rename("n_focus_icd_claims")
        claims_agg = claims_agg.join(focus_agg, how="left")
        claims_agg["n_focus_icd_claims"] = claims_agg["n_focus_icd_claims"].fillna(0).astype(int)
        claims_agg["has_focus_icd"] = claims_agg["n_focus_icd_claims"].gt(0).astype(int)
    members = members.merge(claims_agg, on="member_id", how="left")
    # No row in claims_df means count = 0, not missing.
    members["n_claims"] = members["n_claims"].fillna(0).astype(int)
    members["has_claims"] = members["n_claims"].gt(0).astype(int)
    members["days_since_last_claim"] = (
        ref_date - members["last_claim"]
    ).dt.days.where(members["last_claim"].notna())
    members = members.drop(columns=["last_claim"])
    if "n_focus_icd_claims" not in members.columns:
        members["n_focus_icd_claims"] = 0
        members["has_focus_icd"] = 0
    else:
        members["n_focus_icd_claims"] = members["n_focus_icd_claims"].fillna(0).astype(int)
        members["has_focus_icd"] = members["n_focus_icd_claims"].gt(0).astype(int)
    if "icd_distinct_count" not in members.columns:
        members["icd_distinct_count"] = 0
    else:
        members["icd_distinct_count"] = members["icd_distinct_count"].fillna(0).astype(int)

    return members


def verify_member_aggregates(
    member_agg: pd.DataFrame,
    members_df: pd.DataFrame,
    app_df: pd.DataFrame,
    web_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    log_path: Path | str | None = None,
) -> dict:
    """Verify build_member_aggregates: one row per member, counts match raw tables; count cols are 0 when absent.

    Writes NDJSON to log_path if provided. Returns a dict of check results (ok: bool, details).

    Parameters
    ----------
    member_agg : pd.DataFrame
        Output of build_member_aggregates (one row per member).
    members_df, app_df, web_df, claims_df : pd.DataFrame
        Inputs used to build member_agg (e.g. churn_labels, app_usage, web_visits, claims).
    log_path : Path or str or None, optional
        If set, append one NDJSON line with verification results.

    Returns
    -------
    dict
        Keys: row_count_ok, app_ok, web_ok, claims_ok, app_detail, web_detail, claims_detail, message.
    """
    import json as _json
    base_ids = set(members_df["member_id"].unique())
    agg_ids = set(member_agg["member_id"].unique())
    row_count_ok = len(member_agg) == len(base_ids) and agg_ids == base_ids
    app_counts = app_df.groupby("member_id").size().rename("_cnt")
    web_counts = web_df.groupby("member_id").size().rename("_cnt")
    claims_counts = claims_df.groupby("member_id").size().rename("_cnt")
    # Count columns are 0 when member absent; when present, count must match raw table.
    ma_app = member_agg.set_index("member_id")[["n_sessions"]].join(app_counts, how="left")
    app_match = ((ma_app["_cnt"].isna() & (ma_app["n_sessions"] == 0)) | (ma_app["n_sessions"] == ma_app["_cnt"])).all()
    app_zero_ok = ((~member_agg["member_id"].isin(app_df["member_id"])) & (member_agg["n_sessions"] != 0)).sum() == 0
    app_ok = bool(app_match and app_zero_ok)
    ma_web = member_agg.set_index("member_id")[["n_web_visits"]].join(web_counts, how="left")
    web_match = ((ma_web["_cnt"].isna() & (ma_web["n_web_visits"] == 0)) | (ma_web["n_web_visits"] == ma_web["_cnt"])).all()
    web_zero_ok = ((~member_agg["member_id"].isin(web_df["member_id"])) & (member_agg["n_web_visits"] != 0)).sum() == 0
    web_ok = bool(web_match and web_zero_ok)
    ma_claims = member_agg.set_index("member_id")[["n_claims"]].join(claims_counts, how="left")
    claims_match = ((ma_claims["_cnt"].isna() & (ma_claims["n_claims"] == 0)) | (ma_claims["n_claims"] == ma_claims["_cnt"])).all()
    claims_zero_ok = ((~member_agg["member_id"].isin(claims_df["member_id"])) & (member_agg["n_claims"] != 0)).sum() == 0
    claims_ok = bool(claims_match and claims_zero_ok)
    all_ok = row_count_ok and app_ok and web_ok and claims_ok
    details = {
        "row_count_ok": row_count_ok,
        "app_ok": app_ok,
        "web_ok": web_ok,
        "claims_ok": claims_ok,
        "app_match": bool(app_match),
        "app_zero_ok": bool(app_zero_ok),
        "web_match": bool(web_match),
        "web_zero_ok": bool(web_zero_ok),
        "claims_match": bool(claims_match),
        "claims_zero_ok": bool(claims_zero_ok),
        "n_members": len(member_agg),
        "n_app_rows": len(app_df),
        "n_web_rows": len(web_df),
        "n_claims_rows": len(claims_df),
    }
    if log_path is not None:
        try:
            _payload = {"all_ok": all_ok, **details}
            with open(log_path, "a", encoding="utf-8") as _f:
                _f.write(_json.dumps(_payload, default=str) + "\n")
        except Exception:
            pass
    return {"ok": all_ok, "details": details, "message": "OK" if all_ok else "One or more checks failed"}


# ---------------------------------------------------------------------------
# Feature engineering pipeline (per-member aggregates)
# ---------------------------------------------------------------------------

# Default 8 features used for modeling; recency columns use fixed bins 0 | 1-7 | >7.
MEMBER_AGG_FEATURE_COLS = [
    "days_since_signup",
    "n_sessions",
    "days_since_last_session",
    "wellco_web_visits_count",
    "days_since_last_wellco_web",
    "n_claims",
    "has_focus_icd",
    "days_since_last_claim",
]
RECENCY_BIN_EDGES = (-0.1, 0.0, 7.0, np.inf)  # bins: 0 | 1-7 | >7
RECENCY_BIN_LABELS = ("0", "1-7", ">7")
# Descriptive suffixes for one-hot recency columns: 0 -> today, 1-7 -> last_week, >7 -> older.
RECENCY_BIN_SUFFIXES = ("_today", "_last_week", "_older")
# Base name per recency column so each variable's 3 binaries are clearly named (e.g. wellco_web_today).
RECENCY_COL_BASES = {
    "days_since_last_session": "session",
    "days_since_last_wellco_web": "wellco_web",
    "days_since_last_claim": "claim",
}
RECENCY_COLS = ("days_since_last_session", "days_since_last_wellco_web", "days_since_last_claim")


def _apply_minmax(agg: pd.DataFrame, cols: list[str], reference: pd.DataFrame) -> pd.DataFrame:
    """Apply Min-Max scaling using reference table for min/max. Returns scaled values."""
    scaler = MinMaxScaler()
    scaler.fit(reference[cols])
    return pd.DataFrame(scaler.transform(agg[cols]), columns=cols, index=agg.index)


def _apply_standard(agg: pd.DataFrame, col: str, reference: pd.DataFrame) -> pd.Series:
    """Apply standardization (Z-score) using reference table for mean/std."""
    scaler = StandardScaler()
    scaler.fit(reference[[col]])
    return pd.Series(scaler.transform(agg[[col]]).ravel(), index=agg.index, name=col)


def _apply_recency_binning(series: pd.Series) -> pd.Series:
    """Bin recency column into 0 | 1-7 | >7; NaN -> '>7'."""
    binned = pd.cut(
        series.fillna(np.inf),
        bins=list(RECENCY_BIN_EDGES),
        labels=list(RECENCY_BIN_LABELS),
        include_lowest=True,
    )
    return binned.astype(str).replace("nan", ">7")


def engineer_member_aggregates(
    agg: pd.DataFrame,
    reference_agg: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    out_path: Path | str | None = None,
) -> pd.DataFrame:
    """Engineer an aggregated table (member_id + 8 input features): scaling, log, recency one-hot.

    Recency columns (days_since_last_*) are binned into 0 | 1-7 | >7 and one-hot encoded
    into 3 binary columns each with descriptive names (e.g. wellco_web_today,
    wellco_web_last_week, wellco_web_older; session_*, claim_*). Output has 14 engineered
    feature columns (5 numeric + 9 recency binaries + has_focus_icd).

    Call with reference_agg=None for train (scaling params from agg); call with
    reference_agg=train_agg for test so scaling uses train statistics. Optionally
    saves the result to out_path.

    Parameters
    ----------
    agg : pandas.DataFrame
        Per-member aggregate with member_id and the 8 feature columns (caller filters).
    reference_agg : pandas.DataFrame or None, optional
        If None, use agg for scaling params (train). If provided, use for scaling (test).
    feature_cols : list of str or None, optional
        Feature columns to engineer; default MEMBER_AGG_FEATURE_COLS.
    out_path : path-like or None, optional
        If set, save engineered table to CSV here.

    Returns
    -------
    pandas.DataFrame
        member_id plus engineered feature columns (14 feature columns after one-hot recency).
    """
    cols = feature_cols if feature_cols is not None else list(MEMBER_AGG_FEATURE_COLS)
    ref = reference_agg if reference_agg is not None else agg
    ref = ref[[c for c in cols if c in ref.columns]]
    agg = agg[["member_id"] + [c for c in cols if c in agg.columns]].copy()
    out = agg[["member_id"]].copy()

    # Min-Max: days_since_signup, n_claims
    minmax_cols = [c for c in ["days_since_signup", "n_claims"] if c in cols and c in agg.columns]
    if minmax_cols:
        out[minmax_cols] = _apply_minmax(agg, minmax_cols, ref)

    # Standardization: n_sessions
    if "n_sessions" in cols and "n_sessions" in agg.columns:
        out["n_sessions"] = _apply_standard(agg, "n_sessions", ref)

    # Log: wellco_web_visits_count
    if "wellco_web_visits_count" in cols and "wellco_web_visits_count" in agg.columns:
        out["wellco_web_visits_count"] = np.log1p(agg["wellco_web_visits_count"])

    # Recency binning: 0 | 1-7 | >7, then one-hot encode into 3 binary columns per recency feature.
    for col in RECENCY_COLS:
        if col in cols and col in agg.columns:
            binned = _apply_recency_binning(agg[col])
            base = RECENCY_COL_BASES[col]
            for label, suffix in zip(RECENCY_BIN_LABELS, RECENCY_BIN_SUFFIXES, strict=True):
                out[base + suffix] = (binned == label).astype(int)

    # Keep as-is: has_focus_icd
    if "has_focus_icd" in cols and "has_focus_icd" in agg.columns:
        out["has_focus_icd"] = agg["has_focus_icd"].values

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)

    return out


def load_wellco_brief(path: Path | str) -> str:
    """Read the WellCo client brief from disk and return it as a single string.

    Caller typically passes e.g. (BASE_DIR / 'files' / 'wellco_client_brief.txt').

    Parameters
    ----------
    path : Path or str
        File path to the brief text file.

    Returns
    -------
    str
        Full text content of the brief.
    """
    return Path(path).read_text(encoding="utf-8")


def ref_date_from_tables(*dfs: pd.DataFrame) -> pd.Timestamp:
    """Derive a per-dataset reference date from the latest timestamp in event tables.

    Parameters
    ----------
    *dfs : pd.DataFrame
        One or more event DataFrames with 'timestamp' and/or 'diagnosis_date'.

    Returns
    -------
    pd.Timestamp
        The maximum observed timestamp across all supplied tables.
    """
    max_dates: list[pd.Timestamp] = []
    for df in dfs:
        if "timestamp" in df.columns:
            max_dates.append(df["timestamp"].max())
        if "diagnosis_date" in df.columns:
            max_dates.append(df["diagnosis_date"].max())
    if not max_dates:
        raise ValueError(
            "None of the supplied DataFrames contain 'timestamp' or 'diagnosis_date'."
        )
    return max(max_dates)


def embed_wellco_brief(brief_text: str, model: "ST") -> np.ndarray:
    """Embed the WellCo client brief into a single vector. Call once at startup.

    Parameters
    ----------
    brief_text : str
        Full text of the WellCo client brief.
    model : SentenceTransformer
        Pre-loaded sentence-transformers model.

    Returns
    -------
    np.ndarray
        Shape (1, embedding_dim) — the brief's embedding vector.
    """
    return model.encode([brief_text])


def embed_visit_texts(web_df: pd.DataFrame, model: "ST") -> np.ndarray:
    """Embed the concatenated title + description of each web visit (de-duplicated for speed).

    Parameters
    ----------
    web_df : pd.DataFrame
        Must contain title and description columns.
    model : SentenceTransformer
        Pre-loaded sentence-transformers model.

    Returns
    -------
    np.ndarray
        Shape (len(web_df), embedding_dim) — one embedding per visit.
    """
    texts = (
        web_df["title"].fillna("") + " " + web_df["description"].fillna("")
    ).str.strip()
    codes, uniques = pd.factorize(texts)
    unique_embeddings = model.encode(uniques.tolist())
    print(
        f"  embed_visit_texts: {len(uniques):,} unique texts embedded "
        f"(from {len(texts):,} rows)"
    )
    return unique_embeddings[codes]


def filter_wellco_relevant_visits(
    web_df: pd.DataFrame,
    wellco_embedding: np.ndarray,
    embed_model: "ST",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    ref_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return only web visits that are semantically relevant to the WellCo brief.

    Parameters
    ----------
    web_df : pd.DataFrame
        Raw web-visits with member_id, timestamp, title, description, url.
    wellco_embedding : np.ndarray
        Pre-computed brief embedding, shape (1, dim).
    embed_model : SentenceTransformer
        Pre-loaded model.
    similarity_threshold : float
        Minimum cosine similarity to retain a visit.
    ref_date : pd.Timestamp or None
        If provided, only visits with timestamp <= ref_date are considered.

    Returns
    -------
    pd.DataFrame
        Subset of web_df containing only WellCo-relevant visits.
    """
    df = web_df.copy()
    if ref_date is not None:
        df = df[df["timestamp"] <= ref_date]
    if df.empty:
        return df
    visit_embeddings = embed_visit_texts(df, embed_model)
    similarities = cosine_similarity(visit_embeddings, wellco_embedding).flatten()
    mask = similarities >= similarity_threshold
    relevant = df[mask].copy()
    print(
        f"Web relevance filter: {mask.sum():,} / {len(mask):,} visits retained "
        f"(threshold={similarity_threshold})"
    )
    return relevant


def run_relevance_filter_sanity_check(
    wellco_embedding: np.ndarray,
    embed_model: "ST",
    similarity_threshold: float | None = None,
    ref_date: pd.Timestamp | None = None,
) -> None:
    """Run a sanity check that the WellCo relevance filter behaves as expected on a fixture.

    Uses a small fixture of web visits with known relevant/irrelevant titles and asserts
    that filter_wellco_relevant_visits retains the correct counts per member and does
    not let through non-relevant content. Raises AssertionError if any check fails.

    Parameters
    ----------
    wellco_embedding : np.ndarray
        Pre-computed WellCo brief embedding, shape (1, dim).
    embed_model : SentenceTransformer
        Pre-loaded embedding model.
    similarity_threshold : float or None
        Threshold for cosine similarity; if None, uses SIMILARITY_THRESHOLD.
    ref_date : pd.Timestamp or None
        Reference date for filtering visits; if None, uses 2025-08-15.

    Returns
    -------
    None
    """
    if similarity_threshold is None:
        similarity_threshold = SIMILARITY_THRESHOLD
    if ref_date is None:
        ref_date = pd.Timestamp("2025-08-15")

    # Ground-truth grouping of the 26 unique (title, description) pairs in the
    # web-visits data, classified by WellCo brief alignment.
    wellco_relevant_titles: set[str] = {
        "Diabetes management",
        "Hypertension basics",
        "Stress reduction",
        "Restorative sleep tips",
        "Healthy eating guide",
        "Aerobic exercise",
        "HbA1c targets",
        "Strength training basics",
        "Lowering blood pressure",
        "Sleep hygiene",
        "Mediterranean diet",
        "Cardio workouts",
        "Exercise routines",
        "Meditation guide",
        "Cardiometabolic health",
        "High-fiber meals",
        "Cholesterol friendly foods",
        "Weight management",
    }
    not_relevant_titles: set[str] = {
        "Gadget roundup",
        "Game reviews",
        "New releases",
        "Dog training",
        "Electric vehicles",
        "Budget planning",
        "Match highlights",
        "Top destinations",
    }
    assert len(wellco_relevant_titles) + len(not_relevant_titles) == 26, (
        "Expected 26 unique titles total"
    )

    # Fixture: different members and mix of titles to verify threshold generalizes.
    test_rows = [
        (
            10,
            "https://x.com/1",
            "Stress reduction",
            "Mindfulness and wellness",
            "2025-08-01 10:00:00",
        ),
        (
            10,
            "https://x.com/2",
            "Healthy eating guide",
            "Nutrition and balanced diet",
            "2025-08-02 11:00:00",
        ),
        (
            10,
            "https://x.com/3",
            "Gadget roundup",
            "Smartphones and laptops news",
            "2025-08-03 12:00:00",
        ),
        (
            11,
            "https://y.com/1",
            "Cardio workouts",
            "Exercise and recovery",
            "2025-08-04 09:00:00",
        ),
        (
            11,
            "https://y.com/2",
            "Meditation guide",
            "Mindfulness and relaxation",
            "2025-08-05 10:00:00",
        ),
        (
            11,
            "https://y.com/3",
            "Aerobic exercise",
            "Cardio and endurance",
            "2025-08-06 11:00:00",
        ),
        (
            11,
            "https://y.com/4",
            "New releases",
            "Box office and trailers",
            "2025-08-07 14:00:00",
        ),
        (
            12,
            "https://z.com/1",
            "Match highlights",
            "League standings and transfers",
            "2025-08-08 08:00:00",
        ),
        (
            12,
            "https://z.com/2",
            "Top destinations",
            "City guides and itineraries",
            "2025-08-09 16:00:00",
        ),
    ]
    test_web = pd.DataFrame(
        test_rows,
        columns=["member_id", "url", "title", "description", "timestamp"],
    )
    test_web["timestamp"] = pd.to_datetime(test_web["timestamp"])

    filtered = filter_wellco_relevant_visits(
        test_web,
        wellco_embedding=wellco_embedding,
        embed_model=embed_model,
        similarity_threshold=similarity_threshold,
        ref_date=ref_date,
    )

    counts = filtered.groupby("member_id").size()
    unique_urls = filtered.groupby("member_id")["url"].nunique()

    assert counts.get(10, 0) == 2, (
        f"Member 10: expected 2 relevant visits, got {counts.get(10, 0)}"
    )
    assert counts.get(11, 0) == 3, (
        f"Member 11: expected 3 relevant visits, got {counts.get(11, 0)}"
    )
    assert counts.get(12, 0) == 0, (
        f"Member 12: expected 0 relevant visits, got {counts.get(12, 0)}"
    )
    assert unique_urls.get(10, 0) == 2, (
        f"Member 10: expected 2 unique URLs, got {unique_urls.get(10, 0)}"
    )
    assert unique_urls.get(11, 0) == 3, (
        f"Member 11: expected 3 unique URLs, got {unique_urls.get(11, 0)}"
    )

    assert filtered["title"].isin(not_relevant_titles).sum() == 0, (
        "Filter let through visits with non-relevant titles!"
    )
    assert filtered["title"].isin(wellco_relevant_titles).all(), (
        "Filter retained titles outside the expected relevant set!"
    )

    print("✓ All relevance-filter sanity checks passed.")
    print(
        f"  Member 10: {counts.get(10, 0)} visits, {unique_urls.get(10, 0)} unique URLs"
    )
    print(
        f"  Member 11: {counts.get(11, 0)} visits, {unique_urls.get(11, 0)} unique URLs"
    )
    print(f"  Member 12: {counts.get(12, 0)} visits (correctly excluded)")


def agg_web_features(
    web_relevant_df: pd.DataFrame,
    members_df: pd.DataFrame,
    ref_date: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate WellCo-relevant web visits into member-level features.

    Returns
    -------
    pd.DataFrame
        Columns: member_id, wellco_web_visits_count, days_since_last_wellco_web.
    """
    wdf = web_relevant_df[web_relevant_df["timestamp"] <= ref_date].copy()
    if wdf.empty:
        out = members_df[["member_id"]].copy()
        out["wellco_web_visits_count"] = 0
        out["days_since_last_wellco_web"] = np.nan
        return out
    agg = (
        wdf.groupby("member_id")
        .agg(
            wellco_web_visits_count=("timestamp", "count"),
            _last_visit=("timestamp", "max"),
        )
        .reset_index()
    )
    agg["days_since_last_wellco_web"] = (ref_date - agg["_last_visit"]).dt.days
    agg.drop(columns="_last_visit", inplace=True)
    out = members_df[["member_id"]].merge(agg, on="member_id", how="left")
    out["wellco_web_visits_count"] = (
        out["wellco_web_visits_count"].fillna(0).astype(int)
    )
    return out


def agg_app_features(
    app_df: pd.DataFrame,
    members_df: pd.DataFrame,
    ref_date: pd.Timestamp,
) -> pd.DataFrame:
    """Count app sessions per member and days since last app session up to the reference date.

    Returns
    -------
    pd.DataFrame
        Columns: member_id, app_sessions_count, days_since_last_app.
    """
    adf = app_df[app_df["timestamp"] <= ref_date].copy()
    counts = adf.groupby("member_id").size().rename("app_sessions_count").reset_index()
    last_app = adf.groupby("member_id")["timestamp"].max().reset_index()
    last_app["days_since_last_app"] = (ref_date - last_app["timestamp"]).dt.days
    last_app = last_app[["member_id", "days_since_last_app"]]
    out = members_df[["member_id"]].merge(counts, on="member_id", how="left")
    out = out.merge(last_app, on="member_id", how="left")
    out["app_sessions_count"] = out["app_sessions_count"].fillna(0).astype(int)
    return out


def agg_claims_features(
    claims_df: pd.DataFrame,
    members_df: pd.DataFrame,
    ref_date: pd.Timestamp,
    focus_icd_codes: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate claims into member-level diagnostic features.

    Returns
    -------
    pd.DataFrame
        Columns: member_id, claims_count, icd_distinct_count, has_focus_icd, days_since_last_claim.
    """
    if focus_icd_codes is None:
        focus_icd_codes = FOCUS_ICD_CODES
    cdf = claims_df[claims_df["diagnosis_date"] <= ref_date].copy()
    if cdf.empty:
        out = members_df[["member_id"]].copy()
        out["claims_count"] = 0
        out["icd_distinct_count"] = 0
        out["has_focus_icd"] = 0
        out["days_since_last_claim"] = np.nan
        return out
    cdf["_is_focus"] = cdf["icd_code"].isin(focus_icd_codes)
    agg = (
        cdf.groupby("member_id")
        .agg(
            claims_count=("diagnosis_date", "count"),
            icd_distinct_count=("icd_code", "nunique"),
            has_focus_icd=("_is_focus", "any"),
            _last_claim=("diagnosis_date", "max"),
        )
        .reset_index()
    )
    agg["has_focus_icd"] = agg["has_focus_icd"].astype(int)
    agg["days_since_last_claim"] = (ref_date - agg["_last_claim"]).dt.days
    agg.drop(columns="_last_claim", inplace=True)
    out = members_df[["member_id"]].merge(agg, on="member_id", how="left")
    out["claims_count"] = out["claims_count"].fillna(0).astype(int)
    out["icd_distinct_count"] = out["icd_distinct_count"].fillna(0).astype(int)
    out["has_focus_icd"] = out["has_focus_icd"].fillna(0).astype(int)
    return out


def agg_lifecycle_tenure(
    members_df: pd.DataFrame, ref_date: pd.Timestamp
) -> pd.DataFrame:
    """Compute membership tenure in days as of the reference date.

    Returns
    -------
    pd.DataFrame
        Columns: member_id, tenure_days.
    """
    out = members_df[["member_id"]].copy()
    out["tenure_days"] = (ref_date - members_df["signup_date"]).dt.days
    return out


def build_feature_matrix(
    members_df: pd.DataFrame,
    web_df: pd.DataFrame,
    app_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    ref_date: pd.Timestamp,
    *,
    wellco_embedding: np.ndarray,
    embed_model: "ST",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    focus_icd_codes: list[str] | None = None,
    include_labels: bool = False,
) -> pd.DataFrame:
    """Build the full member-level feature matrix from raw event tables.

    Parameters
    ----------
    members_df : pd.DataFrame
        Member roster with member_id, signup_date; optionally outreach, churn.
    web_df, app_df, claims_df : pd.DataFrame
        Raw event tables.
    ref_date : pd.Timestamp
        Decision / reference date.
    wellco_embedding : np.ndarray
        Pre-computed WellCo brief embedding (1, dim).
    embed_model : SentenceTransformer
        Pre-loaded model.
    similarity_threshold : float
        Cosine-similarity cutoff for web relevance.
    focus_icd_codes : list[str] or None
        ICD-10 focus codes (default: FOCUS_ICD_CODES).
    include_labels : bool
        If True and members_df has outreach/churn, append those columns.

    Returns
    -------
    pd.DataFrame
        One row per member; feature columns + optionally outreach, churn.
    """
    if focus_icd_codes is None:
        focus_icd_codes = FOCUS_ICD_CODES
    web_relevant = filter_wellco_relevant_visits(
        web_df,
        wellco_embedding=wellco_embedding,
        embed_model=embed_model,
        similarity_threshold=similarity_threshold,
        ref_date=ref_date,
    )
    feat_web = agg_web_features(web_relevant, members_df, ref_date)
    feat_app = agg_app_features(app_df, members_df, ref_date)
    feat_claims = agg_claims_features(claims_df, members_df, ref_date, focus_icd_codes)
    feat_life = agg_lifecycle_tenure(members_df, ref_date)
    feature_matrix = (
        members_df[["member_id"]]
        .merge(feat_web, on="member_id", how="left")
        .merge(feat_app, on="member_id", how="left")
        .merge(feat_claims, on="member_id", how="left")
        .merge(feat_life, on="member_id", how="left")
    )
    if include_labels:
        label_cols = [c for c in ("outreach", "churn") if c in members_df.columns]
        if label_cols:
            feature_matrix = feature_matrix.merge(
                members_df[["member_id"] + label_cols], on="member_id", how="left"
            )
    return feature_matrix


# ---------------------------------------------------------------------------
# Model and metric helpers
# ---------------------------------------------------------------------------

# Base-key groups: used to branch on model family throughout the pipeline
# (e.g. skip early stopping, choose SHAP explainer, etc.)
TREE_BASE_KEYS = {"LGBM", "XGB"}
LINEAR_BASE_KEYS = {"Ridge", "Lasso", "ElasticNet"}


def _make_linear_learner(base_key: str, base_params: dict | None = None) -> object:
    """Create a Pipeline(SimpleImputer → StandardScaler → linear model).

    Linear models require NaN-free, standardised inputs.  The pipeline
    handles both automatically so the rest of the code (CausalML fit/predict)
    stays unchanged.

    Parameters
    ----------
    base_key : str
        One of 'Ridge', 'Lasso', 'ElasticNet'.
    base_params : dict or None
        Hyperparameters for the linear model (e.g. {'alpha': 1.0}).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Imputer → Scaler → Linear model.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    params = dict(base_params) if base_params else {}

    if base_key == "Ridge":
        from sklearn.linear_model import Ridge
        # Default alpha=1.0 is fine for Ridge; no change.
        model = Ridge(**params)
    elif base_key == "Lasso":
        from sklearn.linear_model import Lasso
        # sklearn default alpha=1.0 often zeros all coefs on weak signals → constant predictions.
        # Use a gentler default so Section 5 CV differentiates Lasso from Ridge/ElasticNet.
        params.setdefault("alpha", 0.1)
        params.setdefault("max_iter", 10000)
        model = Lasso(**params)
    elif base_key == "ElasticNet":
        from sklearn.linear_model import ElasticNet
        params.setdefault("alpha", 0.1)
        params.setdefault("l1_ratio", 0.5)  # explicit mix so it differs from Lasso (1.0) and Ridge (0)
        params.setdefault("max_iter", 10000)
        model = ElasticNet(**params)
    else:
        raise ValueError(f"Unknown linear base_key: {base_key}")

    # SimpleImputer strategy="median" handles NaN; StandardScaler normalises
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def make_lgbm():
    """Create a LightGBM regressor with default hyperparameters (S/T-learner base)."""
    if LGBMRegressor is None:
        raise ImportError("lightgbm is not installed")
    return LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)


def make_xgb(scale_pos_weight: float = 1.0):
    """Create an XGBoost regressor with default hyperparameters (S/T-learner base)."""
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed")
    return XGBRegressor(
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )


def uplift_at_k(
    y_true: np.ndarray,
    t_true: np.ndarray,
    uplift_scores: np.ndarray,
    k: float,
) -> float:
    """Realised uplift in the top-k fraction of the population ranked by predicted uplift.

    Convention: callers should pass **negated** CausalML CATE for churn tasks
    (i.e. ``-CATE``) so that individuals who benefit most from treatment
    (most negative CATE → most positive -CATE) are ranked first.

    Parameters
    ----------
    y_true : array of int
        Observed churn labels (0/1).
    t_true : array of int
        Treatment indicator (1 = outreach, 0 = control).
    uplift_scores : array of float
        Predicted uplift (higher = more benefit from treatment).  For churn,
        pass ``-CATE`` so that persuadables rank first.
    k : float in (0, 1]
        Fraction of the population to consider (e.g. 0.10 for top 10%).

    Returns
    -------
    float
        churn_rate_control − churn_rate_treated in the top-k segment.
        Positive when treatment reduces churn in the selected segment.
    """
    n = max(1, int(len(uplift_scores) * k))
    idx = np.argsort(-uplift_scores)[:n]
    y_sub, t_sub = y_true[idx], t_true[idx]
    treated = t_sub == 1
    control = t_sub == 0
    if treated.sum() == 0 or control.sum() == 0:
        return np.nan
    return float(y_sub[control].mean() - y_sub[treated].mean())


def uplift_curve(
    y_true: np.ndarray,
    t_true: np.ndarray,
    uplift_scores: np.ndarray,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative uplift curve evaluated at n_points fractions.

    Like :func:`uplift_at_k`, callers should pass ``-CATE`` for churn tasks
    so that persuadables are ranked first.

    Returns
    -------
    ks : np.ndarray
        Fraction values (0.01 ... 1.0).
    uplift_vals : np.ndarray
        Realised uplift (churn_control − churn_treated) at each fraction.
    """
    order = np.argsort(-uplift_scores)
    y_sorted = y_true[order]
    t_sorted = t_true[order]
    ks = np.linspace(0.01, 1.0, n_points)
    vals = []
    for frac in ks:
        n = max(1, int(len(y_sorted) * frac))
        y_sub, t_sub = y_sorted[:n], t_sorted[:n]
        nt, nc = (t_sub == 1).sum(), (t_sub == 0).sum()
        if nt == 0 or nc == 0:
            vals.append(np.nan)
        else:
            vals.append(y_sub[t_sub == 0].mean() - y_sub[t_sub == 1].mean())
    return ks, np.array(vals)


def approx_auuc(ks: np.ndarray, uplift_vals: np.ndarray) -> float:
    """Area under the uplift curve (trapezoidal rule, NaN-safe)."""
    valid = ~np.isnan(uplift_vals)
    if valid.sum() < 2:
        return np.nan
    return float(np.trapz(uplift_vals[valid], ks[valid]))


def assign_segments(
    uplift_scores: np.ndarray,
    baseline_scores: np.ndarray | None = None,
    uplift_threshold: float | str | None = None,
    zero_epsilon: float | None = None,
) -> np.ndarray:
    """Assign four uplift segments using formal definitions (Radcliffe / uplift literature).

    **Formal definitions** (default when *baseline_scores* is provided):
    * **Persuadables**: uplift **> ε** — positive incremental effect; only segment to target.
    * **Sure Things**: **|uplift| ≤ ε** AND **low baseline** (low churn) — would retain anyway; 0 gain.
    * **Lost Causes**: **|uplift| ≤ ε** AND **high baseline** (high churn) — will churn anyway; 0 gain.
    * **Do-Not-Disturb**: uplift **< −ε** — negative incremental effect; avoid.

    Sure Things and Lost Causes both have **zero uplift** (no incremental effect); the only
    difference is baseline churn risk. Segment sizes are data-driven (depend on score distribution).

    *uplift_threshold* (when baseline_scores is set):
    * ``None`` (default): Formal definition — zero band |uplift| ≤ ε, split by baseline.
    * ``"median"``: Persuadables = top 50% (uplift >= median); below median split by baseline.
    * ``"percentile"``: bottom 25% = DND, top 50% = Persuadables, middle 25% by baseline.
    * A float (e.g. 0.01): Persuadables = uplift > that value; zero band and low band by baseline.

    *zero_epsilon*: half-width of zero band (|uplift| ≤ ε). Also DND boundary (uplift < −ε).
    If None, default 0.01.

    When *baseline_scores* is None, falls back to a quartile split on uplift only.

    Parameters
    ----------
    uplift_scores : np.ndarray
        Predicted uplift (higher = more benefit from treatment).
        For churn tasks, pass ``-CATE`` (retention uplift).
    baseline_scores : np.ndarray or None
        Predicted baseline outcome (e.g. P(churn | no treatment)).
        Higher = more at-risk. If None, use quartile fallback.
    uplift_threshold : float, "percentile", "median", "zero", or None, default None
        None (default) = formal definition (zero band |uplift| ≤ ε, split by baseline);
        "median" = top 50% = Persuadables; "percentile" = p25/p50; float = fixed cutoff.
    zero_epsilon : float or None, default None
        Zero-band half-width (|uplift| ≤ ε) and DND boundary. If None, 0.01 for formal default.

    Returns
    -------
    np.ndarray of str
        Segment labels: Persuadables, Sure Things, Lost Causes, Do-Not-Disturb.
    """
    seg = np.empty(len(uplift_scores), dtype=object)

    if baseline_scores is not None:
        baseline_median = float(np.nanmedian(baseline_scores))
        baseline_high = baseline_scores >= baseline_median

        use_percentile = (
            isinstance(uplift_threshold, str)
            and uplift_threshold.strip().lower() == "percentile"
        )
        use_median = (
            isinstance(uplift_threshold, str)
            and uplift_threshold.strip().lower() == "median"
        )

        if use_percentile:
            # ── Percentile-based: top 50% = Persuadables, bottom 25% = DND ─────
            p25 = float(np.nanpercentile(uplift_scores, 25))
            p50 = float(np.nanpercentile(uplift_scores, 50))
            top_uplift = uplift_scores >= p50
            bottom_uplift = uplift_scores < p25
            middle_uplift = (uplift_scores >= p25) & (uplift_scores < p50)
            seg[top_uplift] = "Persuadables"
            seg[bottom_uplift] = "Do-Not-Disturb"
            # Sure Things = low baseline (retain anyway), Lost Causes = high baseline
            seg[middle_uplift & baseline_high] = "Lost Causes"
            seg[middle_uplift & ~baseline_high] = "Sure Things"
        elif use_median:
            # ── Median-based: Persuadables = top 50% (>= median) ─────────────────
            eps = 0.005 if zero_epsilon is None else float(zero_epsilon)
            uplift_median = float(np.nanmedian(uplift_scores))
            seg[uplift_scores < -eps] = "Do-Not-Disturb"
            seg[uplift_scores >= uplift_median] = "Persuadables"
            in_band = (uplift_scores >= -eps) & (uplift_scores < uplift_median)
            seg[in_band & baseline_high] = "Lost Causes"
            seg[in_band & ~baseline_high] = "Sure Things"
        elif uplift_threshold is None or (
            isinstance(uplift_threshold, str)
            and uplift_threshold.strip().lower() in ("zero", "zero_band")
        ):
            # ── Formal definition (default): Sure Things & Lost Causes = zero uplift band ─
            # |uplift| ≤ ε → 0 gain; split by baseline. Persuadables = uplift > ε; DND = uplift < −ε.
            eps = 0.01 if zero_epsilon is None else float(zero_epsilon)
            seg[uplift_scores < -eps] = "Do-Not-Disturb"
            seg[uplift_scores > eps] = "Persuadables"
            in_zero_band = np.abs(uplift_scores) <= eps
            seg[in_zero_band & baseline_high] = "Lost Causes"
            seg[in_zero_band & ~baseline_high] = "Sure Things"
        else:
            # ── Fixed threshold: Persuadables = uplift > threshold ─────────────
            if uplift_threshold is None or (
                isinstance(uplift_threshold, (int, float)) and uplift_threshold == 0
            ):
                uplift_cut = 0.0
            else:
                uplift_cut = float(uplift_threshold)
            if zero_epsilon is None:
                eps = 0.005 if uplift_cut == 0 else max(0.005, uplift_cut / 2.0)
            else:
                eps = float(zero_epsilon)
            seg[uplift_scores < -eps] = "Do-Not-Disturb"
            seg[uplift_scores > uplift_cut] = "Persuadables"
            # Zero band |uplift| < ε: split by baseline (Sure Things = low, Lost Causes = high)
            in_zero_band = np.abs(uplift_scores) < eps
            seg[in_zero_band & baseline_high] = "Lost Causes"
            seg[in_zero_band & ~baseline_high] = "Sure Things"
            # Low-uplift band (ε <= uplift <= threshold): split by baseline
            in_low_band = (uplift_scores >= eps) & (uplift_scores <= uplift_cut)
            seg[in_low_band & baseline_high] = "Lost Causes"
            seg[in_low_band & ~baseline_high] = "Sure Things"
    else:
        # ── Quartile fallback (1-D, backward compatible) ────────────
        q25, q50, q75 = np.nanquantile(uplift_scores, [0.25, 0.50, 0.75])
        seg[uplift_scores >= q75] = "Persuadables"
        seg[(uplift_scores >= q50) & (uplift_scores < q75)] = "Sure Things"
        seg[(uplift_scores >= q25) & (uplift_scores < q50)] = "Lost Causes"
        seg[uplift_scores < q25] = "Do-Not-Disturb"

    return seg


def predict_baseline_control(
    model,
    X: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """Extract P(Y | X, T=0) — the control/baseline prediction from a fitted meta-learner.

    Supports S-Learner, T-Learner, and other CausalML meta-learners that expose
    either ``models_c`` (T-learner control model) or ``models`` (S-learner single model).

    Parameters
    ----------
    model : fitted CausalML meta-learner
        e.g. BaseSRegressor, BaseTRegressor (fitted).
    X : np.ndarray or pd.DataFrame
        Feature matrix (same columns used during ``.fit()``, **without**
        the treatment indicator).

    Returns
    -------
    np.ndarray
        Predicted outcome under control (T=0) for each observation.
    """
    X_arr = np.asarray(X)
    n = X_arr.shape[0]

    # T-Learner: control and treatment models stored in models_c and models_t (X only).
    models_c = getattr(model, "models_c", None)
    models_t = getattr(model, "models_t", None)
    if models_c is not None and models_t is not None and len(models_c) > 0:
        t_groups = getattr(model, "t_groups", None)
        if t_groups is None or len(t_groups) == 0:
            group = next(iter(models_c))
        else:
            group = t_groups[0]
        control_model = models_c[group]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
                module="sklearn.utils.validation",
            )
            return np.asarray(control_model.predict(X_arr)).ravel()
    # S-Learner: single model in self.models, trained on [T | X].
    fitted_models = getattr(model, "models", None)
    if fitted_models is None or len(fitted_models) == 0:
        raise AttributeError(
            "Cannot find fitted internal models. Expected 'models_c' (T-learner) or "
            "'models' (S-learner). Has fit() been called?"
        )
    group_key = next(iter(fitted_models))
    base_model = fitted_models[group_key]
    X_control = np.hstack([np.zeros((n, 1)), X_arr])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
            module="sklearn.utils.validation",
        )
        return base_model.predict(X_control)


# Backward-compatibility alias (name referred to S-learner only; function supports all learners).
predict_baseline_slearner = predict_baseline_control


def build_model(
    meta_key: str,
    base_key: str,
    spw: float,
    base_params: dict | None = None,
):
    """Instantiate a CausalML meta-learner with the requested base learner.

    When *base_params* is None, uses default base learners (make_lgbm / make_xgb).
    When *base_params* is a dict (e.g. from grid search), builds the base learner
    with those keyword arguments plus fixed random_state / scale_pos_weight.

    Parameters
    ----------
    meta_key : str
        One of 'S', 'T'.
    base_key : str
        One of 'LGBM', 'XGB', 'Ridge', 'Lasso', 'ElasticNet'.
    spw : float
        scale_pos_weight for XGBoost (ignored for other base learners).
    base_params : dict | None
        Optional extra kwargs for the base learner (max_depth, learning_rate, etc.).
        If None, defaults are used.

    Returns
    -------
    CausalML meta-learner instance.
    """
    if BaseSRegressor is None or BaseTRegressor is None:
        raise ImportError("causalml is not installed")

    # Linear base learners: Pipeline(Imputer → Scaler → Model)
    if base_key in LINEAR_BASE_KEYS:
        learner = _make_linear_learner(base_key, base_params)
    elif base_params is None:
        learner = make_lgbm() if base_key == "LGBM" else make_xgb(spw)
    else:
        if base_key == "LGBM":
            if LGBMRegressor is None:
                raise ImportError("lightgbm is not installed")
            learner = LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, **base_params)
        elif base_key == "XGB":
            if XGBRegressor is None:
                raise ImportError("xgboost is not installed")
            learner = XGBRegressor(
                random_state=RANDOM_STATE,
                scale_pos_weight=spw,
                verbosity=0,
                **base_params,
            )
        else:
            raise ValueError(f"Unknown base_key: {base_key}")

    meta_map = {"S": BaseSRegressor, "T": BaseTRegressor}
    return meta_map[meta_key](learner=learner)


# ---------------------------------------------------------------------------
# Uplift CV and model comparison (Section 5)
# ---------------------------------------------------------------------------


def qini_coefficient(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
) -> float:
    """Compute Qini coefficient (normalised area between model and random Qini curves).

    Delegates to CausalML's ``qini_score``, which internally sorts by
    descending ``uplift_scores`` and computes
    ``treated_positive − control_positive × (N_T / N_C)``.
    For the coefficient to be positive when the model is good, ``y_true=1``
    must represent the **desirable** outcome and higher ``uplift_scores``
    must correspond to individuals who benefit most from treatment.

    **Churn convention:** callers should pass ``1 − y`` (retention) and
    ``-CATE`` (negated churn CATE = retention CATE) so that retained (Y=1)
    is "good" and persuadables (positive retention uplift) rank first.

    Parameters
    ----------
    y_true : np.ndarray
        Binary outcome where Y=1 is the **desirable** outcome
        (for churn tasks, pass ``1 - churn`` = retention).
    treatment : np.ndarray
        Treatment indicator (1 = treated, 0 = control).
    uplift_scores : np.ndarray
        Predicted uplift where higher = more treatment benefit
        (for churn tasks, pass ``-CATE`` = retention CATE).

    Returns
    -------
    float
        Qini coefficient, or np.nan if causalml.metrics is not available.
    """
    if causalml_qini_score is None:
        return np.nan
    try:
        # CausalML qini_score(df) expects outcome_col, treatment_col, and one or more *model* columns (predicted uplift).
        # Do not set treatment_effect_col so that our prediction column is used as the model column.
        df_qini = pd.DataFrame(
            {"y": y_true, "w": treatment, "pred": np.asarray(uplift_scores).ravel()}
        )
        result = causalml_qini_score(
            df_qini, outcome_col="y", treatment_col="w", normalize=True
        )
        return float(result.iloc[0]) if hasattr(result, "iloc") else float(result)
    except Exception:
        return np.nan


def evaluate_uplift_metrics(
    y: np.ndarray,
    treatment: np.ndarray,
    cate: np.ndarray,
    k_fracs: list[float] | None = None,
    n_points: int = 100,
    print_result: bool = True,
) -> dict:
    """Compute AUUC, Qini, and uplift@k from raw CATE predictions (churn convention).

    Uses negated CATE as the scoring signal (higher = more benefit from treatment).
    Qini is computed with retention labels (1 - y) and negated CATE.

    Parameters
    ----------
    y : np.ndarray
        Churn labels (0/1).
    treatment : np.ndarray
        Treatment indicator (1 = treated, 0 = control).
    cate : np.ndarray
        Raw CATE from the model (churn CATE; will be negated for scoring).
    k_fracs : list of float or None
        Fractions for uplift@k (e.g. [0.10, 0.20]). Default [0.10, 0.20].
    n_points : int
        Number of points for the uplift curve (AUUC). Default 100.
    print_result : bool
        If True, print AUUC, Qini, and uplift@k for each fraction. Default True.

    Returns
    -------
    dict
        Keys: auuc, qini_coefficient, uplift_at_k (dict k_frac -> value), ks, uplift_vals.
    """
    if k_fracs is None:
        k_fracs = [0.10, 0.20]
    cate = np.asarray(cate).ravel()
    uplift_scores = -cate
    y_ret = 1 - np.asarray(y).ravel()

    qini = qini_coefficient(y_ret, treatment, uplift_scores)
    ks, uplift_vals = uplift_curve(y, treatment, uplift_scores, n_points=n_points)
    auuc = approx_auuc(ks, uplift_vals)
    uplift_at_k_vals = {
        k: uplift_at_k(y, treatment, uplift_scores, k=k) for k in k_fracs
    }

    if print_result:
        print("Training-set evaluation (final model on full train):")
        print(f"  AUUC              : {auuc:.4f}")
        print(f"  Qini coefficient  : {qini:.4f}")
        for k in k_fracs:
            print(f"  Uplift@{int(k * 100)}%        : {uplift_at_k_vals[k]:.4f}")

    return {
        "auuc": auuc,
        "qini_coefficient": qini,
        "uplift_at_k": uplift_at_k_vals,
        "ks": ks,
        "uplift_vals": uplift_vals,
    }


# Default colors for Qini curve plots (same order as typical use: models then Random).
# Matplotlib default cycle first 4 + gray for baseline.
QINI_CURVE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def get_qini_curve_single(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_points: int = 101,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Qini curve for a single set (one fold or one test set). General for any holdout.

    Uses CausalML get_qini; returns (fractions, values) on a fixed grid so curves
    from different sample sizes can be averaged (e.g. mean over folds).

    CausalML sorts by descending ``uplift_scores`` and computes cumulative
    ``treated_positive − control_positive × (N_T / N_C)``.  For the curve
    to go **up** when the model is good, ``y_true=1`` must be the
    **desirable** outcome and higher ``uplift_scores`` must correspond
    to individuals who benefit most from treatment.

    **Churn convention:** callers should pass ``1 − y`` (retention) and
    ``-CATE`` so that retained (Y=1) is good and persuadables rank first.

    Parameters
    ----------
    y_true : np.ndarray
        Binary outcome where Y=1 is the desirable outcome
        (for churn tasks, pass ``1 - churn`` = retention).
    treatment : np.ndarray
        Treatment indicator (1 = treated, 0 = control).
    uplift_scores : np.ndarray
        Predicted uplift where higher = more treatment benefit
        (for churn tasks, pass ``-CATE`` = retention CATE).
    n_points : int
        Number of fraction points (0 to 1). Default 101.
    normalize : bool
        If True, normalize curve to max |value| = 1 (CausalML convention).

    Returns
    -------
    fractions : np.ndarray
        Fraction of population (0 to 1), shape (n_points,).
    values : np.ndarray
        Qini curve values at each fraction, shape (n_points,).
    """
    if causalml_get_qini is None or len(y_true) == 0:
        frac = np.linspace(0, 1, n_points)
        return frac, np.full(n_points, np.nan)
    try:
        df_qini = pd.DataFrame(
            {"y": y_true, "w": treatment, "pred": np.asarray(uplift_scores).ravel()}
        )
        qini_df = causalml_get_qini(
            df_qini, outcome_col="y", treatment_col="w", normalize=normalize
        )
        # qini_df: index 0..n, one column "pred"
        if "pred" not in qini_df.columns or len(qini_df) < 2:
            frac = np.linspace(0, 1, n_points)
            return frac, np.full(n_points, np.nan)
        # CausalML returns raw cumulative counts; normalise by population size
        # so the y-axis shows incremental gain as a proportion (matches standard
        # Qini references, e.g. Kaggle/sklift, where y ∈ [0, ~0.06]).
        n_samples = len(y_true)
        vals_raw = qini_df["pred"].values / max(1, n_samples)
        frac_raw = np.arange(len(vals_raw), dtype=float) / max(1, len(vals_raw) - 1)
        if len(frac_raw) == 0:
            frac_raw = np.array([0.0, 1.0])
            vals_raw = np.array([0.0, 0.0])
        frac_grid = np.linspace(0, 1, n_points)
        vals_interp = np.interp(frac_grid, frac_raw, vals_raw)
        return frac_grid, vals_interp
    except Exception:
        frac = np.linspace(0, 1, n_points)
        return frac, np.full(n_points, np.nan)


def aggregate_qini_curves_over_folds(
    curves_per_fold: list[tuple[np.ndarray, np.ndarray]],
    n_points: int = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average Qini curves over folds. Each fold curve should be on the same fraction grid.

    Parameters
    ----------
    curves_per_fold : list of (fractions, values)
        Each from get_qini_curve_single(..., n_points=n_points).
    n_points : int
        Expected length of each curve (for validation).

    Returns
    -------
    fractions : np.ndarray
        Common grid (0 to 1).
    mean_vals : np.ndarray
        Mean curve.
    std_vals : np.ndarray
        Std across folds (NaN if single fold).
    """
    if not curves_per_fold:
        frac = np.linspace(0, 1, n_points)
        return frac, np.full(n_points, np.nan), np.full(n_points, np.nan)
    # Assume all on same grid (from get_qini_curve_single with same n_points)
    stacked = np.array([c[1] for c in curves_per_fold])
    frac = curves_per_fold[0][0]
    mean_vals = np.nanmean(stacked, axis=0)
    std_vals = np.nanstd(stacked, axis=0) if stacked.shape[0] > 1 else np.full(frac.shape[0], np.nan)
    return frac, mean_vals, std_vals


def get_validation_qini_curves(
    X: pd.DataFrame,
    y: np.ndarray,
    treatment: np.ndarray,
    candidate_defs: list[tuple[str, str, str]],
    cv_summary: pd.DataFrame,
    top_n: int = 3,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    scale_pos_weight: float = 1.0,
    n_points: int = 101,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Qini curves for top_n models + Random, mean (and std) across CV folds.

    For each candidate and each fold: fit on train, predict on val, compute Qini curve
    on val. Then average curve values at each fraction across folds. General design
    allows single test set in future by passing n_splits=1 and appropriate data.

    Internally converts churn to retention (``1 − y``) and negates CATE
    (``-raw_pred``) before calling CausalML, so the Qini curve goes UP when
    the model correctly targets persuadables (standard Qini convention where
    ``y=1`` is the desirable outcome).

    Parameters
    ----------
    X, y, treatment : as in run_uplift_cv (y is churn 0/1).
    candidate_defs, cv_summary : candidate list and summary table (for top_n by auuc_mean).
    top_n : int
        Number of top candidates by mean AUUC.
    n_splits, random_state, scale_pos_weight : CV and model args.
    n_points : int
        Points on fraction axis for Qini curve.

    Returns
    -------
    dict[str, (fractions, mean_vals, std_vals)]
        Keys: model name or "Random baseline".
    """
    from sklearn.model_selection import StratifiedKFold

    stratify = 2 * treatment + y
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(X, stratify))
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    order = cv_summary.sort_values("auuc_mean", ascending=False)["candidate"].tolist()
    defs_by_name = {c[0]: (c[1], c[2]) for c in candidate_defs}
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(random_state)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
            module="sklearn.utils.validation",
        )
        for name in order[:top_n]:
            if name not in defs_by_name:
                continue
            meta_key, base_key = defs_by_name[name]
            curves_fold = []
            for train_idx, val_idx in splits:
                X_tr = X_df.iloc[train_idx]
                X_val = X_df.iloc[val_idx]
                y_val = y[val_idx]
                t_val = treatment[val_idx]
                base_params_curves: dict | None = None
                if base_key == "XGB":
                    best_n = _early_stopping_n_estimators_single(
                        X_tr, y[train_idx], treatment[train_idx],
                        base_params={},
                        scale_pos_weight=scale_pos_weight,
                        random_state=random_state,
                        n_estimators_max=800,
                    )
                    base_params_curves = {"n_estimators": best_n}
                model = build_model(meta_key, base_key, scale_pos_weight, base_params=base_params_curves)
                model.fit(X_tr, treatment[train_idx], y[train_idx])
                pred = model.predict(X_val)
                raw_pred = np.asarray(pred).ravel()
                # Churn → retention convention for Qini:
                # CausalML's get_qini sorts by descending pred and computes
                # treated_positive − control_positive * ratio. For the curve to
                # go UP (good model above random diagonal), Y=1 must be the
                # "good" outcome and higher score must mean more treatment benefit.
                # raw CATE = E[churn|T=1] - E[churn|T=0] → negative for persuadables.
                # Flip: retention = 1-y (Y=1 = retained = good), neg_pred = -CATE
                # = E[retention|T=1] - E[retention|T=0] → positive for persuadables.
                neg_pred = -raw_pred
                retention = 1 - y_val
                frac, vals = get_qini_curve_single(retention, t_val, neg_pred, n_points=n_points)
                curves_fold.append((frac, vals))
            frac, mean_vals, std_vals = aggregate_qini_curves_over_folds(curves_fold, n_points=n_points)
            result[name] = (frac, mean_vals, std_vals)

        # Random baseline: one random permutation per fold (still use retention outcome)
        curves_fold = []
        for train_idx, val_idx in splits:
            y_val = y[val_idx]
            t_val = treatment[val_idx]
            n_val = len(y_val)
            retention = 1 - y_val
            random_pred = rng.permutation(np.arange(n_val, dtype=float))  # shuffle indices as proxy
            frac, vals = get_qini_curve_single(retention, t_val, random_pred, n_points=n_points)
            curves_fold.append((frac, vals))
        frac, mean_vals, std_vals = aggregate_qini_curves_over_folds(curves_fold, n_points=n_points)
        result["Random baseline"] = (frac, mean_vals, std_vals)

    return result


def plot_qini_curves_comparison(
    curves: dict[str, tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]],
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    colors: list[str] | None = None,
    legend_outside: bool | None = None,
) -> plt.Axes:
    """Overlay Qini curves (mean over folds) with diagonal random baseline.

    Each value in *curves* is (fractions, mean_vals) or (fractions, mean_vals, std_vals).
    Draws a gray dashed diagonal from (0,0) to (1, gain_at_100%) as the
    random-targeting baseline (standard Qini convention). If "Random baseline"
    is in curves, it is not drawn as a separate curve; only the diagonal
    represents random.

    Parameters
    ----------
    curves : dict mapping label -> (frac, mean[, std])
    ax, save_path : optional axes and save path.
    colors : optional list of color specs; default QINI_CURVE_COLORS.
    legend_outside : bool or None
        If True, place legend outside plot (right) with compact layout.
        If None, auto: outside when more than 4 curve labels (e.g. HP grid plot).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    if colors is None:
        colors = QINI_CURVE_COLORS
    n_curves = sum(1 for k in curves if k != "Random baseline")
    if legend_outside is None:
        legend_outside = n_curves > 4

    # Diagonal end = cumulative gain at 100% (same for any targeting order).
    # Use "Random baseline" curve end point if present; else first curve.
    y_end = np.nan
    if "Random baseline" in curves:
        mean_vals = curves["Random baseline"][1]
        if len(mean_vals) and np.isfinite(mean_vals[-1]):
            y_end = float(mean_vals[-1])
    if not np.isfinite(y_end):
        for _, data in curves.items():
            mean_vals = data[1]
            if len(mean_vals) and np.isfinite(mean_vals[-1]):
                y_end = float(mean_vals[-1])
                break
    if not np.isfinite(y_end):
        y_end = 0.0

    # Draw diagonal random baseline first (gray dashed, like reference Qini plots).
    ax.plot([0, 1], [0, y_end], "--", color="gray", linewidth=2, label="Random baseline")

    # Plot model curves (skip empirical "Random baseline" curve; diagonal is the baseline).
    color_index = 0
    for label, data in curves.items():
        if label == "Random baseline":
            continue
        # Unpack; std_vals are computed but not plotted (clean chart, no bands).
        frac = data[0]
        mean_vals = data[1]
        valid = ~np.isnan(mean_vals)
        if valid.sum() < 2:
            continue
        color = colors[color_index % len(colors)]
        color_index += 1
        ax.plot(frac[valid], mean_vals[valid], label=label, linewidth=2, color=color)
    ax.set_xlabel("Proportion targeted")
    ax.set_ylabel("Uplift")
    ax.set_title("Qini curves (mean over CV folds)")
    # When many curves (e.g. top-10 HP combos), put legend below in 2 columns so it doesn't cover the plot.
    if legend_outside:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=2,
            fontsize="small",
            frameon=True,
        )
        plt.tight_layout(rect=[0, 0.22, 1, 1])  # leave room for legend below (2-col, many rows)
    else:
        ax.legend(loc="best", frameon=True)
        plt.tight_layout()
    set_axes_clear(ax, x_axis_at_zero=False)
    if save_path is not None:
        ax.figure.savefig(Path(save_path), bbox_inches="tight", dpi=150)
    return ax


def run_uplift_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    treatment: np.ndarray,
    candidate_defs: list[tuple[str, str, str]],
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    scale_pos_weight: float = 1.0,
    n_curve_points: int = 100,
    uplift_k_fracs: list[float] | None = None,
) -> list[dict]:
    """Run stratified K-fold CV for each uplift candidate; return per-fold metrics.

    Stratification is by 2*treatment + y so treatment/control and outcome balance
    are preserved in each fold. For each fold and candidate: fit on train, predict
    on val, compute AUUC, Qini, uplift@k. Records n_val, n_treated_val, n_control_val per fold.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric).
    y : np.ndarray
        Binary outcome (e.g. churn 0/1).
    treatment : np.ndarray
        Treatment indicator (1 = treated, 0 = control).
    candidate_defs : list of (name, meta_key, base_key)
        e.g. [('S+LGBM','S','LGBM'), ('T+XGB','T','XGB')].
    n_splits : int
        Number of CV folds.
    random_state : int
        For StratifiedKFold.
    scale_pos_weight : float
        Passed to XGB base learner (ignored for LGBM).
    n_curve_points : int
        Points for uplift curve (AUUC).
    uplift_k_fracs : list of float or None
        Fractions for uplift@k. Default [0.05, 0.1, 0.2].

    Returns
    -------
    list of dict
        Each dict: candidate, fold, auuc, qini, uplift_at_k_*, n_val, n_treated_val, n_control_val.
    """
    if uplift_k_fracs is None:
        uplift_k_fracs = [0.05, 0.1, 0.2]
    # Keep only rows with valid outcome (0/1) and treatment (0/1); drop NaN or invalid so metrics are correct.
    y_ = np.asarray(y, dtype=np.float64)
    t_ = np.asarray(treatment, dtype=np.float64)
    valid = np.isfinite(y_) & np.isin(t_, (0.0, 1.0)) & np.isin(y_, (0.0, 1.0))
    if not np.all(valid):
        X = X.loc[valid] if isinstance(X, pd.DataFrame) else X[valid]
        y = np.asarray(y_[valid], dtype=np.float64)
        treatment = np.asarray(t_[valid], dtype=np.float64)
        # Ensure integer 0/1 for stratification and indexing
        y = (y > 0.5).astype(np.int64)
        treatment = (treatment > 0.5).astype(np.int64)
    else:
        y = (np.asarray(y, dtype=np.float64) > 0.5).astype(np.int64)
        treatment = (np.asarray(treatment, dtype=np.float64) > 0.5).astype(np.int64)
    results: list[dict] = []
    # CausalML converts X to numpy before calling the base learner; sklearn then warns "X does not have valid
    # feature names" at predict. Suppress that specific warning so output is clean.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
            module="sklearn.utils.validation",
        )
        _run_uplift_cv_loop(
            X=X,
            y=y,
            treatment=treatment,
            candidate_defs=candidate_defs,
            n_splits=n_splits,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            uplift_k_fracs=uplift_k_fracs,
            n_curve_points=n_curve_points,
            results_ref=results,
        )
    return results


def _run_uplift_cv_loop(
    X,
    y,
    treatment,
    candidate_defs,
    n_splits,
    random_state,
    scale_pos_weight,
    uplift_k_fracs,
    n_curve_points,
    results_ref: list,
) -> None:
    """Inner loop of run_uplift_cv (fit/predict per candidate per fold). Called with warning filter active."""
    from sklearn.model_selection import StratifiedKFold

    if uplift_k_fracs is None:
        uplift_k_fracs = [0.05, 0.1, 0.2]
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    stratify = 2 * treatment + y
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for name, meta_key, base_key in candidate_defs:
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, stratify)):
            X_tr = X_df.iloc[train_idx]
            X_val = X_df.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            t_tr, t_val = treatment[train_idx], treatment[val_idx]
            n_val = len(val_idx)
            n_treated_val = int((t_val == 1).sum())
            n_control_val = int((t_val == 0).sum())
            # For XGB, apply early stopping so n_estimators is chosen by validation (same in all flows).
            base_params_cv: dict | None = None
            if base_key == "XGB":
                best_n = _early_stopping_n_estimators_single(
                    X_tr, y_tr, t_tr,
                    base_params={},
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    n_estimators_max=800,
                )
                base_params_cv = {"n_estimators": best_n}
            model = build_model(meta_key, base_key, scale_pos_weight, base_params=base_params_cv)
            model.fit(X_tr, t_tr, y_tr)
            pred = model.predict(X_val)
            # CausalML predict returns CATE = E[Y|T=1] - E[Y|T=0].
            # For churn (Y=1 bad), negative CATE means treatment helps.
            raw_pred = np.asarray(pred).ravel()  # raw CATE for CausalML Qini
            neg_pred = -raw_pred  # negated: higher = more benefit

            # Our custom metrics use negated CATE so persuadables (most negative
            # CATE → most positive -CATE) rank first; formula is
            # control_churn − treated_churn, which is positive when treatment helps.
            ks, uplift_vals = uplift_curve(
                y_val, t_val, neg_pred, n_points=n_curve_points
            )
            auuc = approx_auuc(ks, uplift_vals)

            # Churn → retention for Qini: CausalML's qini_score sorts by
            # descending score and computes treated_positive − control_positive
            # ratio.  For the coefficient to be positive when the model is good,
            # Y=1 must be the "good" outcome → use retention (1-y) and negated
            # CATE (neg_pred) so persuadables rank first.
            qini = qini_coefficient(1 - y_val, t_val, neg_pred)

            row: dict = {
                "candidate": name,
                "fold": fold,
                "auuc": auuc,
                "qini": qini,
                "n_val": n_val,
                "n_treated_val": n_treated_val,
                "n_control_val": n_control_val,
            }
            for frac in uplift_k_fracs:
                key = f"uplift_at_k_{int(frac * 100)}"
                row[key] = uplift_at_k(y_val, t_val, neg_pred, frac)
            results_ref.append(row)


def summarize_uplift_cv(
    cv_results: list[dict],
    metric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate per-candidate CV metrics (mean and std).

    Parameters
    ----------
    cv_results : list of dict
        Output from run_uplift_cv.
    metric_cols : list of str or None
        Columns to aggregate. Default: auuc, qini, uplift_at_k_5, uplift_at_k_10, uplift_at_k_20.

    Returns
    -------
    pd.DataFrame
        One row per candidate; columns candidate, and for each metric: mean, std.
    """
    if metric_cols is None:
        metric_cols = [
            "auuc",
            "qini",
            "uplift_at_k_5",
            "uplift_at_k_10",
            "uplift_at_k_20",
        ]
    df = pd.DataFrame(cv_results)
    present = [m for m in metric_cols if m in df.columns]
    agg_dict: dict = {}
    for m in present:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")
    if not agg_dict:
        return df.groupby("candidate").first().reset_index()
    return df.groupby("candidate").agg(**agg_dict).reset_index()


def build_uplift_cv_comparison_table(
    cv_results: list[dict],
    metric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build comparison table: mean±std per candidate and fold sanity (n_val, n_treated_val, n_control_val).

    Parameters
    ----------
    cv_results : list of dict
        Output from run_uplift_cv.
    metric_cols : list of str or None
        Default: auuc, qini, uplift_at_k_5, uplift_at_k_10, uplift_at_k_20.

    Returns
    -------
    pd.DataFrame
        One row per candidate; metric mean/std plus n_val_mean, n_treated_val_mean, n_control_val_mean.
    """
    if metric_cols is None:
        metric_cols = [
            "auuc",
            "qini",
            "uplift_at_k_5",
            "uplift_at_k_10",
            "uplift_at_k_20",
        ]
    df = pd.DataFrame(cv_results)
    present = [m for m in metric_cols if m in df.columns]
    agg_dict: dict = {}
    for m in present:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")
    for col in ["n_val", "n_treated_val", "n_control_val"]:
        if col in df.columns:
            agg_dict[f"{col}_mean"] = (col, "mean")
    if not agg_dict:
        return df.groupby("candidate").first().reset_index()
    return df.groupby("candidate").agg(**agg_dict).reset_index()


def plot_auuc_comparison(
    cv_summary: pd.DataFrame,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Bar plot of mean AUUC with error bars (std) across models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    names = cv_summary["candidate"].astype(str)
    means = cv_summary["auuc_mean"].values
    stds = (
        cv_summary["auuc_std"].values
        if "auuc_std" in cv_summary.columns
        else np.zeros_like(means)
    )
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4, edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("AUUC (mean ± std)")
    ax.set_title("Uplift model comparison: AUUC")
    set_axes_clear(ax, x_axis_at_zero=False)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(Path(save_path), bbox_inches="tight", dpi=150)
    return ax


def plot_uplift_at_k_comparison(
    cv_summary: pd.DataFrame,
    k_fracs: list[float] | None = None,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Bar plot of mean uplift@k (and std) for k=10% and 20% (and optionally 5%) across models."""
    if k_fracs is None:
        k_fracs = [0.1, 0.2]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    names = cv_summary["candidate"].astype(str)
    x = np.arange(len(names))
    width = 0.8 / len(k_fracs)
    for i, frac in enumerate(k_fracs):
        key = f"uplift_at_k_{int(frac * 100)}"
        mean_col = f"{key}_mean"
        std_col = f"{key}_std"
        if mean_col not in cv_summary.columns:
            continue
        means = cv_summary[mean_col].values
        stds = (
            cv_summary[std_col].values
            if std_col in cv_summary.columns
            else np.zeros_like(means)
        )
        offset = (i - len(k_fracs) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=2,
            label=f"uplift@{int(frac * 100)}%",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Uplift (churn rate difference)")
    ax.set_title("Uplift model comparison: uplift@k")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8)
    set_axes_clear(ax, x_axis_at_zero=False)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(Path(save_path), bbox_inches="tight", dpi=150)
    return ax


def save_uplift_cv_report(
    cv_summary: pd.DataFrame,
    table_path: str | Path,
    figure_dir: str | Path,
    cv_results: list[dict] | None = None,
    X: pd.DataFrame | None = None,
    y: np.ndarray | None = None,
    treatment: np.ndarray | None = None,
    candidate_defs: list[tuple[str, str, str]] | None = None,
    top_n_curves: int = 3,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    scale_pos_weight: float = 1.0,
) -> None:
    """Save comparison table (CSV) and figures (AUUC bar, uplift@k bar, curves) to disk."""
    Path(table_path).parent.mkdir(parents=True, exist_ok=True)
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    cv_summary.to_csv(table_path, index=False)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    plot_auuc_comparison(
        cv_summary, ax=ax1, save_path=Path(figure_dir) / "auuc_comparison.png"
    )
    plt.close(fig1)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    plot_uplift_at_k_comparison(
        cv_summary, ax=ax2, save_path=Path(figure_dir) / "uplift_at_k_comparison.png"
    )
    plt.close(fig2)
    if (
        X is not None
        and y is not None
        and treatment is not None
        and candidate_defs is not None
    ):
        curves = get_validation_qini_curves(
            X,
            y,
            treatment,
            candidate_defs,
            cv_summary,
            top_n=top_n_curves,
            n_splits=n_splits,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        )
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        plot_qini_curves_comparison(
            curves, ax=ax3, save_path=Path(figure_dir) / "qini_curves_comparison.png"
        )
        plt.close(fig3)


def select_best_uplift_model(
    cv_summary: pd.DataFrame,
    metric: str = "auuc",
    higher_is_better: bool = True,
) -> str:
    """Return the best candidate name by a given metric (e.g. auuc_mean). Optional; no auto-selection in Section 5."""
    mean_col = f"{metric}_mean" if f"{metric}_mean" in cv_summary.columns else metric
    if mean_col not in cv_summary.columns:
        raise ValueError(f"Metric column {mean_col} not in cv_summary")
    if higher_is_better:
        best_idx = cv_summary[mean_col].idxmax()
    else:
        best_idx = cv_summary[mean_col].idxmin()
    return str(cv_summary.loc[best_idx, "candidate"])


# ---------------------------------------------------------------------------
# Hyperparameter Tuning — Grid Search (Section 6)
# ---------------------------------------------------------------------------


def get_uplift_hp_grid(base_key: str) -> dict[str, list]:
    """Return the default hyperparameter grid for uplift grid search by base learner.

    Ridge/Lasso/ElasticNet use alpha (and for Lasso/ElasticNet: max_iter; for ElasticNet: l1_ratio).
    LGBM uses a grid tuned for less overfitting (shallower trees, stronger regularization).

    Parameters
    ----------
    base_key : str
        One of 'Ridge', 'Lasso', 'ElasticNet', 'LGBM'.

    Returns
    -------
    dict[str, list]
        param_grid suitable for run_uplift_hp_grid_search (keys = param names, values = lists).
    """
    if base_key == "Ridge":
        return {"alpha": [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]}
    if base_key == "Lasso":
        return {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
            "max_iter": [5000, 10000],
        }
    if base_key == "ElasticNet":
        return {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10, 100],
            "l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
            "max_iter": [5000, 10000],
        }
    if base_key == "LGBM":
        return {
            "max_depth": [3, 4],
            "learning_rate": [0.05, 0.1],
            "min_child_samples": [20, 40, 60],
            "reg_alpha": [1, 10, 50],
            "reg_lambda": [5, 10, 20],
            "bagging_fraction": [0.8, 1.0],
            "bagging_freq": [1],
            "feature_fraction": [0.8, 1.0],
            "min_gain_to_split": [0.1, 0.2, 0.3],
        }
    raise ValueError(f"Unknown base_key for HP grid: {base_key!r}. Use one of Ridge, Lasso, ElasticNet, LGBM.")


def run_uplift_hp_grid_search(
    X: pd.DataFrame,
    y: np.ndarray,
    treatment: np.ndarray,
    meta_key: str,
    base_key: str,
    param_grid: dict[str, list],
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    scale_pos_weight: float = 1.0,
    n_curve_points: int = 100,
    uplift_k_fracs: list[float] | None = None,
) -> list[dict]:
    """Run grid search over base-learner hyperparameters for one meta-learner.

    For every combination in *param_grid*, runs stratified K-fold CV and
    records per-fold AUUC, Qini, and uplift@k metrics.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric).
    y : np.ndarray
        Binary outcome (churn 0/1).
    treatment : np.ndarray
        Treatment indicator (1 = treated, 0 = control).
    meta_key : str
        'S' or 'T'.
    base_key : str
        'LGBM' or 'XGB'.
    param_grid : dict[str, list]
        e.g. ``{'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1]}``.
    n_splits : int
        Number of CV folds.
    random_state : int
        Seed for StratifiedKFold.
    scale_pos_weight : float
        Passed to XGBoost base learner (ignored for LGBM).
    n_curve_points : int
        Points for the uplift curve (AUUC computation).
    uplift_k_fracs : list[float] | None
        Fractions for uplift@k.  Default [0.05, 0.1, 0.2].

    Returns
    -------
    list[dict]
        One dict per (param_combo, fold) with keys: params_str, fold, auuc,
        qini, uplift_at_k_*, plus each individual hyperparameter.
    """
    import itertools

    if uplift_k_fracs is None:
        uplift_k_fracs = [0.05, 0.1, 0.2]

    # --- Validate y and treatment (same logic as run_uplift_cv) ---
    y_ = np.asarray(y, dtype=np.float64)
    t_ = np.asarray(treatment, dtype=np.float64)
    valid = np.isfinite(y_) & np.isin(t_, (0.0, 1.0)) & np.isin(y_, (0.0, 1.0))
    if not np.all(valid):
        X = X.loc[valid] if isinstance(X, pd.DataFrame) else X[valid]
        y = np.asarray(y_[valid], dtype=np.float64)
        treatment = np.asarray(t_[valid], dtype=np.float64)
        y = (y > 0.5).astype(np.int64)
        treatment = (treatment > 0.5).astype(np.int64)
    else:
        y = (np.asarray(y, dtype=np.float64) > 0.5).astype(np.int64)
        treatment = (np.asarray(treatment, dtype=np.float64) > 0.5).astype(np.int64)

    # Build all combinations from the parameter grid
    param_names = sorted(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    combos = [dict(zip(param_names, vals)) for vals in itertools.product(*param_values)]
    n_combos = len(combos)

    # Prepare StratifiedKFold (same stratification as Section 5)
    from sklearn.model_selection import StratifiedKFold

    stratify = 2 * treatment + y
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(X, stratify))

    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    results: list[dict] = []

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
            module="sklearn.utils.validation",
        )
        for combo_idx, params in enumerate(combos):
            # Readable label for this combination
            params_str = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
            for fold, (train_idx, val_idx) in enumerate(splits):
                X_tr = X_df.iloc[train_idx]
                X_val = X_df.iloc[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                t_tr, t_val = treatment[train_idx], treatment[val_idx]

                # For XGB, set n_estimators via early stopping on this fold's train set (same in all flows).
                params_fold = dict(params)
                if base_key == "XGB":
                    n_max = params_fold.get("n_estimators", 800)
                    best_n = _early_stopping_n_estimators_single(
                        X_tr, y_tr, t_tr,
                        base_params=params_fold,
                        scale_pos_weight=scale_pos_weight,
                        random_state=random_state,
                        n_estimators_max=n_max,
                    )
                    params_fold["n_estimators"] = best_n

                model = build_model(
                    meta_key, base_key, scale_pos_weight, base_params=params_fold
                )
                model.fit(X_tr, t_tr, y_tr)
                pred = model.predict(X_val)

                # Same sign-convention split as run_uplift_cv
                raw_pred = np.asarray(pred).ravel()
                neg_pred = -raw_pred

                ks, uplift_vals = uplift_curve(
                    y_val, t_val, neg_pred, n_points=n_curve_points
                )
                auuc = approx_auuc(ks, uplift_vals)
                # Churn → retention for Qini (see run_uplift_cv for rationale)
                qini = qini_coefficient(1 - y_val, t_val, neg_pred)

                row: dict = {
                    "params_str": params_str,
                    "fold": fold,
                    "auuc": auuc,
                    "qini": qini,
                }
                # Store each HP as its own column for easy filtering
                for k, v in params.items():
                    row[k] = v
                for frac in uplift_k_fracs:
                    key = f"uplift_at_k_{int(frac * 100)}"
                    row[key] = uplift_at_k(y_val, t_val, neg_pred, frac)
                results.append(row)

            # Progress feedback every 10 combos
            if (combo_idx + 1) % 10 == 0 or combo_idx == n_combos - 1:
                print(
                    f"  Grid search: {combo_idx + 1}/{n_combos} combos done "
                    f"({(combo_idx + 1) * n_splits} fits)"
                )

    return results


def build_hp_grid_search_table(
    grid_results: list[dict],
    metric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate grid-search results into mean ± std per parameter combo.

    Parameters
    ----------
    grid_results : list[dict]
        Output from :func:`run_uplift_hp_grid_search`.
    metric_cols : list[str] | None
        Metrics to aggregate.  Default: auuc, qini, uplift_at_k_5/10/20.

    Returns
    -------
    pd.DataFrame
        One row per combo, sorted by descending auuc_mean.
    """
    if metric_cols is None:
        metric_cols = ["auuc", "qini", "uplift_at_k_5", "uplift_at_k_10", "uplift_at_k_20"]
    df = pd.DataFrame(grid_results)
    present = [m for m in metric_cols if m in df.columns]
    agg_dict: dict = {}
    for m in present:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")
    summary = df.groupby("params_str").agg(**agg_dict).reset_index()
    # Sort by primary metric descending
    if "auuc_mean" in summary.columns:
        summary = summary.sort_values("auuc_mean", ascending=False).reset_index(drop=True)
    return summary


def compute_early_stopping_n_estimators(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
    base_params: dict,
    scale_pos_weight: float = 1.0,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    inner_val_frac: float = 0.2,
    early_stopping_rounds: int = 15,
    n_estimators_max: int = 800,
) -> list[int]:
    """Compute per-fold best iteration via early stopping for S-learner + XGBoost.

    Uses the same CV splits as the grid search. Within each fold's training set,
    an inner train/validation split is used so the base XGB is fit with
    eval_set and early_stopping_rounds. Returns the list of best_iteration
    per fold; the caller typically uses median(best_rounds) as n_estimators
    when training the final model on 100% of the data.

    Only supports S-learner with XGBoost (meta_key='S', base_key='XGB'). The
    internal design matrix is [treatment_indicator | X] as in CausalML.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix (same as used for uplift CV).
    y : np.ndarray
        Binary outcome (churn 0/1).
    treatment : np.ndarray
        Treatment indicator (1 = treated, 0 = control).
    base_params : dict
        Base-learner hyperparameters (e.g. from grid search best). Must include
        max_depth, learning_rate, etc. n_estimators is overridden by
        n_estimators_max for the inner fits.
    scale_pos_weight : float
        Passed to XGBRegressor (for class balance).
    n_splits : int
        Number of CV folds (must match Section 5/6).
    random_state : int
        Seed for StratifiedKFold and inner train_test_split.
    inner_val_frac : float
        Fraction of each fold's training set used as validation for early stopping.
    early_stopping_rounds : int
        Stop if validation metric does not improve for this many rounds.
    n_estimators_max : int
        Maximum number of trees for the inner XGB fit.

    Returns
    -------
    list[int]
        One best_iteration per fold. Use e.g. int(np.median(result)) for final
        n_estimators.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    if XGBRegressor is None:
        raise ImportError("xgboost is required for early stopping")
    y_ = np.asarray(y, dtype=np.float64)
    t_ = np.asarray(treatment, dtype=np.float64)
    valid = np.isfinite(y_) & np.isin(t_, (0.0, 1.0)) & np.isin(y_, (0.0, 1.0))
    if not np.all(valid):
        X = X.loc[valid] if isinstance(X, pd.DataFrame) else X[valid]
        y_ = np.asarray(y_[valid], dtype=np.float64)
        t_ = np.asarray(t_[valid], dtype=np.float64)
    y = (y_ > 0.5).astype(np.int64)
    treatment = (t_ > 0.5).astype(np.int64)
    X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    X_arr = np.asarray(X_df, dtype=np.float64)
    stratify = 2 * treatment + y
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(X_arr, stratify))
    best_rounds: list[int] = []
    params = dict(base_params)
    params["n_estimators"] = n_estimators_max
    for fold, (train_idx, _val_idx) in enumerate(splits):
        X_tr_full = X_arr[train_idx]
        y_tr_full = y[train_idx]
        t_tr_full = treatment[train_idx]
        stratify_inner = 2 * t_tr_full + y_tr_full
        # Inner split: stratify only when every stratum has at least 2 samples
        unique_inner, counts_inner = np.unique(stratify_inner, return_counts=True)
        can_stratify_inner = np.all(counts_inner >= 2)
        if can_stratify_inner:
            train_inner_idx, val_inner_idx = train_test_split(
                np.arange(len(train_idx)),
                test_size=inner_val_frac,
                stratify=stratify_inner,
                random_state=random_state,
            )
        else:
            train_inner_idx, val_inner_idx = train_test_split(
                np.arange(len(train_idx)),
                test_size=inner_val_frac,
                random_state=random_state,
            )
        X_tr = X_tr_full[train_inner_idx]
        y_tr = y_tr_full[train_inner_idx]
        t_tr = t_tr_full[train_inner_idx]
        X_val = X_tr_full[val_inner_idx]
        y_val = y_tr_full[val_inner_idx]
        t_val = t_tr_full[val_inner_idx]
        w_tr = (t_tr == 1).astype(np.float64).reshape(-1, 1)
        w_val = (t_val == 1).astype(np.float64).reshape(-1, 1)
        X_tr_new = np.hstack([w_tr, X_tr])
        X_val_new = np.hstack([w_val, X_val])
        # XGBoost 2.x/3.x: early_stopping_rounds is a constructor arg, not fit()
        xgb = XGBRegressor(
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            verbosity=0,
            early_stopping_rounds=early_stopping_rounds,
            **params,
        )
        xgb.fit(
            X_tr_new,
            y_tr,
            eval_set=[(X_val_new, y_val)],
            verbose=False,
        )
        best = getattr(xgb, "best_iteration", None) or getattr(
            xgb, "best_ntree_limit", None
        )
        if best is not None:
            best_rounds.append(int(best))
        else:
            best_rounds.append(n_estimators_max)
    return best_rounds


def _early_stopping_n_estimators_single(
    X_tr: np.ndarray | pd.DataFrame,
    y_tr: np.ndarray,
    t_tr: np.ndarray,
    base_params: dict,
    scale_pos_weight: float,
    random_state: int,
    inner_val_frac: float = 0.2,
    early_stopping_rounds: int = 15,
    n_estimators_max: int | None = None,
) -> int:
    """Compute best n_estimators for a single training set via inner train/val early stopping.

    Used whenever an S+XGB model is fitted (CV, grid search, final fit) so that
    early stopping is applied consistently. Does one inner stratified split,
    builds S-learner design matrix [treatment | X], fits XGB with eval_set,
    returns best_iteration (or best_ntree_limit).

    Parameters
    ----------
    X_tr : np.ndarray or pd.DataFrame
        Training feature matrix for this fold/split.
    y_tr, t_tr : np.ndarray
        Training outcome and treatment.
    base_params : dict
        Base-learner hyperparameters. n_estimators is set to n_estimators_max
        for the inner fit; other keys passed to XGBRegressor.
    scale_pos_weight : float
        Passed to XGBRegressor.
    random_state : int
        Seed for inner train_test_split.
    inner_val_frac : float
        Fraction of X_tr used as validation for early stopping.
    early_stopping_rounds : int
        Stop if validation metric does not improve for this many rounds.
    n_estimators_max : int or None
        Max trees for inner fit. If None, uses base_params.get("n_estimators", 800).

    Returns
    -------
    int
        Best iteration for this training set.
    """
    from sklearn.model_selection import train_test_split

    if XGBRegressor is None:
        raise ImportError("xgboost is required for early stopping")
    X_arr = np.asarray(X_tr, dtype=np.float64)
    n_max = n_estimators_max if n_estimators_max is not None else base_params.get("n_estimators", 800)
    params = dict(base_params)
    params["n_estimators"] = n_max
    stratify_inner = 2 * t_tr + y_tr
    # Stratification requires at least 2 samples per class in both splits
    unique, counts = np.unique(stratify_inner, return_counts=True)
    can_stratify = np.all(counts >= 2)
    if can_stratify:
        train_idx, val_idx = train_test_split(
            np.arange(len(y_tr)),
            test_size=inner_val_frac,
            stratify=stratify_inner,
            random_state=random_state,
        )
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(y_tr)),
            test_size=inner_val_frac,
            random_state=random_state,
        )
    X_tr_inner = X_arr[train_idx]
    y_tr_inner = y_tr[train_idx]
    t_tr_inner = t_tr[train_idx]
    X_val_inner = X_arr[val_idx]
    y_val_inner = y_tr[val_idx]
    t_val_inner = t_tr[val_idx]
    w_tr = (t_tr_inner == 1).astype(np.float64).reshape(-1, 1)
    w_val = (t_val_inner == 1).astype(np.float64).reshape(-1, 1)
    X_tr_new = np.hstack([w_tr, X_tr_inner])
    X_val_new = np.hstack([w_val, X_val_inner])
    # XGBoost 2.x/3.x: early_stopping_rounds is a constructor arg, not fit()
    xgb = XGBRegressor(
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
        early_stopping_rounds=early_stopping_rounds,
        **params,
    )
    xgb.fit(
        X_tr_new,
        y_tr_inner,
        eval_set=[(X_val_new, y_val_inner)],
        verbose=False,
    )
    best = getattr(xgb, "best_iteration", None) or getattr(xgb, "best_ntree_limit", None)
    return int(best) if best is not None else n_max


def fit_final_slearner(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
    meta_key: str,
    base_key: str,
    base_params: dict,
    scale_pos_weight: float = 1.0,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    early_stopping_rounds: int = 15,
    inner_val_frac: float = 0.2,
    n_estimators_max: int = 800,
):
    """Build and fit the final meta-learner on the full dataset, with early stopping for XGB.

    For base_key == 'XGB', runs compute_early_stopping_n_estimators to get
    per-fold best iterations, sets n_estimators = median(best_rounds), then
    builds and fits. For linear models (Ridge/Lasso/ElasticNet) early
    stopping is skipped — the Pipeline inside build_model handles NaN
    imputation and standardisation automatically.

    Parameters
    ----------
    X, y, treatment : feature matrix, outcome, treatment (same as elsewhere).
    meta_key, base_key : e.g. 'S', 'XGB', 'Ridge', 'Lasso', 'ElasticNet'.
    base_params : dict
        Best hyperparameters from grid search (or defaults). Not mutated.
    scale_pos_weight, n_splits, random_state : as in other flows.
    early_stopping_rounds, inner_val_frac, n_estimators_max : passed to early stopping.

    Returns
    -------
    Fitted CausalML meta-learner (e.g. BaseSRegressor).
    """
    base_params = dict(base_params)
    # Early stopping only applies to XGB; linear models and LGBM skip this
    if base_key == "XGB":
        best_rounds = compute_early_stopping_n_estimators(
            X, y, treatment,
            base_params=base_params,
            scale_pos_weight=scale_pos_weight,
            n_splits=n_splits,
            random_state=random_state,
            inner_val_frac=inner_val_frac,
            early_stopping_rounds=early_stopping_rounds,
            n_estimators_max=n_estimators_max,
        )
        base_params["n_estimators"] = int(np.median(best_rounds))
    model = build_model(meta_key, base_key, scale_pos_weight, base_params=base_params)
    X_arr = np.asarray(X, dtype=np.float64) if not isinstance(X, np.ndarray) else X
    # CausalML/LGBM can trigger "X does not have valid feature names" when design matrix is numpy
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
            module="sklearn.utils.validation",
        )
        model.fit(X=X_arr, treatment=treatment, y=y)
    return model


def get_qini_curves_top_hp_combos(
    X: pd.DataFrame,
    y: np.ndarray,
    treatment: np.ndarray,
    grid_summary: pd.DataFrame,
    grid_results: list[dict],
    meta_key: str,
    base_key: str,
    param_grid: dict[str, list],
    top_n: int = 10,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    scale_pos_weight: float = 1.0,
    n_points: int = 101,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Qini curves for top_n hyperparameter combos, mean (and std) across CV folds.

    Re-fits only the top_n combos over the same CV splits and computes Qini curve
    per fold, then aggregates. General design allows single test set in future.

    Internally converts churn to retention (``1 − y``) and negates CATE before
    calling CausalML (same convention as ``get_validation_qini_curves``).

    Parameters
    ----------
    X, y, treatment : as in run_uplift_hp_grid_search (y is churn 0/1).
    grid_summary : pd.DataFrame
        Output from build_hp_grid_search_table (used to get top params_str by auuc_mean).
    grid_results : list[dict]
        Output from run_uplift_hp_grid_search (used to recover param dict per params_str).
    meta_key, base_key, param_grid : as in run_uplift_hp_grid_search.
    top_n : int
        Number of top combos. Default 10.
    n_splits, random_state, scale_pos_weight, n_points : as in get_validation_qini_curves.

    Returns
    -------
    dict[params_str, (fractions, mean_vals, std_vals)]
    """
    from sklearn.model_selection import StratifiedKFold

    param_names = sorted(param_grid.keys())
    top_params_str = grid_summary.head(top_n)["params_str"].tolist()
    # Recover param dict for each params_str from first occurrence in grid_results
    df_res = pd.DataFrame(grid_results)
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    stratify = 2 * treatment + y
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(X, stratify))
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
            module="sklearn.utils.validation",
        )
        for params_str in top_params_str:
            row = df_res[df_res["params_str"] == params_str].iloc[0]
            params = {k: row[k] for k in param_names if k in row}
            curves_fold = []
            for train_idx, val_idx in splits:
                X_tr = X_df.iloc[train_idx]
                X_val = X_df.iloc[val_idx]
                y_val = y[val_idx]
                t_val = treatment[val_idx]
                params_fold = dict(params)
                if base_key == "XGB":
                    n_max = params_fold.get("n_estimators", 800)
                    best_n = _early_stopping_n_estimators_single(
                        X_tr, y[train_idx], treatment[train_idx],
                        base_params=params_fold,
                        scale_pos_weight=scale_pos_weight,
                        random_state=random_state,
                        n_estimators_max=n_max,
                    )
                    params_fold["n_estimators"] = best_n
                model = build_model(meta_key, base_key, scale_pos_weight, base_params=params_fold)
                model.fit(X_tr, treatment[train_idx], y[train_idx])
                pred = model.predict(X_val)
                raw_pred = np.asarray(pred).ravel()
                # Churn → retention for Qini (see get_validation_qini_curves for rationale)
                neg_pred = -raw_pred
                retention = 1 - y_val
                frac, vals = get_qini_curve_single(retention, t_val, neg_pred, n_points=n_points)
                curves_fold.append((frac, vals))
            frac, mean_vals, std_vals = aggregate_qini_curves_over_folds(curves_fold, n_points=n_points)
            result[params_str] = (frac, mean_vals, std_vals)

    return result


def save_hp_grid_search_report(
    grid_summary: pd.DataFrame,
    table_path: str | Path,
    figure_dir: str | Path,
    top_n_plot: int = 10,
    qini_curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Save grid-search table (CSV) and optionally Qini curves figure to disk.

    Parameters
    ----------
    grid_summary : pd.DataFrame
        Output from :func:`build_hp_grid_search_table`.
    table_path : str | Path
        CSV file path for the full table.
    figure_dir : str | Path
        Directory for saved figures.
    top_n_plot : int
        Unused if qini_curves provided; kept for API compatibility.
    qini_curves : dict or None
        If provided (e.g. from get_qini_curves_top_hp_combos), save Qini curve plot.
    """
    table_path = Path(table_path)
    figure_dir = Path(figure_dir)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    grid_summary.to_csv(table_path, index=False)
    if qini_curves:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_qini_curves_comparison(
            qini_curves, ax=ax, save_path=figure_dir / "hp_grid_search_qini_curves.png"
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Final Model — training, scoring, and test-set deliverables (Section 7)
# ---------------------------------------------------------------------------


def plot_cumulative_uplift_curve(
    ks: np.ndarray,
    uplift_vals: np.ndarray,
    ax: plt.Axes | None = None,
    label: str = "Model",
    show_random: bool = True,
    title: str = "Cumulative uplift curve",
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot a cumulative (realised) uplift curve.

    Generic: accepts (ks, uplift_vals) from :func:`uplift_curve` or any
    function that returns (fractions, values). Useful for train evaluation
    and later for SHAP-based or business-metric curves.

    Parameters
    ----------
    ks : np.ndarray
        Fraction of population targeted (e.g. 0.01 … 1.0).
    uplift_vals : np.ndarray
        Realised uplift at each fraction (e.g. churn_control − churn_treated).
    ax : plt.Axes or None
        Axes to plot on; created if None.
    label : str
        Legend label for the model curve.
    show_random : bool
        If True, draw a horizontal dashed line at the overall uplift (last point).
    title : str
        Plot title.
    save_path : str, Path, or None
        If provided, save figure to this path.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    valid = ~np.isnan(uplift_vals)
    ax.plot(ks[valid], uplift_vals[valid], linewidth=2, label=label, color="#1f77b4")
    if show_random and valid.sum() > 0:
        # Random targeting = constant uplift equal to overall uplift (at 100%)
        overall = uplift_vals[valid][-1] if valid.any() else 0.0
        ax.axhline(overall, linestyle="--", color="gray", linewidth=1.5, label="Random baseline")
    ax.set_xlabel("Proportion targeted")
    ax.set_ylabel("Realised uplift (control − treated)")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    set_axes_clear(ax, x_axis_at_zero=False)
    plt.tight_layout()
    if save_path is not None:
        ax.figure.savefig(Path(save_path), bbox_inches="tight", dpi=150)
    return ax


def plot_uplift_by_decile(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 10,
    ax: plt.Axes | None = None,
    title: str = "Uplift by decile",
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Bar chart of realised uplift per decile (ranked by predicted uplift).

    Sorts population by *uplift_scores* (descending), splits into *n_bins*
    equal-sized bins, and computes realised uplift (control rate − treated rate)
    in each bin. Requires labelled data (y, treatment).

    Generic: can be reused with any score (e.g. SHAP-based) as long as
    higher score = "target first".

    Parameters
    ----------
    y : np.ndarray
        Binary outcome (e.g. churn 0/1).
    treatment : np.ndarray
        Treatment indicator (1 = treated, 0 = control).
    uplift_scores : np.ndarray
        Predicted uplift; higher = target first (for churn pass -CATE).
    n_bins : int
        Number of bins (default 10 = deciles).
    ax : plt.Axes or None
        Axes to plot on; created if None.
    title : str
        Plot title.
    save_path : str, Path, or None
        If provided, save figure to this path.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(-uplift_scores)
    y_s, t_s = y[order], treatment[order]
    bin_size = max(1, len(y_s) // n_bins)
    labels, uplifts = [], []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(y_s)
        yb, tb = y_s[start:end], t_s[start:end]
        nt, nc = (tb == 1).sum(), (tb == 0).sum()
        if nt == 0 or nc == 0:
            uplifts.append(np.nan)
        else:
            uplifts.append(float(yb[tb == 0].mean() - yb[tb == 1].mean()))
        labels.append(f"D{i + 1}")
    # Color bars: positive green, negative red
    colors = ["#2ca02c" if u >= 0 else "#d62728" for u in uplifts]
    ax.bar(labels, uplifts, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Decile (D1 = highest predicted uplift)")
    ax.set_ylabel("Realised uplift (control − treated)")
    ax.set_title(title)
    set_axes_clear(ax, x_axis_at_zero=False)
    plt.tight_layout()
    if save_path is not None:
        ax.figure.savefig(Path(save_path), bbox_inches="tight", dpi=150)
    return ax


def compute_train_holdout_uplift_data(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
    meta_key: str,
    base_key: str,
    base_params: dict,
    scale_pos_weight: float = 1.0,
    train_frac: float = 0.8,
    random_state: int = RANDOM_STATE,
    n_points: int = 100,
    **fit_final_kw: object,
) -> dict:
    """Split data into train/holdout (one stratified 80/20), fit on train, predict on both.

    Use this to evaluate realised uplift on an unseen fold when test has no labels.
    Fits a model on the train portion only, then computes uplift curves and decile
    data for both train and holdout so you can plot train vs holdout.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix (same as for fit_final_slearner).
    y : np.ndarray
        Binary outcome (e.g. churn 0/1).
    treatment : np.ndarray
        Treatment indicator (1 = treated, 0 = control).
    meta_key, base_key : str
        Same as fit_final_slearner (e.g. "S", "LGBM").
    base_params : dict
        Hyperparameters for the base learner (e.g. BEST_HP from grid search).
    scale_pos_weight : float
        Passed to fit_final_slearner.
    train_frac : float
        Fraction used for training (default 0.8); holdout is 1 - train_frac.
    random_state : int
        For reproducible stratified split.
    n_points : int
        Number of points for uplift_curve (default 100).
    **fit_final_kw : optional
        Passed to fit_final_slearner (n_splits, early_stopping_rounds, etc.).

    Returns
    -------
    dict with keys:
        model_80 : fitted meta-learner on train portion
        y_tr, treatment_tr, uplift_scores_tr : train portion labels and scores
        y_ho, treatment_ho, uplift_scores_ho : holdout portion labels and scores
        ks_tr, uplift_vals_tr : cumulative uplift curve (train)
        ks_ho, uplift_vals_ho : cumulative uplift curve (holdout)
    """
    from sklearn.model_selection import StratifiedKFold

    # One stratified split: use first fold of 5-fold so train ≈ 80%, holdout ≈ 20%
    n_splits = max(2, int(round(1 / (1 - train_frac))))  # 5 for train_frac=0.8
    stratify = 2 * np.asarray(treatment, dtype=np.int64) + np.asarray(y, dtype=np.int64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(X, stratify))
    train_idx, holdout_idx = splits[0]

    if isinstance(X, pd.DataFrame):
        X_tr, X_ho = X.iloc[train_idx], X.iloc[holdout_idx]
    else:
        X_tr, X_ho = X[train_idx], X[holdout_idx]
    y_tr = y[train_idx]
    y_ho = y[holdout_idx]
    treatment_tr = treatment[train_idx]
    treatment_ho = treatment[holdout_idx]

    # Fit on train portion only (same API as fit_final_slearner)
    model_80 = fit_final_slearner(
        X_tr, y_tr, treatment_tr,
        meta_key=meta_key,
        base_key=base_key,
        base_params=dict(base_params),
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        **fit_final_kw,
    )

    # Predict CATE on both portions; uplift score = -CATE for churn (higher = better)
    cate_tr = np.asarray(model_80.predict(X=np.asarray(X_tr), treatment=treatment_tr)).ravel()
    cate_ho = np.asarray(model_80.predict(X=np.asarray(X_ho), treatment=treatment_ho)).ravel()
    uplift_scores_tr = -cate_tr
    uplift_scores_ho = -cate_ho

    # Cumulative uplift curves
    ks_tr, uplift_vals_tr = uplift_curve(y_tr, treatment_tr, uplift_scores_tr, n_points=n_points)
    ks_ho, uplift_vals_ho = uplift_curve(y_ho, treatment_ho, uplift_scores_ho, n_points=n_points)

    return {
        "model_80": model_80,
        "y_tr": y_tr,
        "treatment_tr": treatment_tr,
        "uplift_scores_tr": uplift_scores_tr,
        "y_ho": y_ho,
        "treatment_ho": treatment_ho,
        "uplift_scores_ho": uplift_scores_ho,
        "ks_tr": ks_tr,
        "uplift_vals_tr": uplift_vals_tr,
        "ks_ho": ks_ho,
        "uplift_vals_ho": uplift_vals_ho,
    }


def plot_cumulative_uplift_curve_train_holdout(
    ks_tr: np.ndarray,
    uplift_vals_tr: np.ndarray,
    ks_ho: np.ndarray,
    uplift_vals_ho: np.ndarray,
    label_train: str = "Train (80%)",
    label_holdout: str = "Holdout (20%)",
    title: str = "Cumulative uplift curve — train vs holdout",
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot cumulative realised uplift for train and holdout on the same axes.

    Use with outputs from :func:`compute_train_holdout_uplift_data`. Draws two
    lines (train and holdout) so you can compare generalization.

    Parameters
    ----------
    ks_tr, uplift_vals_tr : np.ndarray
        From uplift_curve on train portion.
    ks_ho, uplift_vals_ho : np.ndarray
        From uplift_curve on holdout portion.
    label_train, label_holdout : str
        Legend labels.
    title : str
        Plot title.
    ax : plt.Axes or None
        Axes to plot on; created if None.
    save_path : str, Path, or None
        If set, save figure.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    valid_tr = ~np.isnan(uplift_vals_tr)
    valid_ho = ~np.isnan(uplift_vals_ho)
    ax.plot(ks_tr[valid_tr], uplift_vals_tr[valid_tr], linewidth=2, label=label_train, color="#1f77b4")
    ax.plot(ks_ho[valid_ho], uplift_vals_ho[valid_ho], linewidth=2, label=label_holdout, color="#ff7f0e")
    if valid_tr.sum() > 0:
        overall_tr = uplift_vals_tr[valid_tr][-1]
        ax.axhline(overall_tr, linestyle="--", color="gray", linewidth=1, alpha=0.7, label="Random baseline")
    ax.set_xlabel("Proportion targeted")
    ax.set_ylabel("Realised uplift (control − treated)")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    set_axes_clear(ax, x_axis_at_zero=False)
    plt.tight_layout()
    if save_path is not None:
        ax.figure.savefig(Path(save_path), bbox_inches="tight", dpi=150)
    return ax


def plot_uplift_by_decile_train_holdout(
    y_tr: np.ndarray,
    treatment_tr: np.ndarray,
    uplift_scores_tr: np.ndarray,
    y_ho: np.ndarray,
    treatment_ho: np.ndarray,
    uplift_scores_ho: np.ndarray,
    label_train: str = "Train (80%)",
    label_holdout: str = "Holdout (20%)",
    title: str = "Realised uplift by decile — train vs holdout",
    n_bins: int = 10,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Two side-by-side decile bar charts (train and holdout) for comparison.

    Use with outputs from :func:`compute_train_holdout_uplift_data`. Each panel
    shows realised uplift per decile; compare to see if ranking generalizes.

    Parameters
    ----------
    y_tr, treatment_tr, uplift_scores_tr : np.ndarray
        Train portion labels, treatment, and uplift scores.
    y_ho, treatment_ho, uplift_scores_ho : np.ndarray
        Holdout portion labels, treatment, and uplift scores.
    label_train, label_holdout : str
        Subplot titles.
    title : str
        Figure suptitle.
    n_bins : int
        Number of bins (default 10 = deciles).
    save_path : str, Path, or None
        If set, save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax1, ax2 : matplotlib.axes.Axes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_uplift_by_decile(y_tr, treatment_tr, uplift_scores_tr, n_bins=n_bins, ax=ax1, title=label_train)
    plot_uplift_by_decile(y_ho, treatment_ho, uplift_scores_ho, n_bins=n_bins, ax=ax2, title=label_holdout)
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(Path(save_path), bbox_inches="tight", dpi=150)
    return fig, (ax1, ax2)


# --- Test-set visualisations (prediction-only, no labels needed) -----------


def plot_segment_counts(
    segments: np.ndarray,
    ax: plt.Axes | None = None,
    as_percent: bool = True,
    title: str = "Population segments",
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Bar chart of count or percentage per uplift segment.

    Generic: works with any categorical segment array (e.g. from
    :func:`assign_segments` or a SHAP-based segmentation).

    Parameters
    ----------
    segments : np.ndarray of str
        Segment label per observation (e.g. "Persuadables", "Sure Things", …).
    ax : plt.Axes or None
        Axes to plot on; created if None.
    as_percent : bool
        If True, plot percentage; otherwise raw counts.
    title : str
        Plot title.
    save_path : str, Path, or None
        If provided, save figure to this path.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    # Fixed display order: Persuadables first, Do-Not-Disturb last
    seg_order = ["Persuadables", "Sure Things", "Lost Causes", "Do-Not-Disturb"]
    seg_series = pd.Series(segments)
    counts = seg_series.value_counts()
    vals = [counts.get(s, 0) for s in seg_order]
    if as_percent:
        total = max(1, len(segments))
        vals = [v / total * 100 for v in vals]
        ylabel = "% of population"
    else:
        ylabel = "Count"
    seg_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    bars = ax.bar(seg_order, vals, color=seg_colors, edgecolor="white", linewidth=0.5)
    # Annotate each bar
    for bar, v in zip(bars, vals):
        fmt = f"{v:.1f}%" if as_percent else f"{int(v):,}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals) * 0.02,
            fmt,
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    set_axes_clear(ax, x_axis_at_zero=False)
    plt.tight_layout()
    if save_path is not None:
        ax.figure.savefig(Path(save_path), bbox_inches="tight", dpi=150)
    return ax


def plot_uplift_by_segment(
    segments: np.ndarray,
    uplift_scores: np.ndarray,
    ax: plt.Axes | None = None,
    title: str = "Predicted uplift by segment",
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Box plot of predicted uplift (or any score) per segment.

    Under formal segment definitions, Sure Things and Lost Causes have |uplift| ≤ ε
    (≈ 0); Persuadables > 0, Do-Not-Disturb < 0. Generic: accepts any score + segments.

    Parameters
    ----------
    segments : np.ndarray of str
        Segment label per observation.
    uplift_scores : np.ndarray
        Predicted uplift (or any numeric score, e.g. SHAP value).
    ax : plt.Axes or None
        Axes to plot on; created if None.
    title : str
        Plot title.
    save_path : str, Path, or None
        If provided, save figure to this path.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    seg_order = ["Persuadables", "Sure Things", "Lost Causes", "Do-Not-Disturb"]
    seg_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    data_per_seg = [uplift_scores[segments == s] for s in seg_order]
    bp = ax.boxplot(
        data_per_seg,
        labels=seg_order,
        patch_artist=True,
        widths=0.5,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], seg_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Predicted uplift (−CATE)")
    ax.set_title(title)
    set_axes_clear(ax, x_axis_at_zero=False)
    plt.tight_layout()
    if save_path is not None:
        ax.figure.savefig(Path(save_path), bbox_inches="tight", dpi=150)
    return ax


def plot_cumulative_uplift_by_segment(
    segments: np.ndarray,
    uplift_scores: np.ndarray,
    n_points: int = 100,
    ax: plt.Axes | None = None,
    title: str = "Cumulative predicted uplift by segment",
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Four curves: cumulative predicted uplift contribution by segment.

    Segments follow formal definitions (Persuadables = positive uplift, etc.).
    Sorts population by *uplift_scores* (descending). At each proportion
    targeted x, computes the cumulative **sum** of predicted uplift in the
    top x%, split by segment. The four curves show how much each segment
    contributes as the targeting budget expands.

    Generic: works with any score + segment arrays (e.g. SHAP-based).

    Parameters
    ----------
    segments : np.ndarray of str
        Segment label per observation.
    uplift_scores : np.ndarray
        Predicted uplift (higher = target first).
    n_points : int
        Number of evaluation points along proportion axis.
    ax : plt.Axes or None
        Axes to plot on; created if None.
    title : str
        Plot title.
    save_path : str, Path, or None
        If provided, save figure to this path.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    seg_order = ["Persuadables", "Sure Things", "Lost Causes", "Do-Not-Disturb"]
    seg_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    # Sort by descending score
    order = np.argsort(-uplift_scores)
    scores_sorted = uplift_scores[order]
    segs_sorted = segments[order]
    n_total = len(scores_sorted)
    fracs = np.linspace(0.01, 1.0, n_points)
    # Build cumulative sums per segment
    cum_by_seg = {s: np.zeros(n_points) for s in seg_order}
    for i, frac in enumerate(fracs):
        n = max(1, int(n_total * frac))
        sc_slice = scores_sorted[:n]
        seg_slice = segs_sorted[:n]
        for s in seg_order:
            mask = seg_slice == s
            cum_by_seg[s][i] = float(sc_slice[mask].sum()) if mask.any() else 0.0
    # Normalise by population size so y-axis is "mean cumulative uplift per person"
    for s in seg_order:
        cum_by_seg[s] /= max(1, n_total)
    for s, color in zip(seg_order, seg_colors):
        ax.plot(fracs, cum_by_seg[s], linewidth=2, label=s, color=color)
    ax.set_xlabel("Proportion targeted (by predicted uplift)")
    ax.set_ylabel("Cumulative predicted uplift (per person)")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    set_axes_clear(ax, x_axis_at_zero=False)
    plt.tight_layout()
    if save_path is not None:
        ax.figure.savefig(Path(save_path), bbox_inches="tight", dpi=150)
    return ax


def export_top_fraction(
    member_ids: np.ndarray | pd.Series,
    uplift_scores: np.ndarray,
    fraction: float = 0.10,
    out_path: str | Path = "top_n_members_outreach.csv",
) -> pd.DataFrame:
    """Export the top fraction of users by predicted uplift to CSV.

    Generic: accepts any ID array and numeric score. Can be reused with
    SHAP-based or business-metric scores later.

    Parameters
    ----------
    member_ids : array-like
        User identifiers (e.g. member_id from test features).
    uplift_scores : np.ndarray
        Predicted uplift (higher = more benefit, i.e. −CATE for churn).
    fraction : float
        Fraction of population to export (e.g. 0.10 for top 10%).
    out_path : str or Path
        File path for the output CSV.

    Returns
    -------
    pd.DataFrame
        The exported table (member_id, prioritization_score, rank), sorted
        by rank ascending.
    """
    n = max(1, int(len(uplift_scores) * fraction))
    order = np.argsort(-np.asarray(uplift_scores))[:n]
    df = pd.DataFrame({
        "member_id": np.asarray(member_ids)[order],
        "prioritization_score": np.asarray(uplift_scores)[order],
    })
    df = df.sort_values("prioritization_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


# ===========================================================================
# Section 8 — SHAP interpretability helpers
# ===========================================================================


def make_nan_free(
    X: np.ndarray | pd.DataFrame,
    reference: np.ndarray | pd.DataFrame | None = None,
) -> np.ndarray:
    """Return a NaN-free copy of *X* by imputing missing values with column medians.

    Tree-based models handle NaNs natively, but SHAP's TreeExplainer can
    produce NaN SHAP values when the input contains NaNs.  This helper
    replaces every NaN with the **median** of that column (computed from
    *reference* if given, else from *X* itself).

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix that may contain NaN values.
    reference : np.ndarray or pd.DataFrame or None
        Reference dataset for computing column medians (e.g. training
        data).  If ``None``, medians are computed from *X* itself.

    Returns
    -------
    np.ndarray
        Copy of *X* with NaNs replaced by column medians.
    """
    X_arr = np.asarray(X, dtype=float).copy()
    ref = np.asarray(reference, dtype=float) if reference is not None else X_arr
    # Compute column-wise median ignoring NaNs
    medians = np.nanmedian(ref, axis=0)
    # Replace NaNs in each column with the corresponding median
    for col_idx in range(X_arr.shape[1]):
        nan_mask = np.isnan(X_arr[:, col_idx])
        if nan_mask.any():
            X_arr[nan_mask, col_idx] = medians[col_idx]
    return X_arr


def _extract_slearner_base_model(model):
    """Extract the fitted base model from a CausalML S-Learner (internal helper).

    CausalML's BaseSRegressor stores fitted models in ``self.models``
    (dict keyed by treatment group). Used only in the S-learner branch
    of compute_shap_uplift.

    Parameters
    ----------
    model : BaseSRegressor (fitted)
        A fitted CausalML S-Learner.

    Returns
    -------
    object
        The fitted base estimator (e.g. XGBRegressor).
    """
    fitted_models = getattr(model, "models", None)
    if fitted_models is None or len(fitted_models) == 0:
        raise AttributeError(
            "Cannot find fitted internal models on the S-Learner. "
            "Expected attribute 'models' (dict). Has fit() been called?"
        )
    group_key = next(iter(fitted_models))
    return fitted_models[group_key]


def _fix_xgboost_base_score_for_shap(base_model) -> None:
    """Fix XGBoost 3.1+ base_score format so SHAP TreeExplainer can parse the model.

    XGBoost 3.1+ stores base_score as a string like '[5E-1]' in the saved JSON.
    SHAP's TreeExplainer does float(learner_model_param[\"base_score\"]) and fails.
    This helper saves the booster to JSON, converts base_score to a float, reloads,
    so that the in-memory model is unchanged for prediction but SHAP can read it.

    Modifies the booster in place. No-op if the model is not XGBoost or has no
    get_booster(), or if saving/loading is not supported.

    Parameters
    ----------
    base_model : object
        Fitted model (e.g. XGBRegressor) that may have a get_booster().

    Returns
    -------
    None
    """
    import json
    import tempfile

    try:
        booster = base_model.get_booster()
    except Exception:
        return

    def fix_base_score_in_dict(obj: dict) -> bool:
        """Recursively fix 'base_score' values so float(...) works (e.g. '[5E-1]' -> '5E-1')."""
        changed = False
        for key, val in list(obj.items()):
            if key == "base_score":
                if isinstance(val, str) and ("[" in val or "]" in val):
                    s = val.strip("[]")
                    try:
                        float(s)  # ensure it parses
                        obj[key] = s
                        changed = True
                    except ValueError:
                        pass
                elif isinstance(val, list) and len(val) > 0:
                    v0 = val[0]
                    if isinstance(v0, (int, float)):
                        obj[key] = str(v0)
                        changed = True
                    elif isinstance(v0, str):
                        s = v0.strip("[]")
                        try:
                            float(s)
                            obj[key] = s
                            changed = True
                        except ValueError:
                            pass
            elif isinstance(val, dict):
                changed = fix_base_score_in_dict(val) or changed
            elif isinstance(val, list) and val and isinstance(val[0], dict):
                for item in val:
                    changed = fix_base_score_in_dict(item) or changed
        return changed

    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        booster.save_model(tmp_path)
        try:
            with open(tmp_path, encoding="utf-8") as f:
                config = json.load(f)
            if fix_base_score_in_dict(config):
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2)
                booster.load_model(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        pass


def _is_pipeline_linear(base_model) -> bool:
    """Check if a base model is a sklearn Pipeline ending with a linear model."""
    from sklearn.pipeline import Pipeline
    if isinstance(base_model, Pipeline):
        final_step = base_model.steps[-1][1]
        from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
        return isinstance(final_step, (Ridge, Lasso, ElasticNet, LinearRegression))
    return False


def _make_shap_explainer_for_base(base_model, X_background: np.ndarray):
    """Create the appropriate SHAP explainer for the given base model.

    For tree models (LGBM, XGBoost): uses TreeExplainer.
    For linear Pipelines (Ridge/Lasso/ElasticNet): transforms background
    data through the Pipeline's preprocessing steps, then uses
    LinearExplainer on the final linear model.

    Parameters
    ----------
    base_model : fitted sklearn estimator or Pipeline
        The internal base model from CausalML S-learner.
    X_background : np.ndarray
        Background data for the explainer (augmented with treatment column).

    Returns
    -------
    explainer : shap.Explainer
        A SHAP explainer appropriate for the model type.
    """
    if _is_pipeline_linear(base_model):
        # Transform background through imputer + scaler, then explain the linear model
        from sklearn.pipeline import Pipeline
        preprocessing = Pipeline(base_model.steps[:-1])
        X_bg_transformed = preprocessing.transform(X_background)
        linear_model = base_model.steps[-1][1]
        return shap.LinearExplainer(linear_model, X_bg_transformed)
    else:
        # Tree model: use TreeExplainer with XGBoost base_score fix
        _fix_xgboost_base_score_for_shap(base_model)
        _real_float = float

        def _float_for_shap(x):
            """Allow float() to accept XGBoost 3.1+ base_score strings like '[5E-1]'."""
            if isinstance(x, str):
                x = x.strip().strip("[]")
            if isinstance(x, list) and len(x) > 0:
                x = x[0]
            return _real_float(x)

        import builtins
        try:
            builtins.float = _float_for_shap
            try:
                explainer = shap.TreeExplainer(base_model)
            finally:
                builtins.float = _real_float
        except ValueError as e:
            builtins.float = _real_float
            if "could not convert string to float" in str(e) and "base_score" in str(e).lower():
                _fix_xgboost_base_score_for_shap(base_model)
                builtins.float = _float_for_shap
                try:
                    explainer = shap.TreeExplainer(base_model)
                finally:
                    builtins.float = _real_float
            else:
                raise
        return explainer


def _shap_explain(explainer, base_model, X_data: np.ndarray) -> object:
    """Run SHAP explanation, handling Pipeline preprocessing if needed.

    For linear Pipelines: transforms X through preprocessing before explaining.
    For tree models: passes X directly.

    Parameters
    ----------
    explainer : shap.Explainer
    base_model : fitted base model (Pipeline or tree)
    X_data : np.ndarray
        Augmented data [T | X].

    Returns
    -------
    SHAP explanation object or array.
    """
    if _is_pipeline_linear(base_model):
        from sklearn.pipeline import Pipeline
        preprocessing = Pipeline(base_model.steps[:-1])
        X_transformed = preprocessing.transform(X_data)
        return explainer(X_transformed)
    else:
        return explainer(X_data)


def _is_tlearner(model) -> bool:
    """Return True if the model is a CausalML T-learner (has models_c and models_t)."""
    return (
        getattr(model, "models_c", None) is not None
        and getattr(model, "models_t", None) is not None
    )


def compute_shap_uplift(
    model,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
    reference_X: np.ndarray | pd.DataFrame | None = None,
    impute_nan: bool = False,
) -> tuple:
    """Compute per-feature SHAP values for the **uplift** (−CATE) of any supported meta-learner.

    Supports **S-Learner** and **T-Learner** (and other CausalML meta-learners that expose
    models_c/models_t or models). Tree-based (LGBM, XGB) and linear (Ridge/Lasso/ElasticNet) bases.

    A positive SHAP value = that feature pushes predicted uplift higher (more benefit from treatment).

    **impute_nan:** If False (default), *X* is used as-is. If True, *X* is passed through
    ``make_nan_free`` using *reference_X* before SHAP (use when inputs contain NaNs).

    Parameters
    ----------
    model : fitted CausalML meta-learner
        e.g. BaseSRegressor, BaseTRegressor.
    X : np.ndarray or pd.DataFrame
        Feature matrix (same columns as in fit, **without** treatment).
    feature_names : list[str] or None
        Names for the features. If ``None``, uses ``["f0", "f1", ...]``.
    reference_X : np.ndarray or pd.DataFrame or None
        Used only when impute_nan=True for median imputation reference.
    impute_nan : bool, default False
        If True, impute NaNs in X with column medians before SHAP. If False, use X as-is.

    Returns
    -------
    shap_values_matrix : np.ndarray, shape (n_samples, n_features)
        Per-feature SHAP values for retention uplift (−CATE).
    feature_names : list[str]
        Feature names aligned with columns of *shap_values_matrix*.
    expected_uplift : float
        Baseline expected uplift (difference of base values).
    """
    if shap is None:
        raise ImportError("shap package is required. Install with: pip install shap")

    if impute_nan and reference_X is not None:
        X_clean = make_nan_free(X, reference=reference_X)
    elif impute_nan:
        X_clean = make_nan_free(X, reference=X)
    else:
        X_clean = np.asarray(X, dtype=np.float64) if not isinstance(X, np.ndarray) else X
    n = X_clean.shape[0]

    if _is_tlearner(model):
        # T-Learner: two models (control and treatment), each takes X only. CATE = pred_t - pred_c.
        # Uplift = −CATE = pred_c − pred_t, so uplift_shap = SHAP(model_c) − SHAP(model_t).
        t_groups = getattr(model, "t_groups", None)
        if t_groups is None or len(t_groups) == 0:
            raise AttributeError("T-learner has no t_groups.")
        group = t_groups[0]
        model_c = model.models_c[group]
        model_t = model.models_t[group]
        # Background for explainers (X only)
        explainer_c = _make_shap_explainer_for_base(model_c, X_clean)
        explainer_t = _make_shap_explainer_for_base(model_t, X_clean)
        shap_c = _shap_explain(explainer_c, model_c, X_clean)
        shap_t = _shap_explain(explainer_t, model_t, X_clean)
        vals_c = getattr(shap_c, "values", shap_c)
        vals_t = getattr(shap_t, "values", shap_t)
        vals_c = np.asarray(vals_c).squeeze()
        vals_t = np.asarray(vals_t).squeeze()
        if vals_c.ndim > 2:
            vals_c = vals_c.reshape(vals_c.shape[0], -1)
        if vals_t.ndim > 2:
            vals_t = vals_t.reshape(vals_t.shape[0], -1)
        # Uplift = −CATE = pred_c − pred_t → SHAP(uplift) = SHAP_c − SHAP_t
        uplift_shap = vals_c - vals_t
        base_c = getattr(shap_c, "base_values", np.array(0.0))
        base_t = getattr(shap_t, "base_values", np.array(0.0))
        base_c = np.asarray(base_c).ravel()
        base_t = np.asarray(base_t).ravel()
        expected_uplift = float(np.mean(base_c) - np.mean(base_t))
    else:
        # S-Learner: single model with [T | X]
        base_model = _extract_slearner_base_model(model)
        X_t1 = np.hstack([np.ones((n, 1)), X_clean])
        X_t0 = np.hstack([np.zeros((n, 1)), X_clean])

        explainer = _make_shap_explainer_for_base(base_model, X_t0)
        shap_t1 = _shap_explain(explainer, base_model, X_t1)
        shap_t0 = _shap_explain(explainer, base_model, X_t0)

        vals_t1 = getattr(shap_t1, "values", shap_t1)
        vals_t0 = getattr(shap_t0, "values", shap_t0)
        vals_t1 = np.asarray(vals_t1).squeeze()
        vals_t0 = np.asarray(vals_t0).squeeze()
        if vals_t1.ndim > 2:
            vals_t1 = vals_t1.reshape(vals_t1.shape[0], -1)
        if vals_t0.ndim > 2:
            vals_t0 = vals_t0.reshape(vals_t0.shape[0], -1)

        uplift_shap = vals_t0[:, 1:] - vals_t1[:, 1:]

        base_t1 = getattr(shap_t1, "base_values", np.array(0.0))
        base_t0 = getattr(shap_t0, "base_values", np.array(0.0))
        base_t1 = np.asarray(base_t1).ravel()
        base_t0 = np.asarray(base_t0).ravel()
        expected_uplift = float(np.mean(base_t0) - np.mean(base_t1))

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X_clean.shape[1])]

    return uplift_shap, list(feature_names), expected_uplift


# Backward-compatibility alias (name referred to S-learner; function supports all meta-learners).
compute_shap_uplift_slearner = compute_shap_uplift


def plot_shap_importance_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    title: str = "Feature importance for predicted uplift (mean |SHAP|)",
    top_n: int | None = None,
    figsize: tuple = (8, 5),
    color: str = "#1f77b4",
) -> None:
    """Horizontal bar chart of mean absolute SHAP values (global feature importance).

    Parameters
    ----------
    shap_values : np.ndarray, shape (n_samples, n_features)
        SHAP values matrix (e.g. from ``compute_shap_uplift``).
    feature_names : list[str]
        Feature names aligned with columns of *shap_values*.
    title : str
        Plot title.
    top_n : int or None
        Show only the top *n* features.  ``None`` shows all.
    figsize : tuple
        Figure size.
    color : str
        Bar color.

    Returns
    -------
    None
        Displays the plot.
    """
    # Ensure 2D array (squeeze in case of (n, f, 1) from some SHAP versions)
    shap_values = np.asarray(shap_values).squeeze()
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    if shap_values.ndim > 2:
        shap_values = shap_values.reshape(shap_values.shape[0], -1)

    n_features = shap_values.shape[1]
    # Align feature names: use only as many as we have columns
    names = list(feature_names)[:n_features]
    if len(names) < n_features:
        names = names + [f"f{i}" for i in range(len(names), n_features)]

    # Mean absolute SHAP value per feature
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)  # ascending for horizontal bar
    if top_n is not None:
        order = order[-top_n:]

    sorted_names = [names[i] for i in order]
    sorted_vals = np.asarray(mean_abs[order]).ravel()

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(sorted_names, sorted_vals, color=color)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_display: np.ndarray | pd.DataFrame,
    feature_names: list[str],
    title: str = "SHAP beeswarm — feature effects on predicted uplift",
    figsize: tuple = (9, 6),
) -> None:
    """Beeswarm plot showing per-observation SHAP values coloured by feature value.

    Each dot is one observation; horizontal position is the SHAP value
    (positive → pushes uplift higher); colour represents the feature value
    (red = high, blue = low).

    Parameters
    ----------
    shap_values : np.ndarray, shape (n_samples, n_features)
        SHAP values matrix.
    X_display : np.ndarray or pd.DataFrame
        The original (NaN-free) feature values used for colouring the dots.
    feature_names : list[str]
        Feature names aligned with columns of *shap_values*.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    None
        Displays the plot.
    """
    if shap is None:
        raise ImportError("shap package is required.")

    # Build an Explanation object so shap.plots.beeswarm can render it
    explanation = shap.Explanation(
        values=shap_values,
        data=np.asarray(X_display),
        feature_names=feature_names,
    )
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=13, fontweight="bold")
    shap.plots.beeswarm(explanation, show=False, max_display=len(feature_names))
    plt.tight_layout()


def _partial_corr_one(
    x_j: np.ndarray,
    y: np.ndarray,
    X_other: np.ndarray,
) -> float:
    """Partial correlation of x_j with y controlling for X_other (residual approach).

    Returns 0.0 when x_j or residuals have zero variance (correlation undefined).
    """
    # If x_j has zero variance (e.g. constant SHAP), correlation is undefined → 0.0
    if np.nanstd(x_j) < 1e-12:
        return 0.0
    if X_other.size == 0 or X_other.shape[1] == 0:
        r, _ = pearsonr(x_j, y)
        return float(r) if (r is not None and np.isfinite(r)) else 0.0
    reg = LinearRegression().fit(X_other, x_j)
    r_j = x_j - reg.predict(X_other)
    reg_y = LinearRegression().fit(X_other, y)
    r_y = y - reg_y.predict(X_other)
    # If residuals have zero variance after partialing out, correlation is undefined → 0.0
    if np.nanstd(r_j) < 1e-12 or np.nanstd(r_y) < 1e-12:
        return 0.0
    r, _ = pearsonr(r_j, r_y)
    return float(r) if (r is not None and np.isfinite(r)) else 0.0


def compute_parshap_overfitting(
    model,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    treatment: np.ndarray,
    feature_names: list[str],
    train_frac: float = 0.8,
    random_state: int = RANDOM_STATE,
    reference_X: np.ndarray | pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute partial correlation of SHAP with real outcome on train vs holdout.

    ParSHAP requires **real labels** on both splits to detect overfitting.
    This function splits the labelled data (X, y, treatment) into train
    (train_frac) and holdout (1 - train_frac), fits a **clone** of the
    model on the train portion, computes SHAP on both splits using that
    clone, then measures partial correlation of each feature's SHAP values
    with the real outcome y (controlling for other SHAP columns).

    Features whose partial correlation is much higher on train than on
    holdout are candidates for overfitting (they help explain the target
    in-sample but not out-of-sample).  The scatter plot (see
    ``plot_parshap_overfitting``) puts train on x and holdout on y;
    points below the diagonal indicate potential overfitting.

    Parameters
    ----------
    model : BaseSRegressor (fitted or template)
        S-learner.  A deep copy is fit on the train fraction.
    X : np.ndarray or pd.DataFrame
        Full labelled feature matrix.
    y : np.ndarray
        Binary outcome (churn 0/1).
    treatment : np.ndarray
        Treatment indicator (0/1).
    feature_names : list[str]
        Names for the feature columns.
    train_frac : float
        Fraction of labelled data used for training the clone
        (rest = holdout for comparison).
    random_state : int
        Seed for train/holdout split (stratified by 2*treatment + y).
    reference_X : np.ndarray or pd.DataFrame or None
        Reference for NaN imputation when computing SHAP.  If None,
        uses the full X.

    Returns
    -------
    par_train : np.ndarray, shape (n_features,)
        Partial correlation of each feature's SHAP with real y on train.
    par_holdout : np.ndarray, shape (n_features,)
        Same on holdout (real y, not predicted).
    feature_names : list[str]
        Feature names for plotting.
    """
    from copy import deepcopy
    from sklearn.model_selection import train_test_split

    X_arr = np.asarray(X, dtype=np.float64)
    if reference_X is None:
        reference_X = X_arr
    # Stratified split of labelled data into train / holdout
    stratify = 2 * np.asarray(treatment).ravel() + np.asarray(y).ravel()
    train_idx, hold_idx = train_test_split(
        np.arange(len(y)),
        train_size=train_frac,
        stratify=stratify,
        random_state=random_state,
    )
    X_tr = X_arr[train_idx]
    y_tr = np.asarray(y).ravel()[train_idx]
    t_tr = np.asarray(treatment).ravel()[train_idx]
    X_ho = X_arr[hold_idx]
    y_ho = np.asarray(y).ravel()[hold_idx]
    # Fit a clone on train only (same hyper-params, fresh fit)
    model_clone = deepcopy(model)
    model_clone.fit(X_tr, t_tr, y_tr)
    # Compute SHAP on train and holdout using the clone
    shap_tr, _, _ = compute_shap_uplift(
        model_clone, X_tr, feature_names=feature_names, reference_X=reference_X, impute_nan=True
    )
    shap_ho, _, _ = compute_shap_uplift(
        model_clone, X_ho, feature_names=feature_names, reference_X=reference_X, impute_nan=True
    )
    shap_tr = np.asarray(shap_tr).squeeze()
    shap_ho = np.asarray(shap_ho).squeeze()
    if shap_tr.ndim == 1:
        shap_tr = shap_tr.reshape(-1, 1)
    if shap_ho.ndim == 1:
        shap_ho = shap_ho.reshape(-1, 1)
    n_features = shap_tr.shape[1]
    par_train = np.zeros(n_features)
    par_holdout = np.zeros(n_features)
    for j in range(n_features):
        other_idx = [i for i in range(n_features) if i != j]
        X_other_tr = shap_tr[:, other_idx] if other_idx else np.empty((shap_tr.shape[0], 0))
        X_other_ho = shap_ho[:, other_idx] if other_idx else np.empty((shap_ho.shape[0], 0))
        par_train[j] = _partial_corr_one(shap_tr[:, j], y_tr, X_other_tr)
        par_holdout[j] = _partial_corr_one(shap_ho[:, j], y_ho, X_other_ho)
    par_train = np.asarray(par_train)
    par_holdout = np.asarray(par_holdout)
    names_out = list(feature_names)[:n_features]
    return par_train, par_holdout, names_out


def plot_parshap_overfitting(
    par_train: np.ndarray,
    par_holdout: np.ndarray,
    feature_names: list[str],
    title: str = "ParSHAP: partial corr. of SHAP with target (train vs holdout)",
    figsize: tuple = (8, 8),
    save_path: str | Path | None = None,
) -> None:
    """Scatter plot of partial correlation on train (x) vs holdout (y) per feature.

    Each point is one feature.  The diagonal y=x means perfect consistency
    between in-sample and out-of-sample.  Points **below** the diagonal
    (train > holdout) suggest the feature's SHAP-target link is stronger
    in-sample → potential overfitting.  Points on or above the diagonal
    are well-generalising.  Colour encodes the overfitting gap
    (train − holdout): **red = more overfitting**, **green = consistent**.
    A colorbar legend is displayed.

    Parameters
    ----------
    par_train : np.ndarray, shape (n_features,)
        Partial correlation of each feature's SHAP with real outcome on
        the train split.
    par_holdout : np.ndarray, shape (n_features,)
        Same on the holdout split (real outcome, not predicted).
    feature_names : list[str]
        Feature names aligned with par_train / par_holdout.
    title : str
        Plot title.
    figsize : tuple
        Figure size (square recommended).
    save_path : str or Path or None
        If set, save the figure to this path.
    """
    from matplotlib.colors import Normalize  # local import to avoid top-level dep

    par_train = np.asarray(par_train).ravel()
    par_holdout = np.asarray(par_holdout).ravel()
    n = len(par_train)
    feature_names = list(feature_names)[:n]

    fig, ax = plt.subplots(figsize=figsize)

    # Overfitting gap: positive = train stronger than holdout = potential overfitting
    diffs = par_train - par_holdout
    cmap = plt.cm.RdYlGn_r          # red = high diff (overfitting), green = low
    norm = Normalize(vmin=diffs.min(), vmax=diffs.max())

    scatter = ax.scatter(
        par_train, par_holdout, c=diffs, cmap=cmap, norm=norm,
        s=90, edgecolors="black", linewidths=0.5, zorder=3,
    )

    # Annotate each feature with smart offset to reduce overlap
    for i, name in enumerate(feature_names):
        ax.annotate(
            name, (par_train[i], par_holdout[i]),
            xytext=(6, 6), textcoords="offset points", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, lw=0),
        )

    # Axis range (square, symmetric around the data)
    lim_lo = min(par_train.min(), par_holdout.min()) - 0.03
    lim_hi = max(par_train.max(), par_holdout.max()) + 0.03
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)

    # Reference lines
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="-")
    ax.axvline(0, color="gray", linewidth=0.7, linestyle="-")
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1,
            label="y = x (no overfitting)")

    ax.set_xlabel("Partial corr. (SHAP vs target) — Train split", fontsize=11)
    ax.set_ylabel("Partial corr. (SHAP vs target) — Holdout split", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=10)

    # Colorbar legend explaining the colours
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Overfitting gap (train − holdout)", fontsize=10)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    if save_path is not None:
        plt.close(fig)


# ===========================================================================
# Section 9 — Business metrics helpers (labelled data only)
# ===========================================================================


def compute_incremental_churn_at_k(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    k: float,
) -> dict:
    """Compute incremental churn reduction when targeting the top *k*% by uplift.

    Among the top *k*% of users (ranked by predicted uplift), we compare
    churn rates between treated and control groups.  The difference is the
    **realised incremental churn reduction** attributable to the treatment.

    Parameters
    ----------
    y : np.ndarray
        Binary churn labels (1 = churned).
    treatment : np.ndarray
        Binary treatment indicator (1 = treated / outreached).
    uplift_scores : np.ndarray
        Predicted uplift scores (higher = more benefit from treatment).
    k : float
        Fraction of population to target (e.g. 0.10 for top 10 %).

    Returns
    -------
    dict
        Keys: ``k``, ``n_targeted``, ``n_treated``, ``n_control``,
        ``churn_rate_treated``, ``churn_rate_control``,
        ``incremental_churn_reduction``, ``churns_prevented``.
    """
    y = np.asarray(y)
    treatment = np.asarray(treatment)
    uplift_scores = np.asarray(uplift_scores)
    n = len(y)
    n_target = max(1, int(n * k))

    # Select the top-k% by predicted uplift
    top_idx = np.argsort(-uplift_scores)[:n_target]
    y_top = y[top_idx]
    t_top = treatment[top_idx]

    # Split into treated and control within top-k%
    treated_mask = t_top == 1
    control_mask = t_top == 0

    n_treated = treated_mask.sum()
    n_control = control_mask.sum()

    # Churn rates within each group
    churn_treated = y_top[treated_mask].mean() if n_treated > 0 else np.nan
    churn_control = y_top[control_mask].mean() if n_control > 0 else np.nan

    # Incremental reduction: how much churn drops from treatment
    incremental = churn_control - churn_treated if (n_treated > 0 and n_control > 0) else np.nan

    # Churns prevented = incremental_rate × n_targeted
    churns_prevented = incremental * n_target if not np.isnan(incremental) else np.nan

    return {
        "k": k,
        "n_targeted": n_target,
        "n_treated": n_treated,
        "n_control": n_control,
        "churn_rate_treated": churn_treated,
        "churn_rate_control": churn_control,
        "incremental_churn_reduction": incremental,
        "churns_prevented": churns_prevented,
    }


def compute_lift_over_random(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    k: float,
) -> dict:
    """Compute the model's lift over random targeting at the top *k*%.

    Random targeting yields an expected incremental reduction proportional
    to the **overall** treatment effect.  The lift ratio shows how much
    better the model is than random at identifying treatable users.

    Parameters
    ----------
    y : np.ndarray
        Binary churn labels (1 = churned).
    treatment : np.ndarray
        Binary treatment indicator (1 = treated).
    uplift_scores : np.ndarray
        Predicted uplift scores.
    k : float
        Fraction of population targeted.

    Returns
    -------
    dict
        Keys: ``k``, ``model_reduction``, ``random_reduction``,
        ``lift_ratio``, ``absolute_gain``.
    """
    y = np.asarray(y)
    treatment = np.asarray(treatment)

    # Overall (population-level) incremental churn reduction
    overall_churn_treated = y[treatment == 1].mean()
    overall_churn_control = y[treatment == 0].mean()
    overall_incremental = overall_churn_control - overall_churn_treated

    # Random targeting: expected reduction at k%
    random_reduction = overall_incremental * k

    # Model-based targeting at k%
    model_stats = compute_incremental_churn_at_k(y, treatment, uplift_scores, k)
    model_reduction = model_stats["incremental_churn_reduction"] * k

    # Lift ratio (model / random); guard against division by zero
    lift_ratio = (model_reduction / random_reduction) if random_reduction != 0 else np.nan
    absolute_gain = model_reduction - random_reduction

    return {
        "k": k,
        "model_reduction": model_reduction,
        "random_reduction": random_reduction,
        "lift_ratio": lift_ratio,
        "absolute_gain": absolute_gain,
    }


def build_capacity_curve_data(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    steps: int = 20,
) -> pd.DataFrame:
    """Build a capacity curve: cumulative churns prevented vs. fraction targeted.

    For each step from 0 % to 100 %, compute how many **incremental
    churns** would be prevented if we target the top *k*% by predicted
    uplift.

    Parameters
    ----------
    y : np.ndarray
        Binary churn labels (1 = churned).
    treatment : np.ndarray
        Binary treatment indicator.
    uplift_scores : np.ndarray
        Predicted uplift scores (higher = more benefit).
    steps : int
        Number of evenly spaced capacity levels between 0 and 1.

    Returns
    -------
    pd.DataFrame
        Columns: ``fraction_targeted``, ``incremental_churn_reduction``,
        ``churns_prevented``, ``churns_per_1k_outreaches``.
    """
    fractions = np.linspace(0, 1, steps + 1)[1:]  # skip 0%
    rows = []
    for frac in fractions:
        stats = compute_incremental_churn_at_k(y, treatment, uplift_scores, frac)
        churns_prev = stats["churns_prevented"]
        n_targeted = stats["n_targeted"]
        # Churns prevented per 1,000 outreaches
        per_1k = (churns_prev / n_targeted * 1000) if (n_targeted > 0 and not np.isnan(churns_prev)) else np.nan
        rows.append({
            "fraction_targeted": frac,
            "incremental_churn_reduction": stats["incremental_churn_reduction"],
            "churns_prevented": churns_prev,
            "churns_per_1k_outreaches": per_1k,
        })
    return pd.DataFrame(rows)


def plot_capacity_curve(
    capacity_df: pd.DataFrame,
    title: str = "Capacity curve — incremental churns prevented by contact rate",
    figsize: tuple = (9, 5),
) -> None:
    """Plot cumulative incremental churns prevented vs. fraction of users targeted.

    Shows where marginal benefit flattens: beyond a certain contact rate,
    each additional outreach prevents fewer churns.

    Parameters
    ----------
    capacity_df : pd.DataFrame
        Output of ``build_capacity_curve_data``.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    None
        Displays the plot.
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    fracs = capacity_df["fraction_targeted"] * 100  # percent

    # Primary axis: cumulative churns prevented
    color1 = "#2c7bb6"
    ax1.plot(fracs, capacity_df["churns_prevented"], color=color1,
             marker="o", markersize=4, linewidth=2, label="Churns prevented")
    ax1.set_xlabel("% of users contacted", fontsize=12)
    ax1.set_ylabel("Cumulative churns prevented (count)", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    # Secondary axis: churns per 1k outreaches (efficiency)
    ax2 = ax1.twinx()
    color2 = "#d7191c"
    ax2.plot(fracs, capacity_df["churns_per_1k_outreaches"], color=color2,
             marker="s", markersize=4, linewidth=2, linestyle="--",
             label="Churns prevented per 1k outreaches")
    ax2.set_ylabel("Churns prevented per 1,000 outreaches", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    fig.tight_layout()


def build_segment_quality_table(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    segments: np.ndarray,
) -> pd.DataFrame:
    """Per-segment quality table with realised churn rates and uplift.

    For each segment reports size, average predicted uplift, churn rate
    in treated vs. control, and realised incremental churn reduction.

    Parameters
    ----------
    y : np.ndarray
        Binary churn labels (1 = churned).
    treatment : np.ndarray
        Binary treatment indicator.
    uplift_scores : np.ndarray
        Predicted uplift scores.
    segments : np.ndarray
        Segment labels (e.g. "Persuadables", "Sure Things", etc.).

    Returns
    -------
    pd.DataFrame
        One row per segment with quality metrics.
    """
    y = np.asarray(y)
    treatment = np.asarray(treatment)
    uplift_scores = np.asarray(uplift_scores)
    segments = np.asarray(segments)

    seg_order = ["Persuadables", "Sure Things", "Lost Causes", "Do-Not-Disturb"]
    rows = []
    for seg in seg_order:
        mask = segments == seg
        n = mask.sum()
        if n == 0:
            continue
        y_seg = y[mask]
        t_seg = treatment[mask]
        u_seg = uplift_scores[mask]

        n_treated = (t_seg == 1).sum()
        n_control = (t_seg == 0).sum()
        churn_treated = y_seg[t_seg == 1].mean() if n_treated > 0 else np.nan
        churn_control = y_seg[t_seg == 0].mean() if n_control > 0 else np.nan
        realised_uplift = (churn_control - churn_treated) if (n_treated > 0 and n_control > 0) else np.nan

        rows.append({
            "Segment": seg,
            "N": n,
            "Share (%)": round(n / len(y) * 100, 1),
            "Avg predicted uplift": round(u_seg.mean(), 4),
            "Churn rate (treated)": round(churn_treated, 4) if not np.isnan(churn_treated) else np.nan,
            "Churn rate (control)": round(churn_control, 4) if not np.isnan(churn_control) else np.nan,
            "Realised uplift": round(realised_uplift, 4) if not np.isnan(realised_uplift) else np.nan,
        })
    return pd.DataFrame(rows)


def build_business_metrics_summary(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    ks: list[float] | None = None,
) -> pd.DataFrame:
    """Compact business-metrics summary table at multiple targeting thresholds.

    Each column includes a measurement unit in its name. Churns prevented
    is expressed as a percentage (rate in the targeted slice), not as a count.

    Parameters
    ----------
    y : np.ndarray
        Binary churn labels (1 = churned).
    treatment : np.ndarray
        Binary treatment indicator.
    uplift_scores : np.ndarray
        Predicted uplift scores.
    ks : list[float] or None
        Targeting fractions.  Defaults to ``[0.05, 0.10, 0.20, 0.30, 0.50]``.

    Returns
    -------
    pd.DataFrame
        One row per *k* with columns: Top k% (%), N targeted (n), Churn treated (fraction),
        Churn control (fraction), Churns prevented (%), Lift over random (ratio).
    """
    if ks is None:
        ks = [0.05, 0.10, 0.20, 0.30, 0.50]

    rows = []
    for k in ks:
        churn_stats = compute_incremental_churn_at_k(y, treatment, uplift_scores, k)
        lift_stats = compute_lift_over_random(y, treatment, uplift_scores, k)
        n_target = churn_stats["n_targeted"]
        inc = churn_stats["incremental_churn_reduction"]
        # Churns prevented as % (rate in targeted slice), not count
        churns_prevented_pct = (inc * 100) if not np.isnan(inc) else np.nan

        row = {
            "Top k% (%)": f"{k*100:.0f}%",
            "N targeted (n)": n_target,
            "Churn treated (fraction)": round(churn_stats["churn_rate_treated"], 4),
            "Churn control (fraction)": round(churn_stats["churn_rate_control"], 4),
            "Churns prevented (%)": round(churns_prevented_pct, 2) if not np.isnan(churns_prevented_pct) else np.nan,
            "Lift over random (ratio)": round(lift_stats["lift_ratio"], 2) if not np.isnan(lift_stats["lift_ratio"]) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_business_churns_prevented_bar(
    summary_df: pd.DataFrame,
    title: str = "Churns prevented by targeting top k%",
    figsize: tuple = (8, 4),
) -> None:
    """Bar chart of churns prevented (%) at each top-k% threshold (for stakeholder presentation).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of ``build_business_metrics_summary`` (must have "Top k% (%)" and "Churns prevented (%)").
    title : str
        Plot title.
    figsize : tuple
        Figure size (width, height).

    Returns
    -------
    None
        Displays the plot.
    """
    col_k = "Top k% (%)"
    col_cp = "Churns prevented (%)"
    if col_cp not in summary_df.columns:
        return
    fig, ax = plt.subplots(figsize=figsize)
    x_labels = summary_df[col_k].astype(str)
    vals = summary_df[col_cp].values
    vals_plot = np.where(np.isnan(vals), 0.0, vals)
    x_pos = np.arange(len(x_labels))
    bars = ax.bar(x_pos, vals_plot, color="#2c7bb6", edgecolor="none")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Top k% targeted (%)", fontsize=11)
    ax.set_ylabel("Churns prevented (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (bar, v) in enumerate(zip(bars, vals)):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    plt.show()


def plot_business_incremental_reduction_bar(
    summary_df: pd.DataFrame,
    title: str = "Incremental churn reduction (Uplift@k) by top k%",
    figsize: tuple = (8, 4),
) -> None:
    """Bar chart of incremental churn reduction (Uplift@k) at each top-k% threshold.

    Uses "Churns prevented (%)" from the summary table (same metric as incremental reduction in %).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of ``build_business_metrics_summary`` ("Top k% (%)", "Churns prevented (%)").
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    None
        Displays the plot.
    """
    col_k = "Top k% (%)"
    col_pct = "Churns prevented (%)"
    if col_pct not in summary_df.columns:
        return
    fig, ax = plt.subplots(figsize=figsize)
    x_labels = summary_df[col_k].astype(str)
    vals = summary_df[col_pct].values
    vals_plot = np.where(np.isnan(vals), 0.0, vals)
    x_pos = np.arange(len(x_labels))
    ax.bar(x_pos, vals_plot, color="#2c7bb6", edgecolor="none")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Top k% targeted (%)", fontsize=11)
    ax.set_ylabel("Uplift@k (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    plt.show()


def plot_business_lift_over_random_bar(
    summary_df: pd.DataFrame,
    title: str = "Lift over random targeting by top k%",
    figsize: tuple = (8, 4),
) -> None:
    """Bar chart of lift over random at each top-k% threshold.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of ``build_business_metrics_summary`` ("Top k% (%)", "Lift over random (ratio)").
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    None
        Displays the plot.
    """
    col_k = "Top k% (%)"
    col_lift = "Lift over random (ratio)"
    if col_lift not in summary_df.columns:
        return
    fig, ax = plt.subplots(figsize=figsize)
    x_labels = summary_df[col_k].astype(str)
    vals = summary_df[col_lift].values
    vals_plot = np.where(np.isnan(vals), 0.0, vals)
    x_pos = np.arange(len(x_labels))
    ax.bar(x_pos, vals_plot, color="#2c7bb6", edgecolor="none")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Top k% targeted (%)", fontsize=11)
    ax.set_ylabel("Lift over random (ratio)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    plt.show()


def plot_business_revenue_saved_bar(
    summary_df: pd.DataFrame,
    title: str = "Revenue saved by targeting top k%",
    figsize: tuple = (8, 4),
) -> None:
    """Bar chart of revenue saved at each top-k% (no-op; revenue column removed from summary).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of ``build_business_metrics_summary`` (no revenue column by default).
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    None
        Displays the plot. No-op if "Revenue saved" column is missing.
    """
    if "Revenue saved" not in summary_df.columns:
        return
    fig, ax = plt.subplots(figsize=figsize)
    x_labels = summary_df["Top k%"].astype(str)
    vals = summary_df["Revenue saved"].values
    vals_plot = np.where(np.isnan(vals), 0.0, vals)
    x_pos = np.arange(len(x_labels))
    bars = ax.bar(x_pos, vals_plot, color="#2ca02c", edgecolor="none")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Top k% targeted (%)", fontsize=11)
    ax.set_ylabel("Revenue saved ($)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"${v:,.0f}",
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    plt.show()


def plot_segment_quality_grouped_bar(
    segment_df: pd.DataFrame,
    title: str = "Segment quality: churn rate (treated vs control)",
    figsize: tuple = (8, 4),
) -> None:
    """Grouped bar chart: per-segment churn rate (treated vs control) for stakeholder presentation.

    Parameters
    ----------
    segment_df : pd.DataFrame
        Output of ``build_segment_quality_table`` (columns: Segment, Churn rate (treated), Churn rate (control)).
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    None
        Displays the plot.
    """
    if "Churn rate (treated)" not in segment_df.columns or "Churn rate (control)" not in segment_df.columns:
        return
    fig, ax = plt.subplots(figsize=figsize)
    segments = segment_df["Segment"].astype(str)
    x = np.arange(len(segments))
    w = 0.35
    treated = segment_df["Churn rate (treated)"].fillna(0).values
    control = segment_df["Churn rate (control)"].fillna(0).values
    ax.bar(x - w / 2, treated, w, label="Churn (treated)", color="#2c7bb6")
    ax.bar(x + w / 2, control, w, label="Churn (control)", color="#d7191c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(segments)
    ax.set_xlabel("Segment", fontsize=11)
    ax.set_ylabel("Churn rate (fraction; 0–1)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    plt.show()


def compute_and_print_baseline_benchmark(
    y: np.ndarray,
    treatment: np.ndarray,
    benchmark_financial: tuple[float, float] = (2.0, 4.0),
    benchmark_wellness: tuple[float, float] = (7.0, 10.0),
    print_result: bool = True,
) -> dict:
    """Compare churn with outreach (treated) to industry benchmarks; control reported for context.

    The meaningful comparison for stakeholders is: when we do outreach (treated
    group), where does our churn sit vs. 2–4% (financial) and 7–10% (wellness)?
    Control (no outreach) churn is reported only as context.

    Parameters
    ----------
    y : np.ndarray
        Binary churn labels (1 = churned).
    treatment : np.ndarray
        Binary treatment indicator (0 = control, 1 = treated/outreach).
    benchmark_financial : tuple[float, float]
        (low, high) monthly churn % for financial benchmark. Default (2, 4).
    benchmark_wellness : tuple[float, float]
        (low, high) monthly churn % for wellness benchmark. Default (7, 10).
    print_result : bool
        If True, print control (context), treated churn, and benchmark comparison.

    Returns
    -------
    dict
        Keys: ``control_churn_pct``, ``n_control``, ``treated_churn_pct``,
        ``n_treated``, ``vs_financial``, ``vs_wellness`` (comparison for treated).
    """
    y_arr = np.asarray(y)
    t_arr = np.asarray(treatment)
    control_mask = t_arr == 0
    treated_mask = t_arr == 1
    n_control = int(control_mask.sum())
    n_treated = int(treated_mask.sum())

    control_churn_rate = float(y_arr[control_mask].mean()) if n_control > 0 else np.nan
    treated_churn_rate = float(y_arr[treated_mask].mean()) if n_treated > 0 else np.nan
    control_churn_pct = control_churn_rate * 100 if not np.isnan(control_churn_rate) else np.nan
    treated_churn_pct = treated_churn_rate * 100 if not np.isnan(treated_churn_rate) else np.nan

    # Compare *treated* (with outreach) churn to benchmarks
    pct = treated_churn_pct if not np.isnan(treated_churn_pct) else 0.0
    fin_lo, fin_hi = benchmark_financial
    if pct < fin_lo:
        vs_fin = "below the 2–4% financial/subscription benchmark"
    elif pct <= fin_hi:
        vs_fin = "within the 2–4% financial/subscription benchmark"
    else:
        vs_fin = "above the 2–4% financial/subscription benchmark"

    well_lo, well_hi = benchmark_wellness
    if pct < well_lo:
        vs_well = "below the 7–10% wellness/engagement benchmark"
    elif pct <= well_hi:
        vs_well = "within the 7–10% wellness/engagement benchmark"
    else:
        vs_well = "above the 7–10% wellness/engagement benchmark"

    out = {
        "control_churn_rate": control_churn_rate,
        "control_churn_pct": control_churn_pct,
        "n_control": n_control,
        "treated_churn_rate": treated_churn_rate,
        "treated_churn_pct": treated_churn_pct,
        "n_treated": n_treated,
        "vs_financial": vs_fin,
        "vs_wellness": vs_well,
    }
    if print_result:
        print(f"Control (no outreach) monthly churn: {control_churn_pct:.2f}% (n={n_control:,}).")
        print(f"Treated (with outreach) monthly churn: {treated_churn_pct:.2f}% (n={n_treated:,}).")
        print(f"When we do outreach, churn is {vs_fin} and {vs_well}.")
    return out
