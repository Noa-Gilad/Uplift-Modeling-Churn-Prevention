"""
Utility functions for uplift churn modeling: EDA, feature engineering, and model helpers.

This module is intended to be imported by the uplift_churn_modeling notebook.
All logic is documented in docstrings; see the notebook for end-to-end workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore[misc, assignment]
try:
    from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor
    from causalml.metrics import qini_auc_score
except Exception:
    BaseSRegressor = BaseTRegressor = BaseXRegressor = None  # type: ignore[misc, assignment]
    qini_auc_score = None
try:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
except Exception:
    LGBMRegressor = XGBRegressor = None  # type: ignore[misc, assignment]

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
                ax.set_ylabel("% of members with zero activity")
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
                plt.tight_layout()
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
    plt.tight_layout()
    plt.show()


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
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tick_params(labelsize=11)
    plt.tight_layout()
    plt.show()


def engagement_feat_summary(
    eng: pd.DataFrame,
    feat: str,
    bins: int = 50,
) -> None:
    """Print quantile summary and show a log-scaled histogram for one engagement feature.

    Used for distribution sanity checks: call once per feature (e.g. web_visits_count,
    app_sessions_count, url_nunique). Plots log(1 + values) to handle skew.

    Parameters
    ----------
    eng : pandas.DataFrame
        Engagement dataframe with at least the column named by feat (zero-filled).
    feat : str
        Column name to summarize and plot (e.g. 'web_visits_count', 'app_sessions_count', 'url_nunique').
    bins : int, optional
        Number of histogram bins (default 50).

    Returns
    -------
    None
    """
    print(f"\n{feat}:")
    print(eng[feat].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]).to_string())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(np.log1p(eng[feat]), bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel(f"log(1 + {feat})", fontsize=11)
    ax.set_ylabel("Number of members", fontsize=11)
    ax.set_title(feat, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


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
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(bin_names)))
    ax.set_xticklabels(bin_names, rotation=20, ha="right")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Uplift (churn-rate difference)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
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


# ---------------------------------------------------------------------------
# Feature engineering pipeline
# ---------------------------------------------------------------------------


def load_wellco_brief(path: Path | str) -> str:
    """Read the WellCo client brief from disk and return it as a single string.

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
    """Count app sessions per member up to the reference date.

    Returns
    -------
    pd.DataFrame
        Columns: member_id, app_sessions_count.
    """
    adf = app_df[app_df["timestamp"] <= ref_date].copy()
    counts = adf.groupby("member_id").size().rename("app_sessions_count").reset_index()
    out = members_df[["member_id"]].merge(counts, on="member_id", how="left")
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
        Columns: member_id, icd_distinct_count, has_focus_icd, days_since_last_claim.
    """
    if focus_icd_codes is None:
        focus_icd_codes = FOCUS_ICD_CODES
    cdf = claims_df[claims_df["diagnosis_date"] <= ref_date].copy()
    if cdf.empty:
        out = members_df[["member_id"]].copy()
        out["icd_distinct_count"] = 0
        out["has_focus_icd"] = 0
        out["days_since_last_claim"] = np.nan
        return out
    cdf["_is_focus"] = cdf["icd_code"].isin(focus_icd_codes)
    agg = (
        cdf.groupby("member_id")
        .agg(
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


def make_lgbm():
    """Create a LightGBM regressor configured for shallow, regularised trees."""
    if LGBMRegressor is None:
        raise ImportError("lightgbm is not installed")
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def make_xgb(scale_pos_weight: float = 1.0):
    """Create an XGBoost regressor configured for shallow, regularised trees."""
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed")
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
    )


def uplift_at_k(
    y_true: np.ndarray,
    t_true: np.ndarray,
    uplift_scores: np.ndarray,
    k: float,
) -> float:
    """Realised uplift in the top-k fraction of the population ranked by predicted uplift.

    Parameters
    ----------
    y_true : array of int
        Observed churn labels (0/1).
    t_true : array of int
        Treatment indicator (1 = outreach, 0 = control).
    uplift_scores : array of float
        Predicted uplift (higher = more benefit from treatment).
    k : float in (0, 1]
        Fraction of the population to consider (e.g. 0.10 for top 10%).

    Returns
    -------
    float
        churn_rate_control - churn_rate_treated in the top-k segment.
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

    Returns
    -------
    ks : np.ndarray
        Fraction values (0.01 ... 1.0).
    uplift_vals : np.ndarray
        Realised uplift at each fraction.
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


def assign_segments(uplift_scores: np.ndarray) -> np.ndarray:
    """Assign four uplift segments based on predicted-uplift quartiles.

    Returns
    -------
    np.ndarray of str
        Segment labels: Persuadables, Sure Things, Lost Causes, Do-Not-Disturb.
    """
    q25, q50, q75 = np.nanquantile(uplift_scores, [0.25, 0.50, 0.75])
    seg = np.empty(len(uplift_scores), dtype=object)
    seg[uplift_scores >= q75] = "Persuadables"
    seg[(uplift_scores >= q50) & (uplift_scores < q75)] = "Sure Things"
    seg[(uplift_scores >= q25) & (uplift_scores < q50)] = "Lost Causes"
    seg[uplift_scores < q25] = "Do-Not-Disturb"
    return seg


def _build_model(meta_key: str, base_key: str, spw: float):
    """Instantiate a CausalML meta-learner with the requested base learner.

    Parameters
    ----------
    meta_key : str
        One of 'S', 'T', 'X'.
    base_key : str
        One of 'LGBM', 'XGB'.
    spw : float
        scale_pos_weight for XGBoost (ignored for LGBM).

    Returns
    -------
    CausalML meta-learner instance.
    """
    if BaseSRegressor is None or BaseTRegressor is None or BaseXRegressor is None:
        raise ImportError("causalml is not installed")
    learner = make_lgbm() if base_key == "LGBM" else make_xgb(spw)
    meta_map = {"S": BaseSRegressor, "T": BaseTRegressor, "X": BaseXRegressor}
    return meta_map[meta_key](learner=learner)
