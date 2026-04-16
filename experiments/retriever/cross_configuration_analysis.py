"""
Phase 2: Cross-Configuration Failure Analysis
==============================================

Builds a unified per-(question, ground_truth_chunk, config) dataset from
the pre-computed retrieval results, then runs four analyses:

  2a. Coverage matrix: universally hit / universally missed / conditional
  2b. Negative interactions: chunks lost when adding a retriever
  2c. Retriever uniqueness: chunks only one config finds
  2d. Per-question vulnerability: which questions are hardest

Usage (from repo root):
    python experiments/retriever/phase2_analysis.py

Output:
    experiments/retriever/phase2_output/
        phase2_dataset.csv            — the unified long-format table
        coverage_summary.csv          — 2a results
        negative_interactions.csv     — 2b results
        retriever_uniqueness.csv      — 2c results
        question_vulnerability.csv    — 2d results
        coverage_heatmap.html         — visual: per-question x config coverage
"""

import json
import os
import hashlib
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration — adjust these paths if your layout differs
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # experiments/retriever -> repo root

GROUND_TRUTH_FILE = SCRIPT_DIR / "answer" / "75_testingset_75updated.json"
RESULT_DIRS = [
    SCRIPT_DIR / "result_same_num_chunks",
    SCRIPT_DIR / "result_hyde",
]
OUTPUT_DIR = SCRIPT_DIR / "phase2_output"

# Config hierarchy: maps child -> parent, meaning child = parent + something.
# Used in negative interaction analysis (2b) to detect chunks lost by adding.
CONFIG_HIERARCHY = {
    "faiss_expand":         "faiss",
    "faiss_bm25":           "faiss",
    "faiss_ts":             "faiss",
    "faiss_hyde(qwen7b)":   "faiss",
    "faiss_hyde(qwen72b)":  "faiss",
    "faiss_hyde(qwen7b-sft)": "faiss",
    "faiss_bm25_ts":        "faiss_bm25",
    "faiss_bm25_ts_hyde":   "faiss_bm25_ts",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chunk_hash(text: str) -> str:
    """Short hash for a chunk, used as a stable ID."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:12]


def load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 1: Build the unified dataset
# ---------------------------------------------------------------------------

def build_unified_dataset() -> pd.DataFrame:
    """
    Create a long-format DataFrame with one row per
    (question_id, gt_chunk_hash, config) triple.

    Columns:
        question_id       int (0-74)
        question_text     str (original question, may be Chinese)
        gt_chunk_hash     str (12-char hex)
        gt_chunk_preview  str (first 100 chars for readability)
        config            str (folder name like 'faiss_bm25')
        was_retrieved     bool
        recall            float (per-question recall for this config)
        num_gt_chunks     int (how many GT chunks this question has)
    """

    # --- Load ground truth ---
    gt_data = load_json(GROUND_TRUTH_FILE)
    print(f"Loaded ground truth: {len(gt_data)} questions")

    # Build question -> GT chunks mapping
    gt_by_question = {}
    for idx, entry in enumerate(gt_data):
        q = entry["question"]
        chunks = [c.strip() for c in entry.get("content_list", [])]
        gt_by_question[q] = {
            "question_id": idx,
            "chunks": chunks,
            "chunk_hashes": [chunk_hash(c) for c in chunks],
        }

    # --- Scan all result directories ---
    rows = []
    configs_found = []

    for result_root in RESULT_DIRS:
        if not result_root.is_dir():
            print(f"Warning: result directory not found: {result_root}")
            continue

        for config_dir in sorted(result_root.iterdir()):
            result_file = config_dir / "result_2.json"
            if not result_file.is_file():
                continue

            config_name = config_dir.name
            configs_found.append(config_name)
            result_data = load_json(result_file)

            # First element is aggregate metrics — skip it
            question_results = result_data[1:]

            if len(question_results) != len(gt_data):
                print(
                    f"Warning: {config_name} has {len(question_results)} questions, "
                    f"expected {len(gt_data)}. Matching by question text."
                )

            # Build a lookup: question_text -> pos_recalls set
            retrieved_by_question = {}
            recall_by_question = {}
            for entry in question_results:
                q = entry["question"]
                pos = set(c.strip() for c in entry.get("pos_recalls", []))
                retrieved_by_question[q] = pos
                recall_by_question[q] = entry.get("recall", 0.0)

            # Join with ground truth
            for q, gt_info in gt_by_question.items():
                pos_recalls = retrieved_by_question.get(q, set())
                q_recall = recall_by_question.get(q, 0.0)

                for chunk_text, c_hash in zip(gt_info["chunks"], gt_info["chunk_hashes"]):
                    rows.append({
                        "question_id": gt_info["question_id"],
                        "question_text": q,
                        "gt_chunk_hash": c_hash,
                        "gt_chunk_preview": chunk_text[:100],
                        "config": config_name,
                        "was_retrieved": chunk_text in pos_recalls,
                        "recall": q_recall,
                        "num_gt_chunks": len(gt_info["chunks"]),
                    })

    df = pd.DataFrame(rows)
    print(f"Built unified dataset: {len(df)} rows, "
          f"{df['config'].nunique()} configs, "
          f"{df['question_id'].nunique()} questions, "
          f"{df['gt_chunk_hash'].nunique()} unique GT chunks")
    print(f"Configs found: {sorted(configs_found)}")

    return df


# ---------------------------------------------------------------------------
# Analysis 2a: Coverage matrix
# ---------------------------------------------------------------------------

def analysis_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (question, gt_chunk), how many configs retrieved it?

    Output columns: question_id, gt_chunk_hash, gt_chunk_preview,
                    configs_hit, configs_total, category
    """
    grouped = (
        df.groupby(["question_id", "gt_chunk_hash", "gt_chunk_preview"])
        .agg(
            configs_hit=("was_retrieved", "sum"),
            configs_total=("was_retrieved", "count"),
        )
        .reset_index()
    )

    def categorize(row):
        if row["configs_hit"] == 0:
            return "universally_missed"
        elif row["configs_hit"] == row["configs_total"]:
            return "universally_retrieved"
        else:
            return "conditional"

    grouped["category"] = grouped.apply(categorize, axis=1)

    # Print summary
    summary = grouped["category"].value_counts()
    total = len(grouped)
    print("\n=== 2a: Coverage Summary ===")
    for cat, count in summary.items():
        print(f"  {cat}: {count} chunks ({count/total*100:.1f}%)")

    return grouped


# ---------------------------------------------------------------------------
# Analysis 2b: Negative interactions
# ---------------------------------------------------------------------------

def analysis_negative_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    When going from config_parent -> config_child (adding a retriever),
    which GT chunks were LOST (retrieved by parent, missed by child)?

    This detects cases where combining retrievers hurts under a fixed
    chunk budget.
    """
    results = []

    # Pivot: rows = (question_id, gt_chunk_hash), columns = config, values = was_retrieved
    pivot = df.pivot_table(
        index=["question_id", "gt_chunk_hash"],
        columns="config",
        values="was_retrieved",
        aggfunc="first",
    )

    for child, parent in CONFIG_HIERARCHY.items():
        if parent not in pivot.columns or child not in pivot.columns:
            continue

        parent_hit = pivot[parent] == True
        child_miss = pivot[child] == False

        lost = pivot[parent_hit & child_miss].index.tolist()
        gained_mask = (pivot[parent] == False) & (pivot[child] == True)
        gained = pivot[gained_mask].index.tolist()

        results.append({
            "parent_config": parent,
            "child_config": child,
            "added_component": child.replace(parent + "_", "").replace(parent, "expanded"),
            "chunks_lost": len(lost),
            "chunks_gained": len(gained),
            "net_change": len(gained) - len(lost),
            "lost_details": lost[:10],  # first 10 for inspection
        })

    results_df = pd.DataFrame(results)

    print("\n=== 2b: Negative Interactions ===")
    for _, row in results_df.iterrows():
        sign = "+" if row["net_change"] >= 0 else ""
        print(
            f"  {row['parent_config']} -> {row['child_config']}: "
            f"lost={row['chunks_lost']}, gained={row['chunks_gained']}, "
            f"net={sign}{row['net_change']}"
        )

    return results_df


# ---------------------------------------------------------------------------
# Analysis 2c: Retriever uniqueness
# ---------------------------------------------------------------------------

def analysis_uniqueness(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each config, how many GT chunks does ONLY that config retrieve?
    These are the chunks that would be lost if you removed that retriever.
    """
    pivot = df.pivot_table(
        index=["question_id", "gt_chunk_hash"],
        columns="config",
        values="was_retrieved",
        aggfunc="first",
    )

    results = []
    for config in pivot.columns:
        # This config retrieved it
        this_hit = pivot[config] == True
        # No other config retrieved it
        others = [c for c in pivot.columns if c != config]
        others_all_miss = (pivot[others] == False).all(axis=1)

        unique_chunks = pivot[this_hit & others_all_miss]
        total_hit = this_hit.sum()

        results.append({
            "config": config,
            "unique_chunks": len(unique_chunks),
            "total_chunks_retrieved": int(total_hit),
            "uniqueness_pct": len(unique_chunks) / total_hit * 100 if total_hit > 0 else 0,
        })

    results_df = pd.DataFrame(results).sort_values("unique_chunks", ascending=False)

    print("\n=== 2c: Retriever Uniqueness ===")
    for _, row in results_df.iterrows():
        print(
            f"  {row['config']}: "
            f"{row['unique_chunks']} unique / {row['total_chunks_retrieved']} total "
            f"({row['uniqueness_pct']:.1f}%)"
        )

    return results_df


# ---------------------------------------------------------------------------
# Analysis 2d: Per-question vulnerability
# ---------------------------------------------------------------------------

def analysis_vulnerability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank questions by difficulty: how many configs achieve perfect recall
    on each question? Questions where NO config gets all GT chunks are
    'universally hard'.
    """
    # For each (question, config), did we get ALL gt chunks?
    q_config = (
        df.groupby(["question_id", "question_text", "config"])
        .agg(
            gt_total=("was_retrieved", "count"),
            gt_found=("was_retrieved", "sum"),
        )
        .reset_index()
    )
    q_config["perfect_recall"] = q_config["gt_total"] == q_config["gt_found"]

    # Per question: how many configs got perfect recall?
    q_summary = (
        q_config.groupby(["question_id", "question_text"])
        .agg(
            configs_with_perfect_recall=("perfect_recall", "sum"),
            configs_total=("perfect_recall", "count"),
            avg_recall=("gt_found", lambda x: x.sum() / q_config.loc[x.index, "gt_total"].sum()),
            num_gt_chunks=("gt_total", "first"),
        )
        .reset_index()
        .sort_values("configs_with_perfect_recall")
    )

    # Categorize
    def difficulty(row):
        ratio = row["configs_with_perfect_recall"] / row["configs_total"]
        if ratio == 0:
            return "universally_hard"
        elif ratio < 0.5:
            return "hard"
        elif ratio < 1.0:
            return "medium"
        else:
            return "easy"

    q_summary["difficulty"] = q_summary.apply(difficulty, axis=1)

    print("\n=== 2d: Question Vulnerability ===")
    diff_counts = q_summary["difficulty"].value_counts()
    for cat in ["universally_hard", "hard", "medium", "easy"]:
        count = diff_counts.get(cat, 0)
        print(f"  {cat}: {count} questions")

    print(f"\n  Top 5 hardest questions:")
    for _, row in q_summary.head(5).iterrows():
        preview = row["question_text"][:60]
        print(
            f"    Q{row['question_id']}: {row['configs_with_perfect_recall']}/{row['configs_total']} "
            f"configs perfect | {preview}..."
        )

    return q_summary


# ---------------------------------------------------------------------------
# Coverage heatmap (HTML)
# ---------------------------------------------------------------------------

def save_coverage_heatmap(df: pd.DataFrame, output_path: Path):
    """
    Create an HTML heatmap: rows = questions, columns = configs,
    cell = fraction of GT chunks retrieved (0.0 to 1.0).
    """
    pivot = (
        df.groupby(["question_id", "config"])["was_retrieved"]
        .mean()
        .reset_index()
        .pivot(index="question_id", columns="config", values="was_retrieved")
        .fillna(0)
    )

    # Sort columns by avg recall (best config on the right)
    col_order = pivot.mean().sort_values().index.tolist()
    pivot = pivot[col_order]

    # Sort rows by average recall across configs (hardest on top)
    pivot["_avg"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_avg")
    pivot = pivot.drop(columns=["_avg"])

    # Build HTML table
    html_parts = [
        "<html><head><style>",
        "body { font-family: 'Segoe UI', sans-serif; margin: 20px; }",
        "table { border-collapse: collapse; font-size: 12px; }",
        "th, td { padding: 6px 10px; text-align: center; border: 1px solid #ddd; }",
        "th { background: #333; color: white; position: sticky; top: 0; }",
        ".q-label { text-align: left; font-size: 11px; max-width: 200px; "
        "overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }",
        "</style></head><body>",
        "<h2>Coverage Heatmap: GT Chunk Recall per Question × Config</h2>",
        "<p>Cell value = fraction of ground-truth chunks retrieved. "
        "Green = 1.0 (perfect), Red = 0.0 (total miss). "
        "Sorted: hardest questions on top, weakest configs on left.</p>",
        "<table><tr><th>Q#</th>",
    ]

    for col in pivot.columns:
        html_parts.append(f"<th>{col}</th>")
    html_parts.append("</tr>")

    for q_id, row in pivot.iterrows():
        html_parts.append(f'<tr><td class="q-label">Q{q_id}</td>')
        for val in row:
            r = int(255 * (1 - val))
            g = int(255 * val)
            color = f"rgb({r},{g},80)"
            text_color = "white" if val < 0.5 else "black"
            html_parts.append(
                f'<td style="background:{color};color:{text_color}">'
                f"{val:.2f}</td>"
            )
        html_parts.append("</tr>")

    html_parts.append("</table></body></html>")

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"\nHeatmap saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build unified dataset
    print("=" * 60)
    print("Building unified dataset...")
    print("=" * 60)
    df = build_unified_dataset()
    dataset_path = OUTPUT_DIR / "phase2_dataset.csv"
    df.to_csv(dataset_path, index=False)
    print(f"Saved to {dataset_path}")

    # Run analyses
    print("\n" + "=" * 60)
    print("Running analyses...")
    print("=" * 60)

    coverage = analysis_coverage(df)
    coverage.to_csv(OUTPUT_DIR / "coverage_summary.csv", index=False)

    interactions = analysis_negative_interactions(df)
    interactions.drop(columns=["lost_details"], errors="ignore").to_csv(
        OUTPUT_DIR / "negative_interactions.csv", index=False
    )
    # Save full details as JSON for inspection
    interactions.to_json(
        OUTPUT_DIR / "negative_interactions_detail.json",
        orient="records", indent=2, force_ascii=False,
    )

    uniqueness = analysis_uniqueness(df)
    uniqueness.to_csv(OUTPUT_DIR / "retriever_uniqueness.csv", index=False)

    vulnerability = analysis_vulnerability(df)
    vulnerability.to_csv(OUTPUT_DIR / "question_vulnerability.csv", index=False)

    # Heatmap
    save_coverage_heatmap(df, OUTPUT_DIR / "coverage_heatmap.html")

    print("\n" + "=" * 60)
    print(f"All outputs saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()