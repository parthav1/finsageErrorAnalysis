import json
import os
import pandas as pd

RESULTS_DIR = os.path.join("experiments", "retriever", "result_same_num_chunks")
BASELINE_CSV = os.path.join("experiments", "retriever", "baseline_results.csv")
PER_QUESTION_CSV = os.path.join("experiments", "retriever", "per_question_analysis.csv")

def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_display_name(folder_name):
    display_names = {
        "faiss": "dense",
        "faiss_expand": "faiss_expand",
        "faiss_bm25": "bm25",
        "faiss_ts": "ts",
        "faiss_bm25_ts": "bm25_ts",
        "faiss_bm25_ts_hyde": "bm25_ts_hyde",
    }
    return display_names.get(folder_name, folder_name)


def extract_english_question(entry):
    """Read the English rewrite if it exists."""
    rewritten = entry.get("rewritten", [])
    if isinstance(rewritten, list) and rewritten:
        return str(rewritten[0])
    if isinstance(rewritten, str):
        return rewritten
    return ""


def load_result_file(result_file):
    """Load one result_2.json file and validate the basic format."""
    try:
        with open(result_file, "r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        print(f"Skipping {result_file}: invalid JSON ({error})")
        return None
    except OSError as error:
        print(f"Skipping {result_file}: could not read file ({error})")
        return None

    if not data:
        print(f"Skipping {result_file}: empty JSON file")
        return None

    if not isinstance(data, list):
        print(f"Skipping {result_file}: expected a list at the top level")
        return None

    return data


def collect_results(results_dir):
    """
    Read all retriever result files from the target directory

    Returns a list of dictionaries, one per retriever
    """
    retriever_results = []

    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return retriever_results

    for folder_name in sorted(os.listdir(results_dir)):
        folder_path = os.path.join(results_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        result_file = os.path.join(folder_path, "result_2.json")
        if not os.path.isfile(result_file):
            print(f"Skipping {folder_name}: missing file at {result_file}")
            continue

        data = load_result_file(result_file)
        if data is None:
            continue

        retriever_results.append(
            {
                "folder_name": folder_name,
                "display_name": get_display_name(folder_name),
                "data": data,
            }
        )

    return retriever_results


def analyze_retriever(retriever_result):
    """
    Compute baseline metrics and question-level rows for one retriever
    """
    folder_name = retriever_result["folder_name"]
    display_name = retriever_result["display_name"]
    data = retriever_result["data"]

    summary = data[0] if isinstance(data[0], dict) else {}
    question_entries = [entry for entry in data[1:] if isinstance(entry, dict) and "question" in entry]

    if not question_entries:
        print(f"Skipping {folder_name}: no valid per-question entries found")
        return None, []

    recall_values = [to_float(entry.get("recall")) for entry in question_entries]
    precision_values = [to_float(entry.get("precision")) for entry in question_entries]
    f1_values = [to_float(entry.get("f1")) for entry in question_entries]

    baseline_row = {
        "retriever": folder_name,
        "display_name": display_name,
        "reported_avg_recall": to_float(summary.get("avg_recall")),
        "reported_avg_precision": to_float(summary.get("avg_precision")),
        "reported_avg_f1": to_float(summary.get("avg_f1")),
        "avg_recall": sum(recall_values) / len(recall_values),
        "avg_precision": sum(precision_values) / len(precision_values),
        "avg_f1": sum(f1_values) / len(f1_values),
        "success_count": sum(1 for value in recall_values if value == 1.0),
        "failure_count": sum(1 for value in recall_values if value == 0.0),
        "question_count": len(question_entries),
    }

    per_question_rows = []
    for entry in question_entries:
        per_question_rows.append(
            {
                "retriever": folder_name,
                "display_name": display_name,
                "question_zh": entry.get("question", ""),
                "question_en": extract_english_question(entry),
                "recall": to_float(entry.get("recall")),
                "precision": to_float(entry.get("precision")),
                "f1": to_float(entry.get("f1")),
                "num_recalls": entry.get("num_recalls", 0),
            }
        )

    return baseline_row, per_question_rows


def print_table(baseline_rows):
    headers = [
        "Retriever",
        "Avg Recall",
        "Avg Precision",
        "Avg F1",
        "Success Count",
        "Failure Count",
    ]

    rows = []
    for row in baseline_rows:
        rows.append(
            [
                row["display_name"],
                f"{row['avg_recall']:.4f}",
                f"{row['avg_precision']:.4f}",
                f"{row['avg_f1']:.4f}",
                str(row["success_count"]),
                str(row["failure_count"]),
            ]
        )

    if not rows:
        print("No valid retriever results were found.")
        return

    widths = []
    for index, header in enumerate(headers):
        column_width = len(header)
        for row in rows:
            column_width = max(column_width, len(row[index]))
        widths.append(column_width)

    header_line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    separator_line = "-+-".join("-" * widths[i] for i in range(len(headers)))

    print(header_line)
    print(separator_line)
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def save_results(baseline_rows, per_question_rows):
    os.makedirs(os.path.dirname(BASELINE_CSV), exist_ok=True)

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df = baseline_df[
        [
            "retriever",
            "display_name",
            "reported_avg_recall",
            "reported_avg_precision",
            "reported_avg_f1",
            "avg_recall",
            "avg_precision",
            "avg_f1",
            "success_count",
            "failure_count",
            "question_count",
        ]
    ]

    per_question_df = pd.DataFrame(per_question_rows)
    per_question_df = per_question_df[
        [
            "retriever",
            "display_name",
            "question_zh",
            "question_en",
            "recall",
            "precision",
            "f1",
            "num_recalls",
        ]
    ]

    baseline_df.to_csv(BASELINE_CSV, index=False)
    per_question_df.to_csv(PER_QUESTION_CSV, index=False)

def main():
    retriever_results = collect_results(RESULTS_DIR)

    baseline_rows = []
    per_question_rows = []

    for retriever_result in retriever_results:
        baseline_row, question_rows = analyze_retriever(retriever_result)
        if baseline_row is None:
            continue

        baseline_rows.append(baseline_row)
        per_question_rows.extend(question_rows)

    baseline_rows.sort(key=lambda row: row["avg_recall"], reverse=True)

    print_table(baseline_rows)
    save_results(baseline_rows, per_question_rows)


if __name__ == "__main__":
    main()
