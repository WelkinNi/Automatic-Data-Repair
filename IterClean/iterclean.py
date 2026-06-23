"""
IterClean: Iterative Data Cleaning with Large Language Models (ACM 2024)

Ni et al., "IterClean: An Iterative Data Cleaning Framework with Large
Language Models", ACM Turing Award Celebration Conference-China 2024.

Algorithm:
  1. For each column, select a representative batch of tuples via
     two-stage selection: K-means clustering + Isolation Forest.
  2. Detector role: LLM identifies anomalies in the batch and explains.
  3. Verifier role: LLM cross-checks explanations against data context
     to suppress false positives.
  4. Apply verified repairs to the working copy.
  5. Repeat for up to MAX_ITERATIONS rounds; stop early if no new repairs.

Requires: pip install anthropic scikit-learn
          export ANTHROPIC_API_KEY=<your-key>
"""

import os
import re
import sys
import json
import time
import signal
import argparse
import warnings
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import LabelEncoder
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_string(string: str):
    if re.search(r"-inner_error-", string):
        return "-inner_error-" + string[-6:-4]
    elif re.search(r"-outer_error-", string):
        return "-outer_error-" + string[-6:-4]
    elif re.search(r"-inner_outer_error-", string):
        return "-inner_outer_error-" + string[-6:-4]
    elif re.search(r"-dirty-original_error-", string):
        return "-original_error-" + string[-9:-4]


def handler(signum, frame):
    raise TimeoutError("Time exceeded")


def _encode_df(df):
    """Label-encode all columns to numeric for sklearn."""
    encoded = pd.DataFrame(index=df.index)
    le = LabelEncoder()
    for col in df.columns:
        try:
            encoded[col] = le.fit_transform(df[col].astype(str))
        except Exception:
            encoded[col] = 0
    return encoded


def _extract_json(text):
    """Try to extract the first JSON object from an LLM response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Try to find a JSON array wrapped in an outer object
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return {"errors": json.loads(match.group())}
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# IterClean
# ---------------------------------------------------------------------------

class IterClean:
    """
    Parameters
    ----------
    dirty_path : str
    clean_path : str
    model : str   Anthropic model id
    batch_size : int   Tuples per LLM call
    max_iterations : int
    """

    def __init__(self, dirty_path, clean_path,
                 model="claude-haiku-4-5-20251001",
                 batch_size=10,
                 max_iterations=2):
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError(
                "The 'anthropic' package is required for IterClean.\n"
                "Install with: pip install anthropic\n"
                "Then set: export ANTHROPIC_API_KEY=<your-key>"
            )
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for IterClean.\n"
                "Install with: pip install scikit-learn"
            )

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it before running IterClean."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_iterations = max_iterations

        self.dirty_df = pd.read_csv(dirty_path).astype(str).fillna("nan")
        self.clean_df = pd.read_csv(clean_path).astype(str).fillna("nan")
        self.schema = list(self.dirty_df.columns)

        self.repaired_df = self.dirty_df.copy()
        self.repaired_cells = []
        self.repaired_cells_value = {}
        self.repair_right_cells = []
        self.wrong_cells = []

        for i in range(len(self.dirty_df)):
            for j in range(len(self.schema)):
                if self.dirty_df.iloc[i, j] != self.clean_df.iloc[i, j]:
                    self.wrong_cells.append((i, j))

    # ------------------------------------------------------------------
    # Two-stage tuple selection
    # ------------------------------------------------------------------

    def _select_tuples(self, df):
        """Return up to self.batch_size row *positions* (0-based within df)
        using K-means (representative) + Isolation Forest (outliers)."""
        n = len(df)
        if n <= self.batch_size:
            return list(range(n))

        encoded = _encode_df(df)
        selected = set()

        # Stage 1 – K-means representatives
        n_clusters = min(5, n, self.batch_size // 2)
        if n_clusters > 1:
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            km.fit(encoded)
            for cid in range(n_clusters):
                mask = km.labels_ == cid
                idxs = np.where(mask)[0]
                if len(idxs):
                    selected.add(int(idxs[0]))

        # Stage 2 – Isolation Forest outliers
        remaining = self.batch_size - len(selected)
        if remaining > 0 and n > 1:
            iso = IsolationForest(contamination=0.15, random_state=42)
            iso.fit(encoded)
            scores = iso.score_samples(encoded)
            outlier_positions = np.argsort(scores)[:remaining]
            selected.update(int(p) for p in outlier_positions)

        return sorted(selected)[: self.batch_size]

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt, max_tokens=1024):
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def _detector_prompt(self, col_name, batch_df):
        rows_csv = batch_df.reset_index(drop=True).to_csv(index=True)
        return (
            f"You are a data quality expert analysing the column '{col_name}'.\n\n"
            f"Data (index = row position in this batch, starting at 0):\n"
            f"{rows_csv}\n\n"
            f"Identify errors or anomalies in the '{col_name}' column only.\n"
            f"Respond ONLY with valid JSON — no markdown, no commentary:\n"
            f'{{"errors": [{{"row_index": <int>, "current_value": "<str>", '
            f'"reason": "<str>", "suggested_value": "<str>"}}]}}\n'
            f"If no errors: {{\"errors\": []}}"
        )

    def _verifier_prompt(self, col_name, batch_df, detected):
        rows_csv = batch_df.reset_index(drop=True).to_csv(index=True)
        errors_str = json.dumps(detected, indent=2)
        return (
            f"You are a data quality verifier for column '{col_name}'.\n\n"
            f"Potential errors detected:\n{errors_str}\n\n"
            f"Data context (same batch):\n{rows_csv}\n\n"
            f"For each detected error, confirm whether it is genuinely erroneous.\n"
            f"Respond ONLY with valid JSON:\n"
            f'{{"verified": [{{"row_index": <int>, "confirmed": true/false, '
            f'"suggested_value": "<str>"}}]}}'
        )

    # ------------------------------------------------------------------
    # Process one column
    # ------------------------------------------------------------------

    def _process_column(self, col_name):
        """Return dict mapping actual df-row-index → suggested repair value."""
        col_idx = self.schema.index(col_name)
        repairs = {}

        positions = self._select_tuples(self.repaired_df)
        batch_df = self.repaired_df.iloc[positions].copy()

        # --- Detector ---
        try:
            detect_text = self._call_llm(self._detector_prompt(col_name, batch_df))
            detect_result = _extract_json(detect_text)
        except Exception as exc:
            print(f"    [detector] {col_name}: {exc}")
            return repairs

        if not detect_result or not detect_result.get("errors"):
            return repairs

        detected_errors = detect_result["errors"]

        # --- Verifier ---
        try:
            verify_text = self._call_llm(
                self._verifier_prompt(col_name, batch_df, detected_errors)
            )
            verify_result = _extract_json(verify_text)
        except Exception as exc:
            print(f"    [verifier] {col_name}: {exc}")
            # Fall back: trust the detector
            verify_result = {
                "verified": [
                    {
                        "row_index": e.get("row_index"),
                        "confirmed": True,
                        "suggested_value": e.get("suggested_value", ""),
                    }
                    for e in detected_errors
                ]
            }

        if not verify_result:
            return repairs

        # --- Map batch positions → actual dataframe indices ---
        for v in verify_result.get("verified", []):
            if not v.get("confirmed"):
                continue
            batch_pos = v.get("row_index")
            suggested = str(v.get("suggested_value", "")).strip()
            if batch_pos is None or not suggested:
                continue
            try:
                batch_pos = int(batch_pos)
            except (ValueError, TypeError):
                continue
            if 0 <= batch_pos < len(positions):
                actual_idx = positions[batch_pos]
                repairs[actual_idx] = suggested

        return repairs

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            any_repair = False

            for col_name in self.schema:
                print(f"  Column: {col_name}")
                col_idx = self.schema.index(col_name)
                repairs = self._process_column(col_name)

                for actual_idx, new_val in repairs.items():
                    old_val = self.repaired_df.iloc[actual_idx, col_idx]
                    if old_val == new_val:
                        continue
                    self.repaired_df.iloc[actual_idx, col_idx] = new_val
                    cell = (actual_idx, col_idx)
                    self.repaired_cells.append(cell)
                    self.repaired_cells_value[cell] = new_val
                    if new_val == self.clean_df.iloc[actual_idx, col_idx]:
                        self.repair_right_cells.append(cell)
                    any_repair = True

            print(
                f"  End of iteration {iteration + 1}: "
                f"{len(self.repaired_cells)} total repairs so far."
            )
            if not any_repair:
                print("  No new repairs — stopping early.")
                break

    # ------------------------------------------------------------------
    # Evaluation and output
    # ------------------------------------------------------------------

    def evaluate(self, task_name, dirty_path, start_time, perfected=False):
        end_time = time.time()
        elapsed = end_time - start_time

        self.repaired_cells = list(set(self.repaired_cells))
        self.repair_right_cells = list(set(self.repair_right_cells))
        self.wrong_cells = list(set(self.wrong_cells))

        dataset = task_name[:-1]
        suffix = check_string(dirty_path.split("/")[-1])

        if not perfected:
            det_right = sum(1 for c in self.repaired_cells if c in self.wrong_cells)
            pre_ed = det_right / (len(self.repaired_cells) + 1e-10)
            rec_ed = det_right / (len(self.wrong_cells) + 1e-10)
            f1_ed = 2 * pre_ed * rec_ed / (pre_ed + rec_ed + 1e-10)

            ed_path = f"./Exp_result/iterclean/{dataset}/onlyED_{task_name}{suffix}.txt"
            os.makedirs(os.path.dirname(ed_path), exist_ok=True)
            with open(ed_path, "w") as fh:
                fh.write(f"{pre_ed}\n{rec_ed}\n{f1_ed}\n{elapsed}")

            rec_right = sum(1 for c in self.wrong_cells if c in self.repair_right_cells)
            rep_right = len(self.repair_right_cells)
            rep_total = len(self.repaired_cells)
            pre_ec = rep_right / (rep_total + 1e-10)
            rec_ec = rec_right / (len(self.wrong_cells) + 1e-10)
            f1_ec = 2 * pre_ec * rec_ec / (pre_ec + rec_ec + 1e-10)

            ec_path = f"./Exp_result/iterclean/{dataset}/oriED+EC_{task_name}{suffix}.txt"
            res_path = f"./Repaired_res/iterclean/{dataset}/repaired_{task_name}{suffix}.csv"
            os.makedirs(os.path.dirname(ec_path), exist_ok=True)
            os.makedirs(os.path.dirname(res_path), exist_ok=True)

            self.repaired_df.to_csv(res_path, index=False)
            with open(ec_path, "w") as fh:
                fh.write(f"{pre_ec}\n{rec_ec}\n{f1_ec}\n{elapsed}")

            sys.stdout = open("/dev/stdout", "w")
            print(f"ED  — pre: {pre_ed:.4f}  rec: {rec_ed:.4f}  f1: {f1_ed:.4f}")
            print(f"EC  — pre: {pre_ec:.4f}  rec: {rec_ec:.4f}  f1: {f1_ec:.4f}")
            print(f"Runtime: {elapsed:.1f}s")

        else:
            rep_t = sum(1 for c in self.wrong_cells if c in self.repaired_cells)
            rec_right = sum(1 for c in self.wrong_cells if c in self.repair_right_cells)
            pre_ec = rec_right / (rep_t + 1e-10)
            rec_ec = rec_right / (len(self.wrong_cells) + 1e-10)
            f1_ec = 2 * pre_ec * rec_ec / (pre_ec + rec_ec + 1e-10)

            ec_path = f"./Exp_result/iterclean/{dataset}/perfectED+EC_{task_name}{suffix}.txt"
            res_path = f"./Repaired_res/iterclean/{dataset}/perfect_repaired_{task_name}{suffix}.csv"
            os.makedirs(os.path.dirname(ec_path), exist_ok=True)
            os.makedirs(os.path.dirname(res_path), exist_ok=True)

            self.repaired_df.to_csv(res_path, index=False)
            with open(ec_path, "w") as fh:
                fh.write(f"{pre_ec}\n{rec_ec}\n{f1_ec}\n{elapsed}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IterClean: LLM-based iterative data cleaning")
    parser.add_argument("--clean_path", type=str, required=True)
    parser.add_argument("--dirty_path", type=str, required=True)
    parser.add_argument("--rule_path", type=str, default=None)  # accepted but unused
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--onlyed", type=int, default=0)
    parser.add_argument("--perfected", type=int, default=0)
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--max_iterations", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(24 * 3600)

    try:
        start_time = time.time()
        cleaner = IterClean(
            args.dirty_path,
            args.clean_path,
            model=args.model,
            batch_size=args.batch_size,
            max_iterations=args.max_iterations,
        )
        cleaner.run()
        cleaner.evaluate(
            args.task_name, args.dirty_path, start_time, bool(args.perfected)
        )
    except TimeoutError as e:
        print(f"Time exceeded: {e} — {args.task_name} {args.dirty_path}")
        os.makedirs("./aggre_results", exist_ok=True)
        with open("./aggre_results/timeout_log.txt", "a") as fh:
            fh.write(
                f"{datetime.now()} IterClean: {args.task_name} {args.dirty_path}\n"
            )
