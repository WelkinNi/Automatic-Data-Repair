"""
BClean: Bayesian Data Cleaning (ICDE 2024)

Qin et al., "BClean: A Bayesian Data Cleaning System", ICDE 2024.

Algorithm:
  1. Discover functional dependencies from DC rule file and data mining.
  2. Build conditional probability tables (CPTs) P(rhs | lhs) from data.
  3. Detect FD-violating cells.
  4. Repair each violated cell via MAP estimation with compensatory
     co-occurrence scoring: score(b|a) = alpha*P_bayes(b|a) + (1-alpha)*P_cooc(a,b).
"""

import os
import re
import sys
import copy
import time
import signal
import argparse
import warnings
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers shared with other methods in this benchmark
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


# ---------------------------------------------------------------------------
# BClean
# ---------------------------------------------------------------------------

class BClean:
    def __init__(self, dirty_path, clean_path, rule_path, alpha=0.7):
        self.dirty_df = pd.read_csv(dirty_path).astype(str).fillna("nan")
        self.clean_df = pd.read_csv(clean_path).astype(str).fillna("nan")
        self.schema = list(self.dirty_df.columns)
        self.rule_path = rule_path
        self.alpha = alpha

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
    # FD discovery
    # ------------------------------------------------------------------

    def parse_rule_fds(self):
        """Parse FDs from dc_rules_holoclean.txt (EQ=LHS, IQ=RHS) or
        dc_rules-validate-fd-horizon.txt (A => B)."""
        fds = []
        if not self.rule_path or not os.path.exists(self.rule_path):
            return fds
        try:
            with open(self.rule_path, "r", encoding="utf-8-sig") as f:
                lines = [l.strip() for l in f if l.strip()]
        except Exception:
            return fds

        for line in lines:
            # Holoclean format: t1&t2&EQ(t1.A,t2.A)&IQ(t1.B,t2.B)
            if "EQ(" in line or "IQ(" in line:
                eq_cols = re.findall(r"EQ\(t[12]\.(\w+),t[12]\.(\w+)\)", line)
                iq_cols = re.findall(r"IQ\(t[12]\.(\w+),t[12]\.(\w+)\)", line)
                lhs_candidates = [a for a, b in eq_cols if a == b and a in self.schema]
                rhs_candidates = [a for a, b in iq_cols if a == b and a in self.schema]
                for lhs in lhs_candidates:
                    for rhs in rhs_candidates:
                        if lhs != rhs:
                            fds.append((lhs, rhs))
            # Horizon FD format: A => B  or  A ⇒ B
            elif "=>" in line or "⇒" in line:
                parts = re.split(r"=>|⇒", line)
                if len(parts) == 2:
                    lhs = parts[0].strip()
                    rhs = parts[1].strip()
                    if lhs in self.schema and rhs in self.schema:
                        fds.append((lhs, rhs))
        return fds

    def mine_data_fds(self, threshold=0.95):
        """Mine pairwise FDs from the data: A→B holds if ≥threshold of
        distinct A-values each map to a single B-value."""
        fds = []
        df = self.dirty_df
        for lhs in self.schema:
            # Skip high-cardinality columns as LHS (too many unique values)
            if df[lhs].nunique() > len(df) * 0.8:
                continue
            for rhs in self.schema:
                if lhs == rhs:
                    continue
                grouped = df.groupby(lhs)[rhs].nunique()
                if (grouped == 1).mean() >= threshold:
                    fds.append((lhs, rhs))
        return fds

    # ------------------------------------------------------------------
    # Probability tables
    # ------------------------------------------------------------------

    def build_cpts(self, fds):
        """Build P(rhs | lhs) tables from the data."""
        cpts = {}
        for lhs, rhs in fds:
            key = (lhs, rhs)
            if key in cpts:
                continue
            counts = defaultdict(lambda: defaultdict(int))
            for _, row in self.dirty_df.iterrows():
                counts[row[lhs]][row[rhs]] += 1
            probs = {}
            for lhs_val, rhs_counts in counts.items():
                total = sum(rhs_counts.values())
                probs[lhs_val] = {rv: c / total for rv, c in rhs_counts.items()}
            cpts[key] = probs
        return cpts

    def build_cooccurrence(self, fds):
        """Build joint co-occurrence P(lhs_val, rhs_val) tables."""
        cooc = {}
        for lhs, rhs in fds:
            key = (lhs, rhs)
            if key in cooc:
                continue
            pair_counts = defaultdict(int)
            for _, row in self.dirty_df.iterrows():
                pair_counts[(row[lhs], row[rhs])] += 1
            total = sum(pair_counts.values()) or 1
            cooc[key] = {pair: c / total for pair, c in pair_counts.items()}
        return cooc

    # ------------------------------------------------------------------
    # Violation detection
    # ------------------------------------------------------------------

    def find_violations(self, fds):
        """Return set of (row_idx, col_idx) that violate at least one FD."""
        violations = set()
        for lhs, rhs in fds:
            rhs_idx = self.schema.index(rhs)
            for lhs_val, group in self.dirty_df.groupby(lhs):
                rhs_counts = group[rhs].value_counts()
                if len(rhs_counts) <= 1:
                    continue
                modal_rhs = rhs_counts.index[0]
                for row_idx in group.index:
                    if self.dirty_df.at[row_idx, rhs] != modal_rhs:
                        violations.add((row_idx, rhs_idx))
        return violations

    # ------------------------------------------------------------------
    # MAP repair
    # ------------------------------------------------------------------

    def compensatory_score(self, lhs_col, rhs_col, lhs_val, candidate,
                           cpts, cooc):
        p_bayes = cpts.get((lhs_col, rhs_col), {}).get(lhs_val, {}).get(candidate, 1e-10)
        p_cooc = cooc.get((lhs_col, rhs_col), {}).get((lhs_val, candidate), 1e-10)
        return self.alpha * p_bayes + (1 - self.alpha) * p_cooc

    def map_repair(self, row_idx, col_idx, fds, cpts, cooc):
        """Return the MAP-optimal repair value for (row_idx, col_idx)."""
        col = self.schema[col_idx]
        row = self.dirty_df.iloc[row_idx]

        relevant_fds = [(lhs, rhs) for lhs, rhs in fds if rhs == col]
        if not relevant_fds:
            mode = self.dirty_df[col].mode()
            return mode.iloc[0] if len(mode) > 0 else "nan"

        candidate_scores = defaultdict(float)
        for lhs, rhs in relevant_fds:
            lhs_val = row[lhs]
            rhs_probs = cpts.get((lhs, rhs), {}).get(lhs_val, {})
            for candidate in rhs_probs:
                candidate_scores[candidate] += self.compensatory_score(
                    lhs, rhs, lhs_val, candidate, cpts, cooc
                )

        if candidate_scores:
            return max(candidate_scores, key=candidate_scores.get)
        mode = self.dirty_df[col].mode()
        return mode.iloc[0] if len(mode) > 0 else "nan"

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self):
        print("Parsing FDs from rules...")
        rule_fds = self.parse_rule_fds()
        print(f"  Found {len(rule_fds)} FDs from rule file.")

        print("Mining FDs from data...")
        mined_fds = self.mine_data_fds(threshold=0.95)
        print(f"  Found {len(mined_fds)} mined FDs.")

        all_fds = list({(lhs, rhs) for lhs, rhs in rule_fds + mined_fds})
        print(f"  Total unique FDs: {len(all_fds)}")

        if not all_fds:
            print("No FDs found; outputting dirty data unchanged.")
            self.repaired_df = self.dirty_df.copy()
            return

        print("Building CPTs and co-occurrence tables...")
        cpts = self.build_cpts(all_fds)
        cooc = self.build_cooccurrence(all_fds)

        print("Detecting FD violations...")
        violations = self.find_violations(all_fds)
        print(f"  {len(violations)} violated cells found.")

        print("Applying MAP repairs...")
        self.repaired_df = self.dirty_df.copy()
        for row_idx, col_idx in violations:
            repair_val = self.map_repair(row_idx, col_idx, all_fds, cpts, cooc)
            old_val = self.dirty_df.iloc[row_idx, col_idx]
            if repair_val != old_val:
                self.repaired_df.iloc[row_idx, col_idx] = repair_val
                self.repaired_cells.append((row_idx, col_idx))
                self.repaired_cells_value[(row_idx, col_idx)] = repair_val
                if repair_val == self.clean_df.iloc[row_idx, col_idx]:
                    self.repair_right_cells.append((row_idx, col_idx))
        print(f"  Applied {len(self.repaired_cells)} repairs.")

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

        if not perfected:
            # --- Error detection metrics (treated repaired cells as detected) ---
            det_right = sum(1 for c in self.repaired_cells if c in self.wrong_cells)
            pre_ed = det_right / (len(self.repaired_cells) + 1e-10)
            rec_ed = det_right / (len(self.wrong_cells) + 1e-10)
            f1_ed = 2 * pre_ed * rec_ed / (pre_ed + rec_ed + 1e-10)

            ed_path = f"./Exp_result/bclean/{dataset}/onlyED_{task_name}{check_string(dirty_path.split('/')[-1])}.txt"
            os.makedirs(os.path.dirname(ed_path), exist_ok=True)
            with open(ed_path, "w") as f:
                f.write(f"{pre_ed}\n{rec_ed}\n{f1_ed}\n{elapsed}")

            # --- Error correction metrics ---
            rec_right = sum(1 for c in self.wrong_cells if c in self.repair_right_cells)
            rep_right = len(self.repair_right_cells)
            rep_total = len(self.repaired_cells)
            pre_ec = rep_right / (rep_total + 1e-10)
            rec_ec = rec_right / (len(self.wrong_cells) + 1e-10)
            f1_ec = 2 * pre_ec * rec_ec / (pre_ec + rec_ec + 1e-10)

            ec_path = f"./Exp_result/bclean/{dataset}/oriED+EC_{task_name}{check_string(dirty_path.split('/')[-1])}.txt"
            res_path = f"./Repaired_res/bclean/{dataset}/repaired_{task_name}{check_string(dirty_path.split('/')[-1])}.csv"
            os.makedirs(os.path.dirname(ec_path), exist_ok=True)
            os.makedirs(os.path.dirname(res_path), exist_ok=True)

            self.repaired_df.to_csv(res_path, index=False)
            with open(ec_path, "w") as f:
                f.write(f"{pre_ec}\n{rec_ec}\n{f1_ec}\n{elapsed}")

            sys.stdout = open("/dev/stdout", "w")
            print(f"ED  — pre: {pre_ed:.4f}  rec: {rec_ed:.4f}  f1: {f1_ed:.4f}")
            print(f"EC  — pre: {pre_ec:.4f}  rec: {rec_ec:.4f}  f1: {f1_ec:.4f}")
            print(f"Runtime: {elapsed:.1f}s")

        else:
            # Perfect-ED mode: only count repairs on true wrong cells
            rep_t = sum(1 for c in self.wrong_cells if c in self.repaired_cells)
            rec_right = sum(1 for c in self.wrong_cells if c in self.repair_right_cells)
            pre_ec = rec_right / (rep_t + 1e-10)
            rec_ec = rec_right / (len(self.wrong_cells) + 1e-10)
            f1_ec = 2 * pre_ec * rec_ec / (pre_ec + rec_ec + 1e-10)

            ec_path = f"./Exp_result/bclean/{dataset}/perfectED+EC_{task_name}{check_string(dirty_path.split('/')[-1])}.txt"
            res_path = f"./Repaired_res/bclean/{dataset}/perfect_repaired_{task_name}{check_string(dirty_path.split('/')[-1])}.csv"
            os.makedirs(os.path.dirname(ec_path), exist_ok=True)
            os.makedirs(os.path.dirname(res_path), exist_ok=True)

            self.repaired_df.to_csv(res_path, index=False)
            with open(ec_path, "w") as f:
                f.write(f"{pre_ec}\n{rec_ec}\n{f1_ec}\n{elapsed}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BClean: Bayesian data cleaning")
    parser.add_argument("--clean_path", type=str, required=True)
    parser.add_argument("--dirty_path", type=str, required=True)
    parser.add_argument("--rule_path", type=str, default=None)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--onlyed", type=int, default=0)
    parser.add_argument("--perfected", type=int, default=0)
    args = parser.parse_args()

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(24 * 3600)

    try:
        start_time = time.time()
        cleaner = BClean(args.dirty_path, args.clean_path, args.rule_path)
        cleaner.run()
        cleaner.evaluate(
            args.task_name, args.dirty_path, start_time, bool(args.perfected)
        )
    except TimeoutError as e:
        print(f"Time exceeded: {e} — {args.task_name} {args.dirty_path}")
        os.makedirs("./aggre_results", exist_ok=True)
        with open("./aggre_results/timeout_log.txt", "a") as f:
            f.write(f"{datetime.now()} BClean: {args.task_name} {args.dirty_path}\n")
