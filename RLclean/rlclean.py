"""
RLclean: Deep RL-based Integrated Error Detection and Repair (Information Sciences 2024)

Peng et al., "RLclean: An Unsupervised Integrated Data Cleaning Framework
Based on Deep Reinforcement Learning", Information Sciences 682, 2024.

Implements the RLclean-OP variant: a DQN agent learns sequences of
column-level repair operations by observing per-column FD violation
rates as state and minimizing total violations as reward signal.
Training is fully unsupervised — only the dirty data and DC rules are used.

Requires: PyTorch (available in the project conda environment)
"""

import os
import re
import sys
import copy
import time
import random
import signal
import argparse
import warnings
from collections import deque, defaultdict
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


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


# ---------------------------------------------------------------------------
# DQN Network
# ---------------------------------------------------------------------------

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# RLclean Environment
# ---------------------------------------------------------------------------

class DataCleaningEnv:
    """
    State  : normalised per-column FD violation rates (dim = n_rhs_cols + 1)
    Actions: for each column, two operations plus a terminate action
             action = 2 * col_idx + 0  → FD-based modal repair
             action = 2 * col_idx + 1  → global mode repair
             action = 2 * n_cols       → terminate
    Reward : reduction in FD violation fraction after the action
    """

    def __init__(self, dirty_df, schema, fds):
        self.original_df = dirty_df.copy()
        self.schema = schema
        self.fds = fds
        self.n_cols = len(schema)
        # Action space: 2 ops × n_cols + 1 terminate
        self.n_actions = 2 * self.n_cols + 1
        self.terminate_action = 2 * self.n_cols
        self.reset()

    def reset(self):
        self.df = self.original_df.copy()
        self.total_violations = self._count_violations(self.df)
        return self._get_state()

    def _count_violations(self, df):
        total = 0
        for lhs, rhs in self.fds:
            if lhs not in self.schema or rhs not in self.schema:
                continue
            for _, group in df.groupby(lhs):
                counts = group[rhs].value_counts()
                if len(counts) > 1:
                    # All non-modal rows are violations
                    total += int(counts.iloc[1:].sum())
        return total

    def _col_violations(self, df):
        """Per-column violation counts (for RHS columns)."""
        col_viol = defaultdict(int)
        for lhs, rhs in self.fds:
            if lhs not in self.schema or rhs not in self.schema:
                continue
            for _, group in df.groupby(lhs):
                counts = group[rhs].value_counts()
                if len(counts) > 1:
                    col_viol[rhs] += int(counts.iloc[1:].sum())
        return col_viol

    def _get_state(self):
        n = len(self.df) or 1
        col_viol = self._col_violations(self.df)
        state = np.zeros(self.n_cols + 1, dtype=np.float32)
        for j, col in enumerate(self.schema):
            state[j] = col_viol.get(col, 0) / n
        state[-1] = self._count_violations(self.df) / (n * len(self.fds) + 1e-10)
        return state

    def _fd_modal_repair(self, col_idx):
        """Replace violating cells in col with the modal value given LHS."""
        col = self.schema[col_idx]
        relevant_fds = [(lhs, rhs) for lhs, rhs in self.fds if rhs == col]
        if not relevant_fds:
            return
        for lhs, rhs in relevant_fds:
            if lhs not in self.schema:
                continue
            for lhs_val, group in self.df.groupby(lhs):
                counts = group[rhs].value_counts()
                if len(counts) <= 1:
                    continue
                modal = counts.index[0]
                violation_mask = self.df.loc[group.index, rhs] != modal
                self.df.loc[group.index[violation_mask], rhs] = modal

    def _global_mode_repair(self, col_idx):
        """Replace violating cells with the global mode of the column."""
        col = self.schema[col_idx]
        mode_vals = self.df[col].mode()
        if len(mode_vals) == 0:
            return
        global_mode = mode_vals.iloc[0]
        relevant_fds = [(lhs, rhs) for lhs, rhs in self.fds if rhs == col]
        if not relevant_fds:
            return
        for lhs, rhs in relevant_fds:
            if lhs not in self.schema:
                continue
            for _, group in self.df.groupby(lhs):
                counts = group[rhs].value_counts()
                if len(counts) <= 1:
                    continue
                violation_mask = self.df.loc[group.index, rhs] != counts.index[0]
                self.df.loc[group.index[violation_mask], rhs] = global_mode

    def step(self, action):
        prev_violations = self._count_violations(self.df)

        done = False
        if action == self.terminate_action:
            reward = 1.0 if prev_violations == 0 else -0.1
            done = True
        else:
            col_idx = action // 2
            op = action % 2
            if col_idx >= self.n_cols:
                reward = -0.01
            elif op == 0:
                self._fd_modal_repair(col_idx)
                new_viol = self._count_violations(self.df)
                reward = (prev_violations - new_viol) / (prev_violations + 1e-10)
            else:
                self._global_mode_repair(col_idx)
                new_viol = self._count_violations(self.df)
                reward = (prev_violations - new_viol) / (prev_violations + 1e-10)

        next_state = self._get_state()
        return next_state, reward, done


# ---------------------------------------------------------------------------
# RLclean Agent
# ---------------------------------------------------------------------------

class RLcleanAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.95,
                 buffer_capacity=1000, batch_size=32, target_update_freq=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax().item())

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# ---------------------------------------------------------------------------
# RLclean Main Class
# ---------------------------------------------------------------------------

class RLclean:
    def __init__(self, dirty_path, clean_path, rule_path,
                 n_episodes=50, max_steps=30, epsilon_start=1.0,
                 epsilon_min=0.1, epsilon_decay=0.02):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for RLclean.\n"
                "Install with: conda install pytorch or pip install torch"
            )

        self.dirty_df = pd.read_csv(dirty_path).astype(str).fillna("nan")
        self.clean_df = pd.read_csv(clean_path).astype(str).fillna("nan")
        self.schema = list(self.dirty_df.columns)
        self.rule_path = rule_path

        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

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
    # FD parsing
    # ------------------------------------------------------------------

    def _parse_fds(self):
        fds = []
        if not self.rule_path or not os.path.exists(self.rule_path):
            return fds
        try:
            with open(self.rule_path, "r", encoding="utf-8-sig") as f:
                lines = [l.strip() for l in f if l.strip()]
        except Exception:
            return fds

        for line in lines:
            if "EQ(" in line or "IQ(" in line:
                eq_cols = re.findall(r"EQ\(t[12]\.(\w+),t[12]\.(\w+)\)", line)
                iq_cols = re.findall(r"IQ\(t[12]\.(\w+),t[12]\.(\w+)\)", line)
                lhs_set = {a for a, b in eq_cols if a == b and a in self.schema}
                rhs_set = {a for a, b in iq_cols if a == b and a in self.schema}
                for lhs in lhs_set:
                    for rhs in rhs_set:
                        if lhs != rhs:
                            fds.append((lhs, rhs))
            elif "=>" in line or "⇒" in line:
                parts = re.split(r"=>|⇒", line)
                if len(parts) == 2:
                    lhs, rhs = parts[0].strip(), parts[1].strip()
                    if lhs in self.schema and rhs in self.schema:
                        fds.append((lhs, rhs))
        return list(set(fds))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def run(self):
        print("Parsing FD rules...")
        fds = self._parse_fds()
        print(f"  {len(fds)} FDs loaded.")

        if not fds:
            print("No FDs found — writing dirty data as repaired.")
            self.repaired_df = self.dirty_df.copy()
            return

        env = DataCleaningEnv(self.dirty_df, self.schema, fds)
        state_dim = env.n_cols + 1
        action_dim = env.n_actions

        agent = RLcleanAgent(state_dim, action_dim)

        print(f"Training DQN: {self.n_episodes} episodes, up to {self.max_steps} steps each.")
        epsilon = self.epsilon_start

        for episode in range(self.n_episodes):
            state = env.reset()
            episode_reward = 0.0

            for step in range(self.max_steps):
                action = agent.select_action(state, epsilon)
                next_state, reward, done = env.step(action)
                agent.buffer.push(state, action, reward, next_state, done)
                agent.update()
                state = next_state
                episode_reward += reward
                if done:
                    break

            epsilon = max(self.epsilon_min, epsilon - self.epsilon_decay)
            if (episode + 1) % 10 == 0:
                total_viol = env._count_violations(env.df)
                print(f"  Episode {episode + 1}/{self.n_episodes}  "
                      f"reward={episode_reward:.3f}  violations={total_viol}  ε={epsilon:.2f}")

        # --- Apply learned policy to produce final repaired data ---
        print("Applying learned policy...")
        state = env.reset()
        for _ in range(self.max_steps):
            action = agent.select_action(state, epsilon=0.0)  # greedy
            next_state, _, done = env.step(action)
            state = next_state
            if done:
                break

        final_df = env.df
        # Record repaired cells by comparing with original dirty data
        for i in range(len(self.dirty_df)):
            for j in range(len(self.schema)):
                old_val = self.dirty_df.iloc[i, j]
                new_val = final_df.iloc[i, j]
                if old_val != new_val:
                    self.repaired_cells.append((i, j))
                    self.repaired_cells_value[(i, j)] = new_val
                    if new_val == self.clean_df.iloc[i, j]:
                        self.repair_right_cells.append((i, j))

        self.repaired_df = final_df
        print(f"  {len(self.repaired_cells)} cells changed by policy.")

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

            ed_path = f"./Exp_result/rlclean/{dataset}/onlyED_{task_name}{suffix}.txt"
            os.makedirs(os.path.dirname(ed_path), exist_ok=True)
            with open(ed_path, "w") as fh:
                fh.write(f"{pre_ed}\n{rec_ed}\n{f1_ed}\n{elapsed}")

            rec_right = sum(1 for c in self.wrong_cells if c in self.repair_right_cells)
            rep_right = len(self.repair_right_cells)
            rep_total = len(self.repaired_cells)
            pre_ec = rep_right / (rep_total + 1e-10)
            rec_ec = rec_right / (len(self.wrong_cells) + 1e-10)
            f1_ec = 2 * pre_ec * rec_ec / (pre_ec + rec_ec + 1e-10)

            ec_path = f"./Exp_result/rlclean/{dataset}/oriED+EC_{task_name}{suffix}.txt"
            res_path = f"./Repaired_res/rlclean/{dataset}/repaired_{task_name}{suffix}.csv"
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

            ec_path = f"./Exp_result/rlclean/{dataset}/perfectED+EC_{task_name}{suffix}.txt"
            res_path = f"./Repaired_res/rlclean/{dataset}/perfect_repaired_{task_name}{suffix}.csv"
            os.makedirs(os.path.dirname(ec_path), exist_ok=True)
            os.makedirs(os.path.dirname(res_path), exist_ok=True)

            self.repaired_df.to_csv(res_path, index=False)
            with open(ec_path, "w") as fh:
                fh.write(f"{pre_ec}\n{rec_ec}\n{f1_ec}\n{elapsed}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLclean: DQN-based data cleaning")
    parser.add_argument("--clean_path", type=str, required=True)
    parser.add_argument("--dirty_path", type=str, required=True)
    parser.add_argument("--rule_path", type=str, default=None)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--onlyed", type=int, default=0)
    parser.add_argument("--perfected", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=30)
    args = parser.parse_args()

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(24 * 3600)

    try:
        start_time = time.time()
        cleaner = RLclean(
            args.dirty_path,
            args.clean_path,
            args.rule_path,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
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
                f"{datetime.now()} RLclean: {args.task_name} {args.dirty_path}\n"
            )
