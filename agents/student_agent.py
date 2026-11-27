"""Student agent for COMP 424 Goblet Gobblers.

Organized according to ARCHITECTURE.md:
  - Shared Constants & Move Encoding
  - Section 1: Fast Game State, Move Generation & Terminal Logic (Person 1 — currently minimal helpers.py stub)
  - Section 3: Heuristic Evaluation & Static Move Ordering (Person 3 — currently minimal 0.0 stub)
  - Section 2: Search Engine, Pruning, Time Management & StudentAgent (Person 2)
"""
from __future__ import annotations

import time
from agents.agent import Agent
import constants
from constants import (
    BOARD_SIZE,
    SMALL,
    MEDIUM,
    LARGE,
    EMPTY,
    PLAYER_1,
    PLAYER_2,
    OBSTACLE,
)
from helpers import (
    LINES,
    Move,
    check_endgame,
    check_winner,
    copy_state,
    execute_move,
    get_valid_moves,
    top_view,
)
import numpy as np
from store import register_agent


# ============================================================================
# Shared Constants & Move Encoding (ARCHITECTURE.md Section 2)
# ============================================================================

WIN_SCORE = 100_000.0
MATE_THRESHOLD = 95_000.0
MAX_EVAL_SCORE = 9_000.0
INF_SCORE = 1_000_000.0

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

MOVE_SPACE = 816  # 48 placements (0..47) + 768 board moves (48..815)


def move_id(move: Move) -> int:
    """Map any legal Move on a 4x4 board with 3 sizes to a unique int in [0, 815]."""
    dr, dc = move.dest
    dest_idx = (dr << 2) | dc
    if move.src is None:
        return (move.size << 4) | dest_idx
    sr, sc = move.src
    src_idx = (sr << 2) | sc
    return 48 + (move.size << 8) + (src_idx << 4) + dest_idx


# ============================================================================
# Section 1 — Fast State, Move Generation & Terminal Logic (Person 1)
# Minimal fallback wrapper over helpers.py until Person 1 merges Section 1.
# ============================================================================

class GameState:
    """Fallback state wrapper delegating to helpers.py (Person 1 replaces with fast in-place state)."""

    def __init__(
        self,
        board: np.ndarray,
        reserves: np.ndarray,
        owners: np.ndarray,
        sizes: np.ndarray,
        obstacle: tuple[int, int],
        live_lines: tuple[tuple[tuple[int, int], ...], ...],
        player_to_move: int,
        ply: int,
        zobrist: int,
    ) -> None:
        self.board = board
        self.reserves = reserves
        self.owners = owners
        self.sizes = sizes
        self.obstacle = obstacle
        self.live_lines = live_lines
        self.player_to_move = player_to_move
        self.ply = ply
        self.zobrist = zobrist

    @classmethod
    def from_position(
        cls,
        board: np.ndarray,
        reserves: np.ndarray,
        player_to_move: int,
        ply: int = 0,
    ) -> "GameState":
        b, r = copy_state(board, reserves)
        owners, sizes = top_view(b)
        obs = (-1, -1)
        for rr in range(BOARD_SIZE):
            for cc in range(BOARD_SIZE):
                if b[rr, cc, 0] == OBSTACLE:
                    obs = (rr, cc)
                    break
            if obs[0] != -1:
                break
        live_lines = tuple(ln for ln in LINES[int(constants.WIN_LENGTH)] if obs not in ln)
        h = compute_zobrist(b, r, player_to_move)
        return cls(b, r, owners, sizes, obs, live_lines, player_to_move, ply, h)

    def get_legal_moves(self) -> list[Move]:
        return get_valid_moves(self.board, self.reserves, self.player_to_move)

    def apply_move(self, move: Move) -> tuple:
        saved = (
            self.board.copy(),
            self.reserves.copy(),
            self.owners.copy(),
            self.sizes.copy(),
            self.zobrist,
        )
        execute_move(self.board, self.reserves, self.player_to_move, move)
        self.owners, self.sizes = top_view(self.board)
        self.ply += 1
        self.player_to_move = 3 - self.player_to_move
        self.zobrist = compute_zobrist(self.board, self.reserves, self.player_to_move)
        return saved

    def undo_move(self, move: Move, undo_token: tuple) -> None:
        b, r, owners, sizes, prev_z = undo_token
        self.board = b
        self.reserves = r
        self.owners = owners
        self.sizes = sizes
        self.zobrist = prev_z
        self.ply -= 1
        self.player_to_move = 3 - self.player_to_move

    def check_winner_after_last_move(self) -> int:
        return check_winner(self.board, 3 - self.player_to_move)

    def check_terminal(self, legal_moves: list[Move] | None = None) -> tuple[bool, int, str]:
        return check_endgame(self.board, self.reserves, self.player_to_move, self.ply)


# ============================================================================
# Section 3 — Heuristic Evaluation & Static Move Ordering (Person 3)
# Minimal neutral stubs until Person 3 merges Section 3.
# ============================================================================

def evaluate(state: GameState, perspective_player: int) -> float:
    """Placeholder static evaluation (Person 3 implements domain heuristics here)."""
    return 0.0


def score_moves_static(state: GameState, moves: list[Move]) -> list[float]:
    """Placeholder static move ordering (Person 3 implements move priors here)."""
    return [0.0] * len(moves)


# ============================================================================
# Section 2 — Search Engine, Pruning, Time Management & StudentAgent (Person 2)
# ============================================================================

_CELL_COORDS = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3),
    (3, 0), (3, 1), (3, 2), (3, 3),
)
_ZOBRIST_CACHE: dict[str, object] = {}


def _get_zobrist_tables() -> tuple[list, list, int]:
    """Lazily build deterministic 64-bit Zobrist hash tables once per process."""
    if "tables" in _ZOBRIST_CACHE:
        return _ZOBRIST_CACHE["tables"]  # type: ignore[return-value]

    rng = np.random.default_rng(0xC0424_2026)
    mask64 = (1 << 64) - 1

    raw_board = rng.integers(1, (1 << 63) - 1, size=(16, 3, 4), dtype=np.uint64)
    z_board = [
        [[int(raw_board[cell, s, p]) & mask64 for p in range(4)] for s in range(3)]
        for cell in range(16)
    ]

    raw_res = rng.integers(1, (1 << 63) - 1, size=(3, 3, 5), dtype=np.uint64)
    z_res = [
        [[int(raw_res[p, s, cnt]) & mask64 for cnt in range(5)] for s in range(3)]
        for p in range(3)
    ]

    z_turn = int(rng.integers(1, (1 << 63) - 1, dtype=np.uint64)) & mask64
    tables = (z_board, z_res, z_turn)
    _ZOBRIST_CACHE["tables"] = tables
    return tables


def compute_zobrist(board: np.ndarray, reserves: np.ndarray, player_to_move: int) -> int:
    """Compute full 64-bit Zobrist hash from (board, reserves, player_to_move)."""
    z_board, z_res, z_turn = _get_zobrist_tables()
    h = 0
    for cell_idx, (rr, cc) in enumerate(_CELL_COORDS):
        if board[rr, cc, 0] == OBSTACLE:
            h ^= z_board[cell_idx][0][OBSTACLE]
            continue
        for s in (SMALL, MEDIUM, LARGE):
            p = int(board[rr, cc, s])
            if p != EMPTY:
                h ^= z_board[cell_idx][s][p]
    for p in (PLAYER_1, PLAYER_2):
        for s in (SMALL, MEDIUM, LARGE):
            cnt = int(reserves[p, s])
            if 0 <= cnt < 5:
                h ^= z_res[p][s][cnt]
    if player_to_move == PLAYER_2:
        h ^= z_turn
    return h


class SearchTimeout(Exception):
    """Raised internally when the step wall-clock budget is exhausted."""


class TimeManager:
    """Manages soft (next-iteration) and hard (mid-search abort) wall-clock budgets."""

    __slots__ = ("start_time", "hard_limit", "soft_limit", "hard_deadline", "soft_deadline")

    def __init__(self, start_time: float, step_limit: float, ply: int) -> None:
        self.start_time = start_time
        safety = min(0.32, step_limit * 0.16)
        hard_budget = max(0.04, step_limit - safety)

        phase_scale = 0.85 if ply < 2 else 1.0
        self.hard_limit = hard_budget
        self.soft_limit = hard_budget * 0.42 * phase_scale
        self.hard_deadline = start_time + self.hard_limit
        self.soft_deadline = start_time + self.soft_limit

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time

    def can_start_next_depth(self) -> bool:
        return time.perf_counter() < self.soft_deadline

    def check_hard_deadline(self) -> None:
        if time.perf_counter() >= self.hard_deadline:
            raise SearchTimeout


class TranspositionTable:
    """Fixed-capacity Zobrist transposition table (262,144 slots, <35 MB RSS) with age replacement."""

    __slots__ = ("size", "mask", "keys", "depths", "scores", "flags", "moves", "ages", "age", "hits")

    def __init__(self, power: int = 18) -> None:
        self.size = 1 << power
        self.mask = self.size - 1
        self.keys = [0] * self.size
        self.depths = [-1] * self.size
        self.scores = [0.0] * self.size
        self.flags = [TT_EXACT] * self.size
        self.moves: list[Move | None] = [None] * self.size
        self.ages = [0] * self.size
        self.age = 0
        self.hits = 0

    def new_search(self) -> None:
        self.age = (self.age + 1) & 0xFFFF
        self.hits = 0

    def probe(self, zobrist: int) -> tuple[int, float, int, Move | None] | None:
        idx = zobrist & self.mask
        if self.keys[idx] == zobrist and self.depths[idx] >= 0:
            self.hits += 1
            return (self.depths[idx], self.scores[idx], self.flags[idx], self.moves[idx])
        return None

    def store(self, zobrist: int, depth: int, score: float, flag: int, best_move: Move | None) -> None:
        idx = zobrist & self.mask
        existing_key = self.keys[idx]
        if (
            existing_key == 0
            or self.ages[idx] != self.age
            or existing_key == zobrist
            or depth >= self.depths[idx]
        ):
            if existing_key == zobrist and best_move is None:
                best_move = self.moves[idx]
            self.keys[idx] = zobrist
            self.depths[idx] = depth
            self.scores[idx] = score
            self.flags[idx] = flag
            self.moves[idx] = best_move
            self.ages[idx] = self.age


def _to_tt_score(score: float, root_dist: int) -> float:
    if score >= MATE_THRESHOLD:
        return score + root_dist
    if score <= -MATE_THRESHOLD:
        return score - root_dist
    return score


def _from_tt_score(score: float, root_dist: int) -> float:
    if score >= MATE_THRESHOLD:
        return score - root_dist
    if score <= -MATE_THRESHOLD:
        return score + root_dist
    return score


class SearchEngine:
    """Iterative Deepening Principal Variation Search (Negamax Alpha-Beta) engine."""

    NULL_WIN_EPS = 0.05
    ASPIRATION_DELTA = 140.0
    MAX_PLY_DEPTH = 64

    def __init__(self, tt: TranspositionTable) -> None:
        self.tt = tt
        self.killers: list[list[Move | None]] = [[None, None] for _ in range(self.MAX_PLY_DEPTH + 4)]
        self.history: list[list[int]] = [[0] * MOVE_SPACE for _ in range(3)]
        self.nodes = 0
        self.beta_cutoffs = 0
        self.first_move_cutoffs = 0
        self.selective_depth = 0
        self.contempt = 0.0
        self.timer: TimeManager | None = None
        self.path_hashes: set[int] = set()
        self.game_history: dict[int, int] = {}
        self._partial_root_best: tuple[Move, float] | None = None

    def _decay_tables(self) -> None:
        for p in (PLAYER_1, PLAYER_2):
            row = self.history[p]
            for i in range(MOVE_SPACE):
                row[i] >>= 2
        for k in self.killers:
            k[0] = None
            k[1] = None

    def _record_killer_and_history(self, state: GameState, move: Move, depth: int, root_dist: int) -> None:
        dr, dc = move.dest
        is_gobble = int(state.owners[dr, dc]) != EMPTY
        if not is_gobble and root_dist < len(self.killers):
            slot = self.killers[root_dist]
            if slot[0] != move:
                slot[1] = slot[0]
                slot[0] = move
        mid = move_id(move)
        bonus = depth * depth
        val = self.history[state.player_to_move][mid] + bonus
        self.history[state.player_to_move][mid] = val if val < 500_000 else 500_000

    def _order_moves(
        self,
        state: GameState,
        moves: list[Move],
        tt_move: Move | None,
        root_dist: int,
    ) -> list[Move]:
        if len(moves) <= 1:
            return moves

        static_scores = score_moves_static(state, moves)
        hist_row = self.history[state.player_to_move]
        k0 = self.killers[root_dist][0] if root_dist < len(self.killers) else None
        k1 = self.killers[root_dist][1] if root_dist < len(self.killers) else None

        scored: list[tuple[float, int, Move]] = []
        for idx, mv in enumerate(moves):
            if tt_move is not None and mv == tt_move:
                pri = 10_000_000.0
            else:
                pri = static_scores[idx] + (hist_row[move_id(mv)] * 0.02)
                if k0 is not None and mv == k0:
                    pri += 45_000.0
                elif k1 is not None and mv == k1:
                    pri += 35_000.0
            scored.append((pri, -idx, mv))

        scored.sort(reverse=True)
        return [item[2] for item in scored]

    def search(
        self,
        state: GameState,
        timer: TimeManager,
        game_history: dict[int, int],
    ) -> tuple[Move, dict]:
        self.timer = timer
        self.game_history = game_history
        self.tt.new_search()
        self._decay_tables()

        self.nodes = 0
        self.beta_cutoffs = 0
        self.first_move_cutoffs = 0
        self.selective_depth = 0

        root_player = state.player_to_move
        root_opponent = 3 - root_player
        all_moves = state.get_legal_moves()
        fallback_move = all_moves[0]

        if len(all_moves) == 1:
            return fallback_move, self._make_telemetry(state.ply, 1, 0.0, timer.elapsed())

        # 1. Root 1-ply tactical pre-pass: play immediate win, prune immediate uncover self-losses
        safe_moves: list[Move] = []
        for mv in all_moves:
            tok = state.apply_move(mv)
            self.nodes += 1
            w = state.check_winner_after_last_move()
            state.undo_move(mv, tok)
            if w == root_player:
                return mv, self._make_telemetry(state.ply, 1, WIN_SCORE - 1.0, timer.elapsed())
            if w != root_opponent:
                safe_moves.append(mv)

        candidate_moves = safe_moves if safe_moves else all_moves
        best_move = candidate_moves[0]
        if len(candidate_moves) == 1:
            return best_move, self._make_telemetry(state.ply, 1, 0.0, timer.elapsed())

        # Anti-repetition contempt derived from root static evaluation
        root_eval = round(evaluate(state, root_player), 1)
        if root_eval > 60.0:
            self.contempt = 35.0
        elif root_eval < -60.0:
            self.contempt = -35.0
        else:
            self.contempt = 5.0

        tt_entry = self.tt.probe(state.zobrist)
        root_tt_move = tt_entry[3] if tt_entry is not None else None
        ordered_root = self._order_moves(state, candidate_moves, root_tt_move, 0)
        best_move = ordered_root[0]
        best_score = root_eval
        completed_depth = 0

        max_depth = min(self.MAX_PLY_DEPTH, max(1, constants.MAX_PLIES - state.ply))
        self.path_hashes = {state.zobrist}

        for depth in range(1, max_depth + 1):
            if depth > 1 and not timer.can_start_next_depth():
                break

            self._partial_root_best = None
            try:
                if depth >= 3 and abs(best_score) < MATE_THRESHOLD:
                    alpha = best_score - self.ASPIRATION_DELTA
                    beta = best_score + self.ASPIRATION_DELTA
                    iter_move, iter_score = self._search_root(state, ordered_root, depth, alpha, beta)
                    if iter_score <= alpha:
                        self._partial_root_best = None
                        iter_move, iter_score = self._search_root(state, ordered_root, depth, -INF_SCORE, beta)
                    elif iter_score >= beta:
                        iter_move, iter_score = self._search_root(state, ordered_root, depth, alpha, INF_SCORE)
                else:
                    iter_move, iter_score = self._search_root(state, ordered_root, depth, -INF_SCORE, INF_SCORE)

                best_move = iter_move
                best_score = iter_score
                completed_depth = depth

                ordered_root = [best_move] + [m for m in ordered_root if m != best_move]

                if abs(best_score) >= MATE_THRESHOLD:
                    break

            except SearchTimeout:
                if self._partial_root_best is not None:
                    best_move, best_score = self._partial_root_best
                break

        return best_move, self._make_telemetry(state.ply, completed_depth, best_score, timer.elapsed())

    def _search_root(
        self,
        state: GameState,
        root_moves: list[Move],
        depth: int,
        alpha: float,
        beta: float,
    ) -> tuple[Move, float]:
        assert self.timer is not None
        alpha_orig = alpha
        best_move = root_moves[0]
        best_score = -INF_SCORE

        for idx, mv in enumerate(root_moves):
            self.timer.check_hard_deadline()
            tok = state.apply_move(mv)
            self.nodes += 1
            child_z = state.zobrist
            try:
                is_rep = (child_z in self.path_hashes) or (self.game_history.get(child_z, 0) >= 1)
                if is_rep:
                    score = -self.contempt
                else:
                    self.path_hashes.add(child_z)
                    try:
                        if idx == 0:
                            score = -self._negamax(state, depth - 1, -beta, -alpha, 1)
                        else:
                            score = -self._negamax(state, depth - 1, -alpha - self.NULL_WIN_EPS, -alpha, 1)
                            if score > alpha and score < beta:
                                score = -self._negamax(state, depth - 1, -beta, -alpha, 1)
                    finally:
                        self.path_hashes.discard(child_z)
            finally:
                state.undo_move(mv, tok)

            if score > best_score:
                best_score = score
                best_move = mv
            if score > alpha:
                alpha = score
            if best_score > alpha_orig and best_score < beta:
                self._partial_root_best = (best_move, best_score)
            if alpha >= beta:
                self.beta_cutoffs += 1
                if idx == 0:
                    self.first_move_cutoffs += 1
                self._record_killer_and_history(state, mv, depth, 0)
                break

        flag = TT_EXACT
        if best_score <= alpha_orig:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        self.tt.store(state.zobrist, depth, _to_tt_score(best_score, 0), flag, best_move)
        return best_move, best_score

    def _negamax(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        root_dist: int,
    ) -> float:
        if (self.nodes & 63) == 0 and self.timer is not None:
            self.timer.check_hard_deadline()

        if root_dist > self.selective_depth:
            self.selective_depth = root_dist

        # 1. Terminal line & ply-cap checks before transposition lookup or move generation
        winner = state.check_winner_after_last_move()
        if winner != 0:
            if winner == state.player_to_move:
                return WIN_SCORE - root_dist
            return -WIN_SCORE + root_dist

        if state.ply >= constants.MAX_PLIES:
            return 0.0

        max_possible = WIN_SCORE - (root_dist + 1)
        if beta > max_possible:
            beta = max_possible
            if alpha >= beta:
                return beta

        # 2. Transposition Table probe
        alpha_orig = alpha
        zobrist = state.zobrist
        tt_entry = self.tt.probe(zobrist)
        tt_move: Move | None = None
        if tt_entry is not None:
            tt_depth, raw_tt_score, tt_flag, tt_move = tt_entry
            if tt_depth >= depth:
                tt_score = _from_tt_score(raw_tt_score, root_dist)
                if tt_flag == TT_EXACT:
                    return tt_score
                if tt_flag == TT_LOWER and tt_score > alpha:
                    alpha = tt_score
                elif tt_flag == TT_UPPER and tt_score < beta:
                    beta = tt_score
                if alpha >= beta:
                    return tt_score

        # 3. Depth cutoff -> check terminal no-moves/exhausted via GameState.check_terminal(), else evaluate()
        if depth <= 0:
            is_over, term_winner, _ = state.check_terminal()
            if is_over:
                if term_winner == 0:
                    return 0.0
                return WIN_SCORE - root_dist if term_winner == state.player_to_move else -WIN_SCORE + root_dist
            return round(evaluate(state, state.player_to_move), 1)

        # 4. Generate & order legal moves
        moves = state.get_legal_moves()
        if not moves:
            is_over, term_winner, _ = state.check_terminal(moves)
            if is_over:
                if term_winner == 0:
                    return 0.0
                return WIN_SCORE - root_dist if term_winner == state.player_to_move else -WIN_SCORE + root_dist

            # Forced pass when ALLOW_MOVES is False and opponent still has placements
            _, _, z_turn = _get_zobrist_tables()
            prev_z = state.zobrist
            state.zobrist = prev_z ^ z_turn
            state.ply += 1
            state.player_to_move = 3 - state.player_to_move
            self.nodes += 1
            try:
                return -self._negamax(state, depth - 1, -beta, -alpha, root_dist + 1)
            finally:
                state.zobrist = prev_z
                state.ply -= 1
                state.player_to_move = 3 - state.player_to_move

        ordered = self._order_moves(state, moves, tt_move, root_dist)
        best_score = -INF_SCORE
        best_move = ordered[0]
        used_cycle_score = False

        node_contempt = self.contempt if (root_dist & 1) == 0 else -self.contempt

        for idx, mv in enumerate(ordered):
            tok = state.apply_move(mv)
            self.nodes += 1
            child_z = state.zobrist
            try:
                if child_z in self.path_hashes:
                    score = -node_contempt
                    cycle_hit = True
                else:
                    cycle_hit = False
                    self.path_hashes.add(child_z)
                    try:
                        if idx == 0:
                            score = -self._negamax(state, depth - 1, -beta, -alpha, root_dist + 1)
                        else:
                            score = -self._negamax(state, depth - 1, -alpha - self.NULL_WIN_EPS, -alpha, root_dist + 1)
                            if score > alpha and score < beta:
                                score = -self._negamax(state, depth - 1, -beta, -alpha, root_dist + 1)
                    finally:
                        self.path_hashes.discard(child_z)
            finally:
                state.undo_move(mv, tok)

            if score > best_score:
                best_score = score
                best_move = mv
                used_cycle_score = cycle_hit
            if score > alpha:
                alpha = score
            if alpha >= beta:
                self.beta_cutoffs += 1
                if idx == 0:
                    self.first_move_cutoffs += 1
                self._record_killer_and_history(state, mv, depth, root_dist)
                break

        if not used_cycle_score:
            flag = TT_EXACT
            if best_score <= alpha_orig:
                flag = TT_UPPER
            elif best_score >= beta:
                flag = TT_LOWER
            self.tt.store(zobrist, depth, _to_tt_score(best_score, root_dist), flag, best_move)

        return best_score

    def _make_telemetry(self, ply: int, completed_depth: int, root_score: float, elapsed_s: float) -> dict:
        return {
            "ply": ply,
            "completed_depth": completed_depth,
            "selective_depth": max(completed_depth, self.selective_depth),
            "nodes": self.nodes,
            "nps": round(self.nodes / elapsed_s, 1) if elapsed_s > 1e-6 else 0.0,
            "tt_hits": self.tt.hits,
            "beta_cutoffs": self.beta_cutoffs,
            "first_move_cutoffs": self.first_move_cutoffs,
            "root_score": round(root_score, 2),
            "elapsed_s": round(elapsed_s, 4),
        }


@register_agent("student_agent")
class StudentAgent(Agent):
    """Iterative-deepening PVS Alpha-Beta agent with Zobrist TT and strict time control."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "StudentAgent"
        _get_zobrist_tables()
        self.turn_count = 0
        self.tt = TranspositionTable(power=18)
        self.engine = SearchEngine(self.tt)
        self.game_history: dict[int, int] = {}
        self.last_search_info: dict = {}

    def _infer_ply(self, board: np.ndarray, player: int) -> int:
        base_ply = 2 * self.turn_count + (player - 1)
        if constants.ALLOW_MOVES:
            return base_ply
        placed = int(np.count_nonzero((board == PLAYER_1) | (board == PLAYER_2)))
        return max(base_ply, placed)

    def step(self, board: np.ndarray, reserves: np.ndarray, player: int, opponent: int) -> Move:
        start = time.perf_counter()
        ply = self._infer_ply(board, player)
        timer = TimeManager(start, float(constants.STEP_TIME_LIMIT), ply)

        state = GameState.from_position(board, reserves, player, ply=ply)
        self.game_history[state.zobrist] = self.game_history.get(state.zobrist, 0) + 1

        chosen_move, telemetry = self.engine.search(state, timer, self.game_history)

        tok = state.apply_move(chosen_move)
        self.game_history[state.zobrist] = self.game_history.get(state.zobrist, 0) + 1
        state.undo_move(chosen_move, tok)

        self.turn_count += 1
        self.last_search_info = telemetry
        return chosen_move
