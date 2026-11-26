# Goblet Gobblers Agent — Team Architecture & Module Contracts

This document tracks the shared architecture, interfaces, and conventions across our three-way split. Because submission rule `C-03` forbids importing custom helper files, all production code merges into clearly separated sections inside `agents/student_agent.py`.

> **Submission reminder:** On myCourses we upload **only** `agents/student_agent.py` and `authors.yaml` (uncompressed). Do not edit any starter framework files (`helpers.py`, `constants.py`, `world.py`, `simulator.py`, etc.).

---

## 1. Ownership & File Layout Inside `agents/student_agent.py`

To avoid merge conflicts while working on separate branches, keep `agents/student_agent.py` divided into four fixed regions in this order:

1. **Shared Constants & Move Encoding (`Lines ~1–60`)** — Score bounds, `TT` flags, `move_id(move)` helper.
2. **Section 1 — Fast State, Move Generation & Terminal Logic (Person 1)** — `GameState`, `get_legal_moves`, `apply_move`, `undo_move`, `check_terminal`.
3. **Section 3 — Heuristic Evaluation & Static Move Ordering (Person 3)** — `evaluate`, `score_moves_static`.
4. **Section 2 — Search Engine, Pruning, Time Management & `StudentAgent` (Person 2)** — `TimeManager`, `TranspositionTable`, `SearchEngine`, `@register_agent("student_agent") class StudentAgent`.

Each section can ship with a minimal fallback stub while in development so any teammate can run `validate_submission.py` and `simulator.py` independently on their branch.

---

## 2. Shared Conventions (Read Before Implementing)

### 2.1 Rule Flags (`C-13`)
- Never hard-code `ALLOW_MOVES = True` or `WIN_LENGTH = 4`, and never assign to them.
- Always read `constants.ALLOW_MOVES` and `constants.WIN_LENGTH` (e.g., via `helpers.LINES[constants.WIN_LENGTH]`) so the validator passes even if tested under alternate flags (`win_length=3` or `allow_moves=False`).

### 2.2 Score Scale & Perspective
All search and evaluation scores are floats with strict non-overlapping ranges:

| Score Range | Meaning | Owner |
|---|---|---|
| `+100_000.0 - root_dist` | Forced / terminal win (`root_dist` = plies from search root, `>= +99_000`) | Person 2 |
| `[-9_000.0, +9_000.0]` | Non-terminal heuristic evaluation (clamped inside `±9_000`) | Person 3 |
| `0.0` | Exact draw (`ply >= MAX_PLIES`, `exhausted`, or path repetition) | Person 2 & 3 |
| `-100_000.0 + root_dist` | Forced / terminal loss (`<= -99_000`, prefers delaying loss) | Person 2 |

```python
WIN_SCORE = 100_000.0
MATE_THRESHOLD = 95_000.0
MAX_EVAL_SCORE = 9_000.0
```

### 2.3 Move Representation & Fast Indexing
- Pass standard `helpers.Move` objects across all three modules, created with `Move._unchecked(src, dest, size)` inside Person 1's generator to skip dataclass `__post_init__` overhead (`~3x` faster).
- For array-indexed search tables (Killer Moves, History Heuristic) without dict hashing overhead, use this deterministic `0..815` integer mapping:

```python
def move_id(move: Move) -> int:
    """Map any legal Move on a 4x4 board with 3 sizes to a unique int in [0, 815]."""
    dr, dc = move.dest
    dest_idx = (dr << 2) | dc
    if move.src is None:
        return (move.size << 4) | dest_idx                  # 0 .. 47 (placements)
    sr, sc = move.src
    src_idx = (sr << 2) | sc
    return 48 + (move.size << 8) + (src_idx << 4) + dest_idx  # 48 .. 815 (board moves)
```

---

## 3. Module Endpoints & Signatures

### 3.1 Person 1 Endpoints: Fast State, Move Gen & Terminal Checks

Person 2 and Person 3 interact with the board exclusively through `GameState` so no arrays are deep-copied during tree search.

```python
class GameState:
    # Public fields read directly by Person 2 (search) and Person 3 (eval/ordering):
    board: np.ndarray          # (4, 4, 3) int8 — mutated in place by apply_move / undo_move
    reserves: np.ndarray       # (3, 3) int8    — reserves[player, size], row 0 unused
    owners: np.ndarray         # (4, 4) int8    — top-piece owner per cell (0=EMPTY, 1, 2, 3=OBSTACLE)
    sizes: np.ndarray          # (4, 4) int8    — top-piece size per cell (-1 if empty or obstacle)
    obstacle: tuple[int, int]  # (r, c) of the blocked square
    live_lines: tuple[tuple[tuple[int, int], ...], ...]  # LINES[constants.WIN_LENGTH] excluding obstacle lines
    player_to_move: int        # 1 or 2
    ply: int                   # current game ply (0 .. 100)
    zobrist: int               # 64-bit unsigned int hash of (board, reserves, player_to_move)

    @classmethod
    def from_position(
        cls,
        board: np.ndarray,
        reserves: np.ndarray,
        player_to_move: int,
        ply: int = 0,
    ) -> "GameState":
        """Build once at the start of step() from the referee's board and reserves copies."""
        ...

    def get_legal_moves(self) -> list[Move]:
        """Return all legal moves for `self.player_to_move` (placements + board moves if ALLOW_MOVES).
        Uses Move._unchecked(src, dest, size)."""
        ...

    def apply_move(self, move: Move) -> tuple:
        """Apply `move` for `self.player_to_move` in place.
        Updates board, reserves, owners, sizes, zobrist, increments ply, and flips player_to_move.
        Returns lightweight `undo_token` needed by `undo_move`."""
        ...

    def undo_move(self, move: Move, undo_token: tuple) -> None:
        """Restore exact state prior to `apply_move(move)` in O(1) with zero array allocations."""
        ...

    def check_winner_after_last_move(self) -> int:
        """Check if the move JUST played by `last_mover = 3 - self.player_to_move` ended the game.
        CRITICAL (Uncovering Rule):
          1. Check `self.player_to_move` (opponent of last_mover) for a line FIRST -> return `self.player_to_move`.
          2. Check `last_mover` for a line SECOND -> return `last_mover`.
          3. Otherwise return 0."""
        ...

    def check_terminal(self, legal_moves: list[Move] | None = None) -> tuple[bool, int, str]:
        """Evaluate terminal status right before `self.player_to_move` acts at `self.ply`.
        Order of checks:
          1. `w = self.check_winner_after_last_move()`: if `w != 0`, return `(True, w, 'revealed' if w == self.player_to_move else 'line')`.
          2. If `self.ply >= constants.MAX_PLIES`: return `(True, 0, 'ply_cap')`.
          3. Check legal moves (reusing `legal_moves` if already generated to avoid double work):
             - If `constants.ALLOW_MOVES` and no legal moves: return `(True, 3 - self.player_to_move, 'no_moves')`.
             - If not `constants.ALLOW_MOVES` and neither player has moves: return `(True, 0, 'exhausted')`.
        Returns `(False, 0, '')` if the game continues."""
        ...
```

> **Performance note for Person 1:** Passing `legal_moves` optionally into `check_terminal` (or separating `check_winner_after_last_move()` + `ply >= MAX_PLIES` before move generation!) lets Person 2 prune terminal line/ply-cap nodes *before* spending time generating legal moves at leaf/terminal nodes.

---

### 3.2 Person 3 Endpoints: Heuristic Evaluation & Static Move Ordering

```python
def evaluate(state: GameState, perspective_player: int) -> float:
    """Static evaluation of a non-terminal `state` from `perspective_player`'s viewpoint (1 or 2).

    Requirements:
      - Positive means favorable for `perspective_player`; negative means favorable for `3 - perspective_player`.
      - Anti-symmetric: `evaluate(state, 1) == -evaluate(state, 2)` (or if side-to-move tempo bonus is included,
        always compute relative to `perspective_player` given `state.player_to_move`).
      - Return value strictly within `[-MAX_EVAL_SCORE, +MAX_EVAL_SCORE]` (`[-9000.0, +9000.0]`).
      - Must not mutate `state` (or must restore any scratch change).
    """
    ...


def score_moves_static(state: GameState, moves: list[Move]) -> list[float]:
    """Return a static priority score per move in `moves` (same length & order) for `state.player_to_move`.

    Suggested scale (Person 2 adds higher dynamic tiers on top for TT PV move & Killer moves):
      - Immediate winning move / covering opponent's 3-in-a-row threat: +2_000 .. +5_000
      - Gobbling an opponent top piece / advancing own live line:      +200 .. +1_500
      - Center / two-line intersection placement, larger size control:  +10 .. +200
      - Lifting a piece that uncovers an opponent piece underneath:    -500 .. -5_000
    """
    ...
```

---

### 3.3 Person 2 Endpoints: Search Engine, Time Management & Telemetry

Person 2 owns `StudentAgent` and exposes search telemetry on the agent instance (without calling `print()`, which triggers a validator warning) so Person 3's benchmark scripts can log search depth, nodes, and pruning efficiency in-process.

```python
@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.name = "StudentAgent"
        # Persisted across turns within a single game:
        # - self.turn_count: int (used to infer ply = 2 * turn_count + player - 1 when ALLOW_MOVES=True)
        # - self.tt: TranspositionTable (bounded <= 250k entries, ~30 MB << 500 MB limit)
        # - self.game_history: dict[int, int] (zobrist -> count, for anti-repetition contempt when winning)
        # - self.last_search_info: dict (read-only telemetry for Person 3's benchmarking harness)

    def step(self, board: np.ndarray, reserves: np.ndarray, player: int, opponent: int) -> Move:
        """Choose and return a legal helpers.Move within <= 1.70s wall-clock budget."""
        ...
```

#### Telemetry Schema (`agent.last_search_info`) for Person 3's Benchmarks
After every `step()` call, `agent.last_search_info` is populated with:

```python
{
    "ply": int,                   # inferred current ply (0..99)
    "completed_depth": int,       # deepest fully completed iterative-deepening depth
    "selective_depth": int,       # max ply distance reached (including tactical extensions)
    "nodes": int,                 # total nodes visited this step
    "nps": float,                 # nodes per second
    "tt_hits": int,               # transposition table hits
    "beta_cutoffs": int,          # total alpha-beta cutoffs
    "first_move_cutoffs": int,    # cutoffs occurring on move index 0 (measures move-ordering quality!)
    "root_score": float,          # backed-up evaluation from `player`'s perspective
    "elapsed_s": float,           # wall time spent in step()
}
```

---

## 4. Search & Time Management Design Summary (Person 2)

1. **Root Tactical Pre-Pass (`O(b)`)**:
   - Immediately play any 1-ply winning move (`check_winner_after_last_move() == player`).
   - Filter out suicide moves that immediately uncover an opponent line (`winner == opponent`), falling back to all legal moves only if every move loses immediately.
   - If only 1 safe move exists (or forced single defense), return immediately to save clock/process overhead.
2. **Iterative Deepening Negamax with Principal Variation Search (PVS)**:
   - Searches `depth = 1, 2, 3, ...` until time budget expires or a forced win/loss is proven (`abs(score) >= MATE_THRESHOLD`).
   - Uses aspiration windows around previous iteration's score, widening on fail-low/fail-high.
   - Searches move 0 with full window `[-beta, -alpha]` and subsequent moves with null window `[-alpha - 1, -alpha]`, re-searching with full window only if a move beats `alpha`.
3. **Move Ordering Tiers inside Search**:
   1. `TT` best move from previous depth / transposition (`+1_000_000`)
   2. Killer moves (`2` quiet beta-cutoff moves per ply, `+50_000` / `+40_000`) combined with Person 3's `score_moves_static` tactical scores
   3. History heuristic (`history[player][move_id] += depth * depth`) + remaining static scores
4. **Cycle / Repetition Handling**:
   - Tracks `path_hashes` along the active game + search branch.
   - Repeating a position on the current search path returns `0.0` (or slight contempt penalty if our root static eval indicates we are ahead, preventing endless 2-move shuffling in winning endgames). Path-dependent repetition scores are never written as exact `TT` bounds.
5. **Strict Time & Memory Safety (`C-07`, `C-08`, `C-11`)**:
   - Soft limit is `2.0s` (validator warns at `1.80s`). `TimeManager` enforces a hard internal deadline of **`1.68s`** (checking `time.perf_counter()` every `256` nodes) and aborts mid-depth cleanly via `SearchTimeout`, returning the best move from the last completed depth (or a proven better root move already finished at the current depth).
   - `TranspositionTable` is capped at `262_144` (`2^18`) entries (`< 40 MB` RSS total including Python + NumPy).

---

## 5. Pre-Merge Checklist (Every PR / Branch)

Before merging changes into `agents/student_agent.py`:
1. `python validate_submission.py agents/student_agent.py` prints `RESULT: PASS` with **0 warnings** (Python 3.11, NumPy 1.26).
2. `python simulator.py --player_1 greedy_agent --player_2 student_agent --autoplay --autoplay_runs 20 --strict` completes with **0 strikes**, **0 forfeits**, and `max step < 1.80 s`.
3. No starter files are modified (`git diff --name-only` shows only `agents/student_agent.py`, `ARCHITECTURE.md`, or `authors.yaml`).
