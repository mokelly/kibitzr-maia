/**
 * Maia-3 input tokenization: FEN history -> (64, width) float32 tensor.
 *
 * Exact port of the Python chain the production wrappers use:
 *   maia3.dataset.tokenize_board        (64x12 one-hot, mirror when Black moves)
 *   maia3.dataset.get_historical_tokens (8-ply concat, oldest-kept front-pad,
 *                                        /100 clock columns)
 * via build_timed_history_tensor (23M-ponder, 100 cols) and
 * build_history_tensor (untimed 79M, 97 cols with a zero ponder column).
 *
 * Only the FEN placement + side-to-move fields are consumed. The output is
 * intended to be bit-exact with the Python chain.
 */

export const MAIA_HISTORY = 8;
export const PLANES_PER_BOARD = 12;
const HISTORY_COLS = MAIA_HISTORY * PLANES_PER_BOARD; // 96
export const TIMED_TOKEN_WIDTH = HISTORY_COLS + 4; // 100 (base, inc, clk, ponder)
export const UNTIMED_TOKEN_WIDTH = HISTORY_COLS + 1; // 97 (zero ponder column)

// piece -> 1..6 (P,N,B,R,Q,K), +6 when the piece belongs to the side NOT to
// move after mirroring (Python: black pieces on the White-to-move board).
const PIECE_CODE: Record<string, number> = {
  P: 1, N: 2, B: 3, R: 4, Q: 5, K: 6,
  p: 7, n: 8, b: 9, r: 10, q: 11, k: 12,
};

interface ParsedBoard {
  /** Piece code 1-12 per square (a1=0..h8=63), 0 = empty. */
  codes: Uint8Array;
  whiteToMove: boolean;
}

function parseFen(fen: string): ParsedBoard {
  const sp = fen.indexOf(" ");
  const placement = sp < 0 ? fen : fen.slice(0, sp);
  const turnField = sp < 0 ? "w" : fen.slice(sp + 1, sp + 2);
  const codes = new Uint8Array(64);
  let rank = 7;
  let file = 0;
  for (const ch of placement) {
    if (ch === "/") {
      rank -= 1;
      file = 0;
    } else if (ch >= "1" && ch <= "8") {
      file += ch.charCodeAt(0) - 48;
    } else {
      const code = PIECE_CODE[ch];
      if (code === undefined || rank < 0 || file > 7) {
        throw new Error(`maia tokenizer: bad FEN placement "${placement}"`);
      }
      codes[rank * 8 + file] = code;
      file += 1;
    }
  }
  return { codes, whiteToMove: turnField !== "b" };
}

/**
 * Write one board's 12 one-hot planes into `out` at column offset `colBase`
 * of a row-major (64, rowWidth) tensor. When Black is to move the board is
 * mirrored (vertical flip = sq^56, colors swapped), so the model always sees
 * the side to move as White -- identical to `tokenize_board`.
 */
function writeBoardPlanes(
  board: ParsedBoard,
  out: Float32Array,
  rowWidth: number,
  colBase: number,
): void {
  const { codes, whiteToMove } = board;
  for (let sq = 0; sq < 64; sq++) {
    const src = whiteToMove ? sq : sq ^ 56;
    const c = codes[src];
    if (c === 0) continue;
    const token = whiteToMove ? c : c > 6 ? c - 6 : c + 6;
    out[sq * rowWidth + colBase + token - 1] = 1;
  }
}

export interface ClockContext {
  /** Starting time in seconds (e.g. 180 for 3+0). */
  baseS: number;
  /** Increment in seconds. */
  incS: number;
  /** Mover's clock before starting to think, in seconds. */
  clkLeftBeforeS: number;
}

/**
 * Guard against the silent-degradation trap.
 *
 * Front-padding by repeating the oldest position is CORRECT at the start of a
 * game -- at ply 0 there is genuinely nothing older. But a caller that passes
 * only the CURRENT position mid-game gets eight copies of it and a perfectly
 * well-formed, entirely wrong tensor: no throw, no NaN, just quietly worse
 * predictions that no test would notice. Since 8-ply history is what separates
 * Maia-3 from a position-only model, that failure would be invisible and
 * expensive.
 *
 * So callers that know their ply index declare it, and a short history is loud
 * instead of silent.
 */
function assertHistoryDepth(fens: readonly string[], plyIndex: number): void {
  const expected = Math.min(MAIA_HISTORY, plyIndex + 1);
  if (fens.length < expected) {
    throw new Error(
      `maia tokenizer: ply ${plyIndex} needs ${expected} positions of history ` +
        `(got ${fens.length}). Front-padding is only valid near the game start; ` +
        `passing a short history mid-game silently degrades the prediction.`,
    );
  }
}

function buildTokens(
  fens: readonly string[],
  width: number,
  clock: ClockContext | null,
  plyIndex?: number,
): Float32Array {
  if (fens.length === 0) {
    throw new Error("maia tokenizer: empty FEN history");
  }
  if (plyIndex !== undefined) assertHistoryDepth(fens, plyIndex);
  // Most recent MAIA_HISTORY positions, oldest -> newest, the LAST one being
  // the position under evaluation; short runs front-pad with the oldest kept.
  const window = fens.slice(-MAIA_HISTORY).map(parseFen);
  const pad = MAIA_HISTORY - window.length;
  const out = new Float32Array(64 * width);
  for (let slot = 0; slot < MAIA_HISTORY; slot++) {
    const board = window[Math.max(0, slot - pad)];
    writeBoardPlanes(board, out, width, slot * PLANES_PER_BOARD);
  }
  if (clock) {
    // Float32Array assignment rounds exactly like torch.full's f64->f32 cast.
    const base = clock.baseS / 100;
    const inc = clock.incS / 100;
    const clk = clock.clkLeftBeforeS / 100;
    for (let sq = 0; sq < 64; sq++) {
      const r = sq * width + HISTORY_COLS;
      out[r] = base;
      out[r + 1] = inc;
      out[r + 2] = clk;
      // out[r + 3] stays 0: clk_ponder is a training-time label column.
    }
  }
  // Untimed: single trailing ponder column, always 0 -- nothing to write.
  return out;
}

/**
 * (64, 100) input row for the 23M-ponder (time-aware) model.
 *
 * `plyIndex` is optional but STRONGLY recommended for any production caller:
 * it turns a too-short history into a loud error instead of silently
 * front-padded garbage (see assertHistoryDepth).
 */
export function buildTimedHistoryTokens(
  fens: readonly string[],
  clock: ClockContext,
  plyIndex?: number,
): Float32Array {
  return buildTokens(fens, TIMED_TOKEN_WIDTH, clock, plyIndex);
}

/** (64, 97) input row for the untimed models (79M). See plyIndex note above. */
export function buildHistoryTokens(
  fens: readonly string[],
  plyIndex?: number,
): Float32Array {
  return buildTokens(fens, UNTIMED_TOKEN_WIDTH, null, plyIndex);
}

/** Side to move of a FEN -- the mirror flag every consumer needs. */
export function fenWhiteToMove(fen: string): boolean {
  return parseFen(fen).whiteToMove;
}
