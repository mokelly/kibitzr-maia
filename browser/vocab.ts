/**
 * Chess move vocabulary and mirror-normalized move indexing.
 *
 * The 4352-entry canonical enumeration: 4096 from-to moves ordered by source
 * square index (a1=0 .. h8=63) then target square index, followed by 256
 * promotions ordered fileFrom(a-h) x fileTo(a-h) x piece(q,r,b,n), always
 * written rank 7 -> 8 because the board is mirrored when Black is to move.
 *
 * Rather than materialize a 4352-entry table, the mapping is closed-form
 * index arithmetic. `VOCAB_SHA256` pins the resulting enumeration to the
 * server's frozen constant, and `__tests__/vocab.checksum.test.ts` enumerates
 * every index through `vocabIdxToUci` and verifies the digest matches -- so
 * an arithmetic drift that a spot-check would miss fails loudly instead.
 */

export const MAIA_VOCAB_SIZE = 4352;

/**
 * sha256 of ",".join(vocabulary) in canonical order. Frozen contract shared
 * with the Python-side vocabulary; the two MUST stay identical, because this
 * ordering is the index space of every policy vector crossing the wire.
 */
export const VOCAB_SHA256 =
  "07af65839e5324b01a255e2663d03a6dc785f173b2f37ee824535b369906ef3f";
const PROMO_BASE = 4096;
const PROMO_PIECES = "qrbn";

/** a1=0 .. h8=63 (python-chess square numbering); -1 if malformed. */
function squareIndex(fileChar: number, rankChar: number): number {
  const file = fileChar - 97; // 'a'
  const rank = rankChar - 49; // '1'
  if (file < 0 || file > 7 || rank < 0 || rank > 7) return -1;
  return rank * 8 + file;
}

/** Mirror a UCI move vertically (rank r -> 9-r, files unchanged). */
export function mirrorMoveUci(uci: string): string {
  const flip = (sq: string) => sq[0] + String(9 - Number(sq[1]));
  return flip(uci.slice(0, 2)) + flip(uci.slice(2, 4)) + uci.slice(4);
}

/**
 * Vocabulary index of a MODEL-space UCI move (i.e. already mirrored when the
 * mover is Black), or -1 when the move is not in the vocabulary (promotions
 * not written 7->8, malformed input) -- matching the Python dict-miss path.
 */
export function uciToVocabIdx(modelUci: string): number {
  if (modelUci.length === 5) {
    const piece = PROMO_PIECES.indexOf(modelUci[4]);
    if (piece < 0 || modelUci[1] !== "7" || modelUci[3] !== "8") return -1;
    const fileFrom = modelUci.charCodeAt(0) - 97;
    const fileTo = modelUci.charCodeAt(2) - 97;
    if (fileFrom < 0 || fileFrom > 7 || fileTo < 0 || fileTo > 7) return -1;
    return PROMO_BASE + fileFrom * 32 + fileTo * 4 + piece;
  }
  if (modelUci.length !== 4) return -1;
  const from = squareIndex(modelUci.charCodeAt(0), modelUci.charCodeAt(1));
  const to = squareIndex(modelUci.charCodeAt(2), modelUci.charCodeAt(3));
  if (from < 0 || to < 0) return -1;
  return from * 64 + to;
}

/** Inverse of uciToVocabIdx (model space). */
export function vocabIdxToUci(idx: number): string {
  const sqName = (sq: number) =>
    String.fromCharCode(97 + (sq & 7)) + String(1 + (sq >> 3));
  if (idx >= PROMO_BASE) {
    const p = idx - PROMO_BASE;
    const fileFrom = p >> 5;
    const fileTo = (p >> 2) & 7;
    return (
      String.fromCharCode(97 + fileFrom) + "7" +
      String.fromCharCode(97 + fileTo) + "8" +
      PROMO_PIECES[(p & 3)]
    );
  }
  return sqName(idx >> 6) + sqName(idx & 63);
}

/**
 * Vocabulary index of a REAL-board UCI move: mirrored into model space first
 * when Black is to move (the model always sees the mover as White).
 */
export function moveToVocabIdx(uci: string, whiteToMove: boolean): number {
  return uciToVocabIdx(whiteToMove ? uci : mirrorMoveUci(uci));
}

/** Real-board UCI for a model-space vocabulary index. */
export function vocabIdxToMove(idx: number, whiteToMove: boolean): string {
  const uci = vocabIdxToUci(idx);
  return whiteToMove ? uci : mirrorMoveUci(uci);
}

/**
 * Materialize the full canonical vocabulary, in index order.
 *
 * Only used to verify the closed-form arithmetic against `VOCAB_SHA256` --
 * the hot paths index directly and never allocate this list.
 */
export function enumerateVocab(): string[] {
  const out = new Array<string>(MAIA_VOCAB_SIZE);
  for (let i = 0; i < MAIA_VOCAB_SIZE; i++) out[i] = vocabIdxToUci(i);
  return out;
}

/** sha256 of the comma-joined vocabulary; compare against VOCAB_SHA256. */
export async function vocabChecksum(): Promise<string> {
  const bytes = new TextEncoder().encode(enumerateVocab().join(","));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Sorted vocabulary indices for a list of REAL-board legal UCI moves,
 * silently skipping vocabulary misses (matches the wrapper's mask builder).
 */
export function legalMoveVocabIndices(
  legalUcis: readonly string[],
  whiteToMove: boolean,
): number[] {
  const out: number[] = [];
  for (const u of legalUcis) {
    const idx = moveToVocabIdx(u, whiteToMove);
    if (idx >= 0) out.push(idx);
  }
  out.sort((a, b) => a - b);
  return out;
}
