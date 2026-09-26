/**
 * Pins the browser move vocabulary to the server's frozen contract.
 *
 * `vocab.ts` computes indices in closed form rather than holding a table, so
 * a drift in the arithmetic would silently remap policy vectors -- every
 * likelihood we post would index the wrong move, and nothing would throw.
 * Enumerating all 4352 entries and digesting them catches exactly that: the
 * digest is compared against the frozen VOCAB_SHA256, which the Python side
 * self-checks at import. If these two ever disagree, one side moved.
 */
import { describe, expect, it } from "vitest";

import {
  MAIA_VOCAB_SIZE,
  VOCAB_SHA256,
  enumerateVocab,
  mirrorMoveUci,
  uciToVocabIdx,
  vocabChecksum,
  vocabIdxToUci,
} from "../vocab";

describe("vocabulary contract", () => {
  it("digests to the frozen VOCAB_SHA256", async () => {
    await expect(vocabChecksum()).resolves.toBe(VOCAB_SHA256);
  });

  it("enumerates exactly 4352 unique entries", () => {
    const vocab = enumerateVocab();
    expect(vocab).toHaveLength(MAIA_VOCAB_SIZE);
    expect(new Set(vocab).size).toBe(MAIA_VOCAB_SIZE);
  });

  it("round-trips every index through uci and back", () => {
    for (let i = 0; i < MAIA_VOCAB_SIZE; i++) {
      expect(uciToVocabIdx(vocabIdxToUci(i))).toBe(i);
    }
  });

  it("orders the base block from-major, to-minor (a1a1=0, a1b1=1, a1a2=8)", () => {
    // from*64 + to, with squares in python-chess order (a1=0, b1=1, a2=8).
    expect(uciToVocabIdx("a1a1")).toBe(0);
    expect(uciToVocabIdx("a1b1")).toBe(1);
    expect(uciToVocabIdx("a1a2")).toBe(8);
    expect(uciToVocabIdx("b1a1")).toBe(64);
    expect(uciToVocabIdx("h8h8")).toBe(4095);
  });

  it("orders promotions fileFrom x fileTo x piece(q,r,b,n) after the base block", () => {
    expect(uciToVocabIdx("a7a8q")).toBe(4096);
    expect(uciToVocabIdx("a7a8r")).toBe(4097);
    expect(uciToVocabIdx("a7a8b")).toBe(4098);
    expect(uciToVocabIdx("a7a8n")).toBe(4099);
    expect(uciToVocabIdx("a7b8q")).toBe(4100);
    expect(uciToVocabIdx("b7a8q")).toBe(4128);
    expect(uciToVocabIdx("h7h8n")).toBe(4351);
  });

  it("mirrors by rank flip: no castling or promotion special-casing", () => {
    expect(mirrorMoveUci("e1g1")).toBe("e8g8"); // castling is an ordinary king move
    expect(mirrorMoveUci("a7a8q")).toBe("a2a1q"); // promotion piece carried through
    expect(mirrorMoveUci("a1a1")).toBe("a8a8"); // same-square entries mirror positionally
    expect(mirrorMoveUci("e2e4")).toBe("e7e5");
  });

  it("returns -1 for moves outside the vocabulary rather than a wrong index", () => {
    expect(uciToVocabIdx("a2a1q")).toBe(-1); // promotion not written 7->8
    expect(uciToVocabIdx("a7a8k")).toBe(-1); // king is not a promotion piece
    expect(uciToVocabIdx("z9z9")).toBe(-1);
    expect(uciToVocabIdx("e2")).toBe(-1);
  });
});
