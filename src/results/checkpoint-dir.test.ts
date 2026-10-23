import { describe, expect, it } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { runSync } from "effect/Effect";

import { ScoreValue } from "../harness/core";
import { openCheckpointDir } from "./checkpoint-dir";

const RUN = { benchmark: "gpqa_diamond", model: "m", epochs: 2 };

function outcome(sampleId: string, epoch: number) {
  return {
    sampleScore: {
      sampleId,
      epoch,
      score: { value: ScoreValue.Correct, answer: "B", explanation: "" },
      messages: [],
      input: "q",
      target: "B",
    },
    usage: { inputTokens: 1, outputTokens: 2, totalTokens: 3 },
  };
}

describe("openCheckpointDir", () => {
  it("returns saved outcomes when the same run reopens the dir", () => {
    const dir = mkdtempSync(join(tmpdir(), "ckpt-"));
    const first = openCheckpointDir(dir, RUN);
    expect(first.done).toEqual([]);
    runSync(first.save(outcome("a/b", 0)));
    runSync(first.save(outcome("a/b", 1)));
    const again = openCheckpointDir(dir, RUN);
    expect(
      again.done
        .map((o) => `${o.sampleScore.sampleId}#${o.sampleScore.epoch}`)
        .sort()
    ).toEqual(["a/b#0", "a/b#1"]);
  });

  it("refuses a dir that holds a different run", () => {
    const dir = mkdtempSync(join(tmpdir(), "ckpt-"));
    openCheckpointDir(dir, RUN);
    expect(() => openCheckpointDir(dir, { ...RUN, epochs: 3 })).toThrow(
      "holds a different run"
    );
  });

  it("refuses a saved file that is not an outcome", () => {
    const dir = mkdtempSync(join(tmpdir(), "ckpt-"));
    openCheckpointDir(dir, RUN);
    writeFileSync(join(dir, "junk.e0.json"), JSON.stringify({ nope: 1 }));
    expect(() => openCheckpointDir(dir, RUN)).toThrow(
      "is not a saved sample outcome"
    );
  });
});
