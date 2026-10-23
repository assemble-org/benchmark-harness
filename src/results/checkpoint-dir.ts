import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";

import type { Effect } from "effect/Effect";
import { sync } from "effect/Effect";

import type { EvalOutcome } from "../harness/run";

const RUN_FILE = "run.json";

export function openCheckpointDir(
  dir: string,
  run: Readonly<Record<string, unknown>>
): {
  readonly done: readonly EvalOutcome[];
  readonly save: (outcome: EvalOutcome) => Effect<void>;
} {
  mkdirSync(dir, { recursive: true });
  const runPath = join(dir, RUN_FILE);
  const wanted = JSON.stringify(run);
  if (existsSync(runPath)) {
    const found = JSON.stringify(JSON.parse(readFileSync(runPath, "utf8")));
    if (found !== wanted) {
      throw new Error(
        `checkpoint dir ${dir} holds a different run: ${found} (this run: ${wanted})`
      );
    }
  } else {
    writeFileSync(runPath, wanted);
  }
  const done = readdirSync(dir)
    .filter((name) => name.endsWith(".json") && name !== RUN_FILE)
    .map((name) => parseOutcome(join(dir, name)));
  const save = (outcome: EvalOutcome): Effect<void> =>
    sync(() => {
      const { sampleId, epoch } = outcome.sampleScore;
      const path = join(dir, `${encodeURIComponent(sampleId)}.e${epoch}.json`);
      writeFileSync(`${path}.tmp`, JSON.stringify(outcome));
      renameSync(`${path}.tmp`, path);
    });
  return { done, save };
}

function parseOutcome(path: string): EvalOutcome {
  const value: unknown = JSON.parse(readFileSync(path, "utf8"));
  const score = (value as { sampleScore?: unknown } | null)?.sampleScore as
    | { sampleId?: unknown; epoch?: unknown; score?: { value?: unknown } }
    | undefined;
  if (
    typeof score?.sampleId !== "string" ||
    typeof score.epoch !== "number" ||
    typeof score.score?.value !== "string"
  ) {
    throw new TypeError(`${path} is not a saved sample outcome`);
  }
  return value as EvalOutcome;
}
