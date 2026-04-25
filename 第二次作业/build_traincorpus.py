import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA2 = ROOT / "data" / "2.txt"
DATA0 = ROOT / "data" / "0.json"
OUT = ROOT / "traincorpus.txt"
TMP_DIR = ROOT / ".tmp_corpus_build"
MIX_DIR = ROOT / ".tmp_corpus_mix"

TARGET_BYTES = 300 * 1024 * 1024
BUCKETS = 256
INLINE_WS_RE = re.compile(r"[\r\n\t]+")


def to_single_line(text: str) -> str:
    # Keep record boundaries stable: one input record maps to one output line.
    line = INLINE_WS_RE.sub(" ", text).strip()
    return line


def iter_json_array_objects(path: Path):
    with path.open("r", encoding="utf-8") as f:
        in_string = False
        escape = False
        depth = 0
        started = False
        obj_buf = []

        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break

            for ch in chunk:
                if not started:
                    if ch == "[":
                        started = True
                    continue

                if depth == 0:
                    if ch == "{":
                        depth = 1
                        in_string = False
                        escape = False
                        obj_buf = [ch]
                    elif ch == "]":
                        return
                    else:
                        continue
                    continue

                obj_buf.append(ch)

                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            obj_text = "".join(obj_buf)
                            yield json.loads(obj_text)
                            obj_buf = []


def main():
    random.seed()

    for d in [TMP_DIR, MIX_DIR]:
        if d.exists():
            for p in d.glob("bucket_*.txt"):
                p.unlink(missing_ok=True)
        else:
            d.mkdir(parents=True, exist_ok=True)

    bucket_paths = [TMP_DIR / f"bucket_{i:03d}.txt" for i in range(BUCKETS)]
    bucket_files = [p.open("w", encoding="utf-8", newline="\n") for p in bucket_paths]
    mix_bucket_paths = [MIX_DIR / f"bucket_{i:03d}.txt" for i in range(BUCKETS)]
    mix_bucket_files = [p.open("w", encoding="utf-8", newline="\n") for p in mix_bucket_paths]

    completion_pool_bytes = 0
    from_data2_bytes = 0
    data2_lines = 0
    sampled_completions = 0

    def add_completion_line(line: str):
        nonlocal completion_pool_bytes
        idx = random.randrange(BUCKETS)
        bucket_files[idx].write(line)
        bucket_files[idx].write("\n")
        completion_pool_bytes += len(line.encode("utf-8")) + 1

    def add_mix_line(line: str):
        idx = random.randrange(BUCKETS)
        mix_bucket_files[idx].write(line)
        mix_bucket_files[idx].write("\n")

    # Keep all original lines from data/2.txt.
    with DATA2.open("r", encoding="utf-8", errors="ignore") as f2:
        for raw in f2:
            line = raw.rstrip("\r\n")
            add_mix_line(line)
            from_data2_bytes += len(line.encode("utf-8")) + 1
            data2_lines += 1

    if from_data2_bytes > TARGET_BYTES:
        raise RuntimeError("data/2.txt alone is larger than 300MiB; cannot keep all lines within target size.")

    # Randomly sample completion from data/0.json (one completion -> one output line).
    p = 0.70
    passes = 0
    while completion_pool_bytes < int((TARGET_BYTES - from_data2_bytes) * 1.10) and passes < 3:
        passes += 1
        for obj in iter_json_array_objects(DATA0):
            comp = obj.get("completion")
            if not isinstance(comp, str):
                continue
            if random.random() > p:
                continue
            line = to_single_line(comp)
            add_completion_line(line)
            sampled_completions += 1
        p = min(0.95, p + 0.15)

    for bf in bucket_files:
        bf.close()

    # Select completion records up to remaining budget and merge into mix buckets.
    written = from_data2_bytes
    selected_completion_lines = 0
    order = list(range(BUCKETS))
    random.shuffle(order)

    for idx in order:
        bpath = bucket_paths[idx]
        if not bpath.exists():
            continue
        with bpath.open("r", encoding="utf-8", errors="ignore") as bf:
            lines = [ln.rstrip("\n") for ln in bf]
        random.shuffle(lines)

        for line in lines:
            b = len(line.encode("utf-8")) + 1
            if written + b > TARGET_BYTES:
                continue
            add_mix_line(line)
            written += b
            selected_completion_lines += 1
            if written >= TARGET_BYTES:
                break
        if written >= TARGET_BYTES:
            break

    for bf in mix_bucket_files:
        bf.close()

    # Final shuffle output from mix buckets.
    if OUT.exists():
        OUT.unlink()

    lines_written = 0
    final_order = list(range(BUCKETS))
    random.shuffle(final_order)

    with OUT.open("w", encoding="utf-8", newline="\n") as out:
        for idx in final_order:
            bpath = mix_bucket_paths[idx]
            if not bpath.exists():
                continue
            with bpath.open("r", encoding="utf-8", errors="ignore") as bf:
                lines = [ln.rstrip("\n") for ln in bf]
            random.shuffle(lines)

            for line in lines:
                out.write(line)
                out.write("\n")
                lines_written += 1

    # Cleanup temp bucket files.
    for paths, directory in [(bucket_paths, TMP_DIR), (mix_bucket_paths, MIX_DIR)]:
        for pth in paths:
            try:
                pth.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            directory.rmdir()
        except OSError:
            pass

    print(f"target_bytes={TARGET_BYTES}")
    print(f"from_data2_bytes={from_data2_bytes}")
    print(f"data2_lines={data2_lines}")
    print(f"completion_pool_bytes={completion_pool_bytes}")
    print(f"sampled_completions={sampled_completions}")
    print(f"selected_completion_lines={selected_completion_lines}")
    print(f"written_bytes={written}")
    print(f"lines_written={lines_written}")
    print(f"output={OUT}")


if __name__ == "__main__":
    main()
