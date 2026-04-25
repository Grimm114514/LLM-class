import json
import os
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA2 = ROOT / "data" / "2.txt"
DATA0 = ROOT / "data" / "0.json"
OUT = ROOT / "traincorpus.txt"
TMP_DIR = ROOT / ".tmp_corpus_build"

TARGET_BYTES = 300 * 1024 * 1024
BUCKETS = 256
SPLIT_RE = re.compile(r".+?[。！？；.!?;…]+|.+$", re.S)
WS_RE = re.compile(r"\s+")


def split_sentences(text: str):
    for part in SPLIT_RE.findall(text):
        line = WS_RE.sub(" ", part).strip()
        if len(line) >= 2:
            yield line


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

    if TMP_DIR.exists():
        for p in TMP_DIR.glob("bucket_*.txt"):
            p.unlink(missing_ok=True)
    else:
        TMP_DIR.mkdir(parents=True, exist_ok=True)

    bucket_paths = [TMP_DIR / f"bucket_{i:03d}.txt" for i in range(BUCKETS)]
    bucket_files = [p.open("w", encoding="utf-8", newline="\n") for p in bucket_paths]

    pool_bytes = 0
    from_data2_bytes = 0

    def add_line(line: str):
        nonlocal pool_bytes
        idx = random.randrange(BUCKETS)
        bucket_files[idx].write(line)
        bucket_files[idx].write("\n")
        pool_bytes += len(line.encode("utf-8")) + 1

    # Keep all from data/2.txt.
    with DATA2.open("r", encoding="utf-8", errors="ignore") as f2:
        txt2 = f2.read()
    for s in split_sentences(txt2):
        add_line(s)
        from_data2_bytes += len(s.encode("utf-8")) + 1

    # Randomly sample completion from data/0.json.
    p = 0.70
    passes = 0
    while pool_bytes < int(TARGET_BYTES * 1.05) and passes < 3:
        passes += 1
        for obj in iter_json_array_objects(DATA0):
            comp = obj.get("completion")
            if not isinstance(comp, str):
                continue
            if random.random() > p:
                continue
            for s in split_sentences(comp):
                add_line(s)
        p = min(0.95, p + 0.15)

    for bf in bucket_files:
        bf.close()

    # Shuffle and write final output with one sentence per line.
    if OUT.exists():
        OUT.unlink()

    written = 0
    lines_written = 0
    order = list(range(BUCKETS))
    random.shuffle(order)

    with OUT.open("w", encoding="utf-8", newline="\n") as out:
        for idx in order:
            bpath = bucket_paths[idx]
            if not bpath.exists():
                continue
            with bpath.open("r", encoding="utf-8", errors="ignore") as bf:
                lines = [ln.rstrip("\n") for ln in bf if ln.strip()]
            random.shuffle(lines)

            for line in lines:
                b = len(line.encode("utf-8")) + 1
                if written + b > TARGET_BYTES:
                    continue
                out.write(line)
                out.write("\n")
                written += b
                lines_written += 1
                if written >= TARGET_BYTES:
                    break
            if written >= TARGET_BYTES:
                break

    # Cleanup temp bucket files.
    for pth in bucket_paths:
        try:
            pth.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        TMP_DIR.rmdir()
    except OSError:
        pass

    print(f"target_bytes={TARGET_BYTES}")
    print(f"from_data2_bytes={from_data2_bytes}")
    print(f"pool_bytes={pool_bytes}")
    print(f"written_bytes={written}")
    print(f"lines_written={lines_written}")
    print(f"output={OUT}")


if __name__ == "__main__":
    main()
