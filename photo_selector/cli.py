from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .core import (Engine, REVISION, atomic_json, connect, digest, download_models,
                   files_under, identity_score, normalized, phash, prepare_state, read_image)


def enroll(args, state):
    source = args.references.expanduser().resolve(strict=True)
    paths = list(files_under(source))
    if len(paths) < 2:
        raise ValueError(f"Expected at least two distinct reference images, found {len(paths)}")
    engine = Engine(getattr(args, "engine_state", state), args.max_side, args.threads)
    records, failures = [], []
    for path in paths:
        try:
            faces = engine.faces(read_image(path))
            if len(faces) != 1:
                raise ValueError(f"expected exactly one detected face, found {len(faces)}; use a solo photo or a cropped reference")
            face = faces[0]
            if face["face_pixels"] < 70:
                raise ValueError("reference face is smaller than 70 pixels")
            records.append({"path": str(path), "sha256": digest(path), "embedding": face["embedding"].tolist()})
            print(f"Reference OK: {path.name}")
        except Exception as error:
            failures.append(f"{path.name}: {error}")
    if failures:
        if not getattr(args, "skip_invalid", False):
            raise ValueError("Enrollment not saved:\n" + "\n".join(failures))
        for failure in failures:
            print("Reference excluded: " + failure, flush=True)
    if len(records) < 2:
        raise ValueError("At least two valid references are required")
    if len({record["sha256"] for record in records}) != len(records):
        raise ValueError("References include exact duplicates; use different photos")
    vectors = np.array([record["embedding"] for record in records])
    similarity = vectors @ vectors.T
    # A sanity check, not proof of identity: user-selected references are authoritative.
    np.fill_diagonal(similarity, -1)
    suspect = [Path(records[i]["path"]).name for i in range(len(records)) if similarity[i].max() < 0.30]
    if suspect:
        raise ValueError("These references have no convincing match to another reference: " + ", ".join(suspect))
    atomic_json(state / "identity.json", {"source": str(source), "model_revision": REVISION,
                "references": records, "excluded": failures, "input_fingerprint": getattr(args, "input_fingerprint", None),
                "centroid": normalized(vectors.mean(axis=0)).tolist()})
    print(f"Enrolled {len(records)} references. Pairwise similarity range: {similarity[similarity > -1].min():.3f}–{similarity.max():.3f}")


def scan(args, state):
    source = args.library.expanduser().resolve(strict=True)
    identity_path = state / "identity.json"
    if not identity_path.exists():
        raise ValueError("Run 'enroll' before scanning")
    identity = json.loads(identity_path.read_text())
    if identity["model_revision"] != REVISION:
        raise ValueError("Model version changed; enroll references again")
    library_file = state / "library.json"
    if library_file.exists() and json.loads(library_file.read_text())["source"] != str(source):
        raise ValueError("This state belongs to another library; use a separate --state folder")
    references = np.array([r["embedding"] for r in identity["references"]], dtype=np.float32)
    engine = Engine(getattr(args, "engine_state", state), args.max_side, args.threads)
    config = hashlib.sha256(json.dumps({"identity": digest(identity_path), "max_side": args.max_side,
                                      "pipeline": 3, "revision": REVISION}, sort_keys=True).encode()).hexdigest()
    # Finish enumeration before modifying the inventory: inaccessible directories abort safely.
    paths = list(files_under(source))
    atomic_json(library_file, {"source": str(source), "config": config, "supported_files": len(paths), "identity_digest": digest(identity_path)})
    db = connect(state)
    current = {str(p) for p in paths}
    with db:
        for row in db.execute("SELECT path FROM photos").fetchall():
            if row["path"] not in current:
                db.execute("DELETE FROM photos WHERE path=?", (row["path"],))
    processed = skipped = errors = 0
    start = time.monotonic()
    try:
        for path in paths:
            if args.limit and processed >= args.limit:
                break
            stat = path.stat()
            prior = db.execute("SELECT * FROM photos WHERE path=?", (str(path),)).fetchone()
            if prior and not prior["error"] and prior["size"] == stat.st_size and prior["mtime_ns"] == stat.st_mtime_ns and prior["config"] == config:
                skipped += 1
                continue
            data, error = {}, None
            try:
                image = read_image(path)
                faces = engine.faces(image)
                data = {"width": image.width, "height": image.height, "face_count": len(faces),
                        "sha256": digest(path), "phash": phash(image), "similarity": -1}
                if faces:
                    scores = [identity_score(face["embedding"], references) for face in faces]
                    best = faces[int(np.argmax(scores))]
                    data.update({key: value for key, value in best.items() if key != "embedding"})
                    data["similarity"] = max(scores)
                after = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("Source changed during processing; retry scan")
            except Exception as exc:
                error = str(exc)
                errors += 1
                print(f"Skipped unreadable image: {path}: {error}", file=sys.stderr)
            with db:
                db.execute("INSERT OR REPLACE INTO photos VALUES (?,?,?,?,?,?)",
                           (str(path), stat.st_size, stat.st_mtime_ns, config, json.dumps(data), error))
            processed += 1
            if processed % 25 == 0:
                print(f"Processed {processed}, cached {skipped}, errors {errors}; {time.monotonic()-start:.1f}s", flush=True)
    finally:
        db.close()
    print(f"Scan finished: {processed} processed, {skipped} cached, {errors} errors; {len(paths)} supported files in library.")
    print("Identity scan saved. The recommendation pipeline will assess the matching photos next.")



def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Recommend your top dating-profile photos using configurable visual criteria.")
    parser.add_argument("--state", type=Path, default=Path(os.environ.get("PHOTO_SELECTOR_STATE", str(Path.home() / ".local/share/photo-selector-gpu"))))
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "rank"):
        command = subs.add_parser(name, help="Scan and recommend" if name == "run" else "Reassess/rerank a completed scan")
        command.add_argument("--config", type=Path, default=Path("selector.toml"))
        if name == "run":
            command.add_argument("--scan-only", action="store_true", help="Finish the identity stage only")
    subs.add_parser("models", help="Download verified face and visual assessment models")
    doctor = subs.add_parser("doctor", help="Verify face inference executes on the AMD GPU")
    doctor.add_argument("--max-side", type=positive, default=1280)
    doctor.add_argument("--threads", type=positive, default=4)
    for name in ("enroll", "scan"):
        sub = subs.add_parser(name, help="Low-level identity diagnostic")
        sub.add_argument("references" if name == "enroll" else "library", type=Path)
        sub.add_argument("--max-side", type=positive, default=1280)
        sub.add_argument("--threads", type=positive, default=4)
        if name == "scan":
            sub.add_argument("--limit", type=positive)
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        sources = [getattr(args, key) for key in ("references", "library") if hasattr(args, key)]
        state = prepare_state(args.state, sources)
        if args.command in ("run", "rank"):
            from .recommend import run
            run(args, state)
        elif args.command == "models":
            download_models(state)
            from .vision import download_vision
            download_vision(state)
        elif args.command == "doctor":
            report = Engine(state, args.max_side, args.threads, profile=True).diagnose()
            atomic_json(state / "gpu-diagnostic.json", report)
            print(json.dumps(report, indent=2))
        else:
            {"enroll": enroll, "scan": scan}[args.command](args, state)
    except KeyboardInterrupt:
        print("Interrupted; completed work is cached. Run the same command to resume.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
