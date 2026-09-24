# MatchShot

Automatically scan a photo library and recommend the **top 10 photos for a dating
profile**, using your own criteria. All photo analysis runs locally.

The pipeline identifies the person in your reference photos, assesses each matching
photo's expression, pose, composition, presentation, setting, and technical quality,
and selects distinct photos using configurable weights and a variety preference.
The output includes the photos, original paths, scores, and reasons. No manual
labelling or review workflow is required before it can recommend a set.

## Requirements

This release targets **Linux x86_64 with an AMD GPU supported by ROCm 7.2.1 and
Vulkan**. The current configuration selects GPU 0 and assumes a single AMD GPU.
CPU-only, NVIDIA/CUDA, Apple Silicon, and multi-GPU selection are not supported by
this setup. Check [AMD's ROCm compatibility documentation](https://rocm.docs.amd.com/en/docs-7.2.1/compatibility/compatibility-matrix.html)
for host OS and GPU support; running in a container does not replace the host driver.

Install on the host:

- [Docker Engine](https://docs.docker.com/engine/install/), usable by your account
  without running VS Code as root.
- [VS Code](https://code.visualstudio.com/) and its
  [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
- Python **3.11 or newer**, used only to generate private container configuration.
- A supported AMD kernel driver exposing `/dev/kfd` and an AMD `/dev/dri/renderD*`
  device. Host ROCm user-space packages are not required; the image supplies them.

Allow several GB for model downloads, additional space for caches and reports, and
substantial Docker image/build storage (budget tens of GB). GPU memory requirements
vary with image resolution; this release does not establish a minimum VRAM size
across GPUs. `photo-selector doctor` verifies face inference on your actual device.

## First-time setup

Clone this repository and run the configuration generator **on the host**:

```bash
git clone https://github.com/gwyacnt/photo_selector.git
cd photo_selector
python3 .devcontainer/configure.py
```

It detects an AMD render device and its numeric access groups, and creates three
**git-ignored, private files**:

| File | Purpose |
| --- | --- |
| `.env` | Host input/state paths, user IDs, and GPU device/group settings |
| `selector.toml` | Your paths, criteria, weights, and selection rules; copied from `selector.example.toml` |
| `.devcontainer/devcontainer.json` | Generated read-only input mounts and container lifecycle settings |

The initial input folders are empty placeholders under `.local-inputs/`. Edit
`.env` to point to real folders; use absolute paths and quote spaces:

```dotenv
PHOTO_LIBRARY='/absolute/path/to/photo-library'
PHOTO_REFERENCES='/absolute/path/to/reference-photos'
PHOTO_STATE='/absolute/path/to/private-output'
```

Keep `PHOTO_STATE` outside both input folders. Leave the generated device/group
values in place unless you need to correct device selection. Prepare at least two
clear, distinct reference photos, each showing only the intended person's face.
A group photo can be cropped into a separate reference file without altering its
original. Several reference photos with varied angles and lighting work better
than repeated copies of one shot.

Regenerate after editing paths:

```bash
python3 .devcontainer/configure.py
code .
```

In VS Code, run **Dev Containers: Reopen in Container**. The Dockerfile installs all
system and Python runtime dependencies. The container's post-create script installs
the application and automatically downloads/verifies its model weights and visual
runtime into the persistent `/state` mount. Setup does **not** scan your photos.
Internet access is needed for this first build/download; no API key or account is
needed for the public model downloads.

If model setup is interrupted, rerun inside the container:

```bash
photo-selector models
```

Completed downloads are reused and checked. Partial downloads restart. Downloads
are not performed implicitly while enrolling, scanning, or ranking.

## Run and rerun

Inside the container terminal:

```bash
# Verify the face models actually execute on the GPU.
photo-selector doctor

# Scan all supported photos, assess every match, and produce recommendations.
photo-selector run --config selector.toml
```

The command runs three stages in order. A large library can take hours because
finding your face and judging the matching photos are separate jobs.

1. **Find photos of you.** First the app reads your reference photos, or reuses the
   saved references if they have not changed. It then visits every supported photo
   in every library subfolder, detects faces, and compares them with your references.
   It saves the best match score, face location, and information needed to recognise
   duplicate photos. At this stage it is **not yet judging dating-profile quality**.
   Photos with no face or no matching face are normal results, not errors.
2. **Assess the matching photos.** After the entire face scan finishes, the local
   vision model examines each photo whose match score meets your identity threshold.
   It sees the whole photo and a crop identifying the matched face. It applies your
   exclusions and gives each configured criterion a score and explanation: for
   example, expression, pose, composition, and setting. There is no preliminary
   top-100 quality filter: every readable identity match reaches this stage, with
   cached assessments reused when available.
3. **Choose and explain the top photos.** The app combines the criterion scores
   using your weights, applies the minimum score and optional date cutoff, removes
   duplicate candidates, and applies your variety preference. It then writes the
   requested top photos (10 by default) and an HTML report explaining the choices.
   If fewer qualify, it reports the shortfall.

### Reading the terminal output

During **stage 1**, a progress line looks like this (illustrative numbers):

```text
Processed 1250, cached 500, errors 3; 120.0s
```

| Counter | Meaning |
| --- | --- |
| `Processed 1250` | 1,250 files were attempted in this invocation, including the 3 failures. These are not 1,250 photos of you or 1,250 quality assessments. |
| `cached 500` | Valid scan results for 500 unchanged files were reused from earlier work; those files did not need face detection again. |
| `errors 3` | Processing failed for 3 of the attempted files. The file path and error are printed separately. Not finding your face does **not** count as an error. |
| `120.0s` | Time spent in this invocation's scan loop; it excludes earlier runs and initial model loading/file enumeration. |

In that example, the scan has reached **1,750 files**: `processed + cached`.
Do not add `errors` again; they are already included in `processed`. Progress is
printed every 25 attempted files. These counters do not tell you how many photos
matched your face, and completion of this scan is not completion of the whole run.

When the face scan finishes, the terminal prints `Scan finished` and then moves
automatically to **stage 2**:

```text
Visual stage: 300 identity matches, 20 cached, 280 to assess
Assessed 1/280 new photos; 8s elapsed, about 37m remaining
```

These illustrative numbers mean 300 photos matched the references, 20 already have
usable visual assessments, and the remaining 280 need assessment. This stage's
elapsed time and estimated time remaining are separate from the face-scan timer.
The final `Recommended ... Open: .../index.html` message means the report is ready.

All stages run locally. Image decoding/preprocessing uses CPU; the face models run
on the AMD GPU through MIGraphX and the visual model uses Vulkan. The app starts
and stops its own local visual-model server; you do not need to manage that server.

Progress and the final `index.html` path are printed to the terminal. Open the report
in your host browser: replace its `/state` prefix with your configured `PHOTO_STATE`
path. Original-photo links work on the host because input paths are mounted at the
same absolute paths in the container.

Stop with **Ctrl-C**, then rerun the same command to resume. Completed scan records
and visual assessments are cached. Run only one pipeline per state folder. Keep
input files stable during a run; new or changed files require a refreshed scan.

After changing weights or criteria, reuse the completed scan:

```bash
photo-selector rank --config selector.toml
```

| Change | What happens |
| --- | --- |
| Weights, top count, minimum score, date cutoff, variety preference | Reranks cached assessments |
| Criterion descriptions, exclusions, visual input resolution | Reassesses matching photos, without rerunning face detection |
| Reference photos, recognition resolution, library contents | Use `run` to refresh enrollment/scan as needed |
| Library or reference-folder paths | Uses a separate cache for that pair of roots; use `run` |

If fewer than ten photos pass the configured rules, the report explains the
shortfall instead of filling the result with rejected photos. Scores are subjective
model assessments of the rubric, not an objective attractiveness measure or a
prediction of dating success.

## Configure paths and criteria

Edit private **`selector.toml`**. The public, documented defaults are in
[`selector.example.toml`](selector.example.toml). By default its paths use the
`.env` values passed into the container:

```toml
[paths]
library = "${PHOTO_LIBRARY}"
references = "${PHOTO_REFERENCES}"
```

You can instead use absolute paths or paths relative to the TOML file. The container
generator reads those overrides too. After changing mounted roots, run
`python3 .devcontainer/configure.py` **on the host**, then use **Dev Containers:
Rebuild Container**. A path beneath an already mounted root needs no new mount.
Generated container settings are checked before startup; stale settings produce a
message to rerun the generator. Do not commit the generated files or private config.

The `[recognition]` section controls identity threshold, detection resolution, CPU
preprocessing threads, and whether invalid references are skipped. With
`skip_invalid_references = true`, each excluded reference is printed. Set it to
`false` to stop on any invalid reference. Accepted references must still pass a
consistency check. Identity similarity determines **who** appears; it is not the
photo's dating-suitability score.

Each `[criteria.NAME]` contains a description and a nonnegative weight:

```toml
[criteria.expression]
weight = 3.0
description = "Prefer a natural, welcoming smile or relaxed engaged expression."

[criteria.story]
weight = 1.5
description = "Prefer a visible activity or setting that gives a date something to ask about."
```

Add, remove, or rewrite criteria. Each receives a 0–10 score with a reason; the total
is `sum(score × weight) / sum(weight)`. At least one weight must be positive. The
criteria must be assessable from the photo: the model cannot know your personality,
current appearance, or whether an activity is truly your hobby. It can make visual
mistakes. Use dates or a different library if recency matters. Face size has no
independent ranking bonus, so portraits, full-body shots, and activities can compete.

`[vision].exclude` contains editable hard exclusions. Defaults reject visibly
childhood photos, ID/document/collage images, and substantially obscured faces.

`[selection]` controls the final set:

- `top`: requested result count (default 10).
- `minimum_score`: minimum weighted score, from 0 to 10.
- `near_duplicate_distance`: perceptual-hash tolerance. `-1` disables near-duplicate
  suppression; exact duplicate suppression remains enabled.
- `diversity_bonus`: a small, diminishing bonus for underrepresented photo roles
  (portrait, full-body, activity, social, other). Set to `0` for score-only order.
  Roles are inferred by the model, and the bonus is a preference, not a quota.
- `not_before`: optional `YYYY-MM-DD` cutoff, or `""` for all years. Capture dates
  come from EXIF or a complete date in the path. Unknown dates stay eligible and
  are labelled; filesystem modification time is not used as capture time.

## Results, privacy, and format support

Results are written under `/state/projects/<library-reference-id>/results/<run>/`:

| Output | Contents |
| --- | --- |
| `index.html`, `thumbs/` | Recommended photos, explanations, and score breakdowns |
| `recommendations.json` | Ordered original paths, scores, reasons, and metadata |
| `all-scores.csv` | All identity matches, including excluded and duplicate candidates |
| `run.json` | Configuration snapshot, model/rubric fingerprint, coverage, and read errors |

`latest.json` in the project state directory points to its latest completed result.
Previous reports are preserved. All generated state is private and remains outside
the repository by default. It includes face embeddings, paths, caches, and previews;
back up or remove it according to your own needs. It is not uploaded by the app.

Inputs are mounted read-only. Originals are never edited, moved, renamed, tagged,
or deleted. The app has no cloud-photo analysis backend and requires no cloud API
credentials. Model/runtime downloads are the only application network downloads;
Docker/OS/Python setup also uses upstream package repositories.

Supported photos: JPEG, PNG, WebP, TIFF (first frame), BMP, HEIC/HEIF, and AVIF, with
EXIF orientation applied. RAW, videos, and other formats are outside this photo
scan. Symbolic links are skipped. Unreadable files are reported rather than ranked.
Face recognition can miss very small or obscured faces. Perceptual duplicate
matching is a heuristic and can group visually similar but distinct photos.

## Dependencies and model sources

All installation steps are defined in the repository; no manually installed
notebook, Node package, host virtual environment, or cloud service is required.

| Component | Pinned version / source | Installed by |
| --- | --- | --- |
| Ubuntu/ROCm base image | `rocm/dev-ubuntu-24.04:7.2.1`, pinned image digest | [Dockerfile](.devcontainer/Dockerfile) |
| OS libraries | Python 3.12, AMD MIGraphX/ROCm, Mesa Vulkan, image/SSL/OpenMP libraries, Git | Dockerfile apt package list; configured Ubuntu/AMD repositories |
| AMD ONNX Runtime | MIGraphX `1.23.2`, ROCm `7.2.1`, Python 3.12 Linux wheel from `repo.radeon.com` | [Hash-locked Python manifest](.devcontainer/requirements.lock) |
| Python libraries | InsightFace `2.0`, NumPy `2.2.6`, OpenCV `4.13.0.92`, Pillow `12.3.0`, pillow-heif `1.8.0`, ONNX `1.19.1`, plus pinned transitive dependencies | Same manifest, with exact wheel URLs/hashes; packages from PyPI except AMD runtime |
| Face detection + recognition | InsightFace `antelopev2`: `scrfd_10g_bnkps.onnx` and `glintr100.onnx` | [Official model archive](https://github.com/deepinsight/insightface/releases/download/model-zoo/antelopev2.zip), fetched by [`models.py`](photo_selector/models.py) |
| Visual language model | Qwen3-VL-4B-Instruct `Q4_K_M` | [Official GGUF repository, pinned revision](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/tree/1cd86afb9a95c410a6038ab3b40d8b578c892266), fetched by [`vision.py`](photo_selector/vision.py) |
| Vision encoder/projector | `mmproj-Qwen3VL-4B-Instruct-F16.gguf` | Same pinned Qwen repository |
| Local visual runtime | llama.cpp `b11157`, Ubuntu Vulkan x64 build | [Official binary release](https://github.com/ggml-org/llama.cpp/releases/tag/b11157), fetched by `vision.py` |

The model downloaders pin URLs/revisions and SHA-256 checksums in source. Face
weights are stored under `/state/models/antelopev2`; the vision weights and extracted
runtime under `/state/vision`. The initial face archive is approximately 361 MB;
vision weights/runtime are approximately 3.4 GB. These are not committed or baked
into the image: [`post-create.sh`](.devcontainer/post-create.sh) downloads/verifies
them into persistent host storage. A container rebuild can reuse that storage.

The base image, Python wheels, model bytes, and visual runtime are pinned. Apt
packages follow the base image's configured repositories; this is an automated
setup, not a fully hermetic OS-package snapshot. The human-readable primary package
list is [requirements.txt](.devcontainer/requirements.txt); Docker installs the full
hash-locked manifest. Update both when changing dependencies.

Upstream terms differ: InsightFace code is MIT, but its pretrained face models
are for **non-commercial research only** ([upstream terms](https://github.com/deepinsight/insightface/tree/master/python-package#license)).
Qwen's model repository declares Apache-2.0 ([model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/blob/1cd86afb9a95c410a6038ab3b40d8b578c892266/README.md));
llama.cpp is MIT ([license](https://github.com/ggml-org/llama.cpp/blob/b11157/LICENSE)).
Review the applicable upstream terms before using or redistributing those components.

## Development and troubleshooting

Run the tests in the container; they use generated fixtures rather than personal
photos and do not download models:

```bash
python3 -m unittest discover -s tests -v
```

- **Missing models / failed post-create download:** rerun `photo-selector models`.
- **Missing `/dev/kfd`, permissions, or no GPU provider:** check host driver support,
  device IDs/groups in `.env`, regenerate settings, rebuild, then run
  `photo-selector doctor`. Do not install PyPI `onnxruntime` or `onnxruntime-gpu`
  over AMD's runtime. InsightFace's upstream metadata requests CPU ONNX Runtime;
  installation deliberately uses `--no-deps` to preserve the AMD provider.
- **Visual model startup failure:** inspect `/state/vision-server.log`. The runtime
  requires Vulkan GPU 0 and sufficient free GPU memory.
- **Unresolved path variables:** run the app inside the generated devcontainer or
  use explicit accessible paths in your private TOML file.
- **Too few recommendations:** inspect `all-scores.csv`, then adjust the rubric,
  threshold, exclusions, or library. Ranking does not manufacture qualifying photos.

The repository excludes `.env`, private `selector*.toml` configs, generated
container settings, input photos under `.local-inputs`, assistant/editor state,
downloaded weights, and generated reports/caches. Only `selector.example.toml`
is public. The Docker build context is allowlisted to the Dockerfile and locked
requirements. Review staged files before publishing; do not force-add private data.
