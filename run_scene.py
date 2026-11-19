"""Run InfiniSplat v3 on one scene and log the run.

Usage:
    run_scene.bat <scene> <rgb|pose> [views] [offload|cpu] [sky]

Builds scenes/<scene>_v<N> from the source images, converts cameras for pose mode, runs the
official inference (or the offload variant below), and writes the result to
outputs/<mode>[_offload|_cpu][_sky]/<scene>_v<N>. Each run adds a row to outputs/runs.csv and
a GPU log to outputs/logs.
"""

import csv
import datetime
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path





# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parent
V3_DIR = REPO_DIR / "InfiniSplat" / "InfiniSplat_v3"
CLONE_GIT_DIR = REPO_DIR / "InfiniSplat" / ".git"

# The commit the clone is pinned to. run_offload mirrors inference.main at this commit
PINNED_COMMIT = "f2140ecdc94d38691804d689cce412858980a7bc"

# The gsplat-experiment folder holds the datasets. Override with GSPLAT_EXPERIMENT_DIR
GSPLAT_DIR_ENV = "GSPLAT_EXPERIMENT_DIR"
GSPLAT_DIR = Path(os.environ.get(GSPLAT_DIR_ENV) or (REPO_DIR.parent / "gsplat-experiment"))

SCENES_DIR = REPO_DIR / "scenes"
OUTPUTS_DIR = REPO_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
RUNS_CSV = OUTPUTS_DIR / "runs.csv"

# Same extensions as InfiniSplat's list_image_files, compared in lower case
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Resolutions are multiples of 14 with the aspect ratio of each scene, at about the pixel
# count of the default 756x756 (the default crops every image to a square)
RESOLUTION_OBJECT = (756, 756)
RESOLUTION_STUMP = (924, 616)
RESOLUTION_AMORPHOPHALLUS = (1022, 574)

CAMERAS_COLMAP = "colmap"
CAMERAS_JSON = "json"

# Scene table. images: source folder. cameras: model folder or json file for pose mode, or
# None when the scene is rgb only. note: shown when the images folder is missing
SCENES = {
    "object": {
        "images": V3_DIR / "examples" / "object" / "images",
        "cameras": V3_DIR / "examples" / "object" / "prior_camera.json",
        "cameras_kind": CAMERAS_JSON,
        "resolution": RESOLUTION_OBJECT,
        "note": "The bundled example is part of the InfiniSplat clone. Run setup.bat first.",
    },
    "stump": {
        "images": GSPLAT_DIR / "data" / "360_v2" / "stump" / "images",
        "cameras": GSPLAT_DIR / "data" / "360_v2" / "stump" / "sparse" / "0",
        "cameras_kind": CAMERAS_COLMAP,
        "resolution": RESOLUTION_STUMP,
        "note": "Expected in the gsplat-experiment folder (set %s to move it)." % GSPLAT_DIR_ENV,
    },
    "amorphophallus": {
        "images": GSPLAT_DIR / "data" / "amorphophallus" / "frames",
        "cameras": None,
        "cameras_kind": None,
        "resolution": RESOLUTION_AMORPHOPHALLUS,
        "note": "Expected in the gsplat-experiment folder (set %s to move it)." % GSPLAT_DIR_ENV,
    },
    "stump_rs": {
        "images": GSPLAT_DIR / "data" / "stump_rs" / "colmap" / "images",
        "cameras": GSPLAT_DIR / "data" / "stump_rs" / "colmap" / "sparse" / "0",
        "cameras_kind": CAMERAS_COLMAP,
        "resolution": RESOLUTION_STUMP,
        "note": "Only exists after running run_sfm.bat stump_rs in gsplat-experiment.",
    },
    "amorphophallus_rs": {
        "images": GSPLAT_DIR / "data" / "amorphophallus_rs" / "colmap" / "images",
        "cameras": GSPLAT_DIR / "data" / "amorphophallus_rs" / "colmap" / "sparse" / "0",
        "cameras_kind": CAMERAS_COLMAP,
        "resolution": RESOLUTION_AMORPHOPHALLUS,
        "note": "Only exists after running run_sfm.bat amorphophallus_rs in gsplat-experiment.",
    },
}

MODES = ("rgb", "pose")
OPTION_WORDS = ("offload", "cpu", "sky")
VIEWS_ALL = "all"

# Fraction of the card PyTorch may allocate. See set_memory_guard
GPU_MEMORY_FRACTION = 0.90

# Images each decoder turns into full resolution output at once in offload mode. InfiniSplat
# uses 8, which at 756x756 in FP32 takes about 4.7 GB in the depth head alone. Each image is
# decoded on its own, so 1 gives the same output with an eighth of that memory
OFFLOAD_DECODE_FRAMES = 1

# Seconds between nvidia-smi samples in the GPU log, and how long to wait for it to stop
GPU_LOG_INTERVAL_S = 2
GPU_LOG_STOP_TIMEOUT_S = 5
GPU_LOG_QUERY = "timestamp,memory.used,memory.total,temperature.gpu,power.draw,utilization.gpu"

# A ply header is a few hundred bytes. Stop looking for end_header after this many
PLY_HEADER_MAX_BYTES = 65536

GIB = 1024 ** 3
MIB = 1024 ** 2

CSV_HEADER = [
    "timestamp", "scene", "mode", "views", "total_images", "resolution", "options",
    "status", "seconds", "peak_gib", "gaussians", "ply_mb", "output_dir",
]

STATUS_OK = "ok"
STATUS_OOM = "oom"
STATUS_ERROR = "error"

USAGE = """Usage: run_scene.bat <scene> <rgb|pose> [views] [offload|cpu] [sky]
  scene    %s
  mode     rgb predicts the cameras, pose uses the scene's COLMAP model or prior cameras
  views    a positive integer or all (default all), evenly spaced over the sorted images
  offload  keep only one of the two models on the GPU at a time (slower, less memory)
  cpu      run on the CPU (very slow); cannot be combined with offload
  sky      remove the sky before merging Gaussians (--sky_seg)
Options may come in any order after the mode.""" % ", ".join(SCENES)





# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------

class UsageError(Exception):
    """Wrong command line arguments."""



class SceneError(Exception):
    """The scene cannot be built or has no data for the requested mode."""



def parse_cli(argv):
    """
    Parse the command line.

    Args:
        argv: arguments after the script name

    Returns:
        dict with scene, mode, views (int or None for all), offload, cpu, sky

    Raises:
        UsageError: when the arguments do not match the usage
    """
    if len(argv) < 2:
        raise UsageError("Needs at least a scene and a mode.")
    scene, mode = argv[0].lower(), argv[1].lower()
    if scene not in SCENES:
        raise UsageError("Unknown scene %r." % argv[0])
    if mode not in MODES:
        raise UsageError("Mode must be rgb or pose, got %r." % argv[1])
    rest = [a.lower() for a in argv[2:]]
    views = None
    if rest and rest[0] not in OPTION_WORDS:
        word = rest.pop(0)
        if word != VIEWS_ALL:
            if not word.isdigit() or int(word) < 1:
                raise UsageError("Views must be a positive integer or all, got %r." % word)
            views = int(word)
    for word in rest:
        if word not in OPTION_WORDS:
            raise UsageError("Unknown option %r." % word)
    options = {word: word in rest for word in OPTION_WORDS}
    if options["offload"] and options["cpu"]:
        raise UsageError("offload and cpu cannot be combined.")
    return {"scene": scene, "mode": mode, "views": views, **options}



def options_label(cli):
    """
    Names of the options in use, in the order used for the output folder suffix.

    Args:
        cli: dict from parse_cli

    Returns:
        list of option words, empty when none are set
    """
    return [word for word in OPTION_WORDS if cli[word]]



def output_root(cli):
    """
    Output root for a run: outputs/<mode> plus one suffix per option.

    Args:
        cli: dict from parse_cli

    Returns:
        Path of the output root, for example outputs/pose_offload_sky
    """
    return OUTPUTS_DIR / "".join([cli["mode"]] + ["_" + w for w in options_label(cli)])





# -----------------------------------------------------------------------------
# Scene building
# -----------------------------------------------------------------------------

def list_images(folder):
    """
    Image files of a folder, sorted by file name.

    Args:
        folder: Path of a flat image folder

    Returns:
        sorted list of Paths with a supported extension
    """
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)



def select_indices(total, count):
    """
    Evenly spaced indices over a sorted list.

    Args:
        total: number of images available
        count: number of views wanted, or None for all

    Returns:
        list of indices; everything when count is None or at least total, [0] for one view
    """
    if count is None or count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]



def build_scene(scene, views):
    """
    Build scenes/<scene>_v<N>/images with the selected images.

    An existing folder with exactly the same file names is reused, anything else is rebuilt.

    Args:
        scene: scene name, a key of SCENES
        views: number of views, or None for all

    Returns:
        (scene_dir, names, total) where names are the selected file names and total is the
        number of source images

    Raises:
        SceneError: when the source images are missing
    """
    source = SCENES[scene]["images"]
    if not source.is_dir():
        raise SceneError("Images not found at %s. %s" % (source, SCENES[scene]["note"]))
    images = list_images(source)
    if not images:
        raise SceneError("No images in %s." % source)
    total = len(images)
    chosen = [images[i] for i in select_indices(total, views)]
    names = [p.name for p in chosen]
    label = VIEWS_ALL if views is None or views >= total else str(views)
    scene_dir = SCENES_DIR / ("%s_v%s" % (scene, label))
    images_dir = scene_dir / "images"
    if images_dir.is_dir() and sorted(p.name for p in images_dir.iterdir()) == sorted(names):
        print("Reusing %s (%d images)" % (scene_dir, len(names)))
        return scene_dir, names, total
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    images_dir.mkdir(parents=True)
    print("Copying %d of %d images from %s to %s" % (len(chosen), total, source, images_dir))
    for path in chosen:
        shutil.copy2(path, images_dir / path.name)
    return scene_dir, names, total



def prepare_cameras(scene, scene_dir):
    """
    Put prior_camera.json into the scene folder for pose mode.

    Skipped when the json exists and is newer than the images folder. A COLMAP model goes
    through InfiniSplat's own scripts/convert_cameras.py, the bundled example json is copied.

    Args:
        scene: scene name, a key of SCENES
        scene_dir: Path of the scene folder built by build_scene

    Returns:
        Path of prior_camera.json

    Raises:
        SceneError: when the scene has no cameras, they are missing, or the conversion fails
    """
    config = SCENES[scene]
    cameras = config["cameras"]
    if cameras is None:
        raise SceneError("Scene %s has no camera model, so pose mode is not available. Use rgb." % scene)
    if not cameras.exists():
        raise SceneError("Cameras not found at %s. %s" % (cameras, config["note"]))
    images_dir = scene_dir / "images"
    target = scene_dir / "prior_camera.json"
    if target.is_file() and target.stat().st_mtime >= images_dir.stat().st_mtime:
        print("Reusing %s" % target)
        return target
    if config["cameras_kind"] == CAMERAS_JSON:
        shutil.copyfile(cameras, target)
        print("Copied %s" % target)
        return target
    command = [sys.executable, str(Path("scripts") / "convert_cameras.py"), "colmap",
               "--model", str(cameras), "--images", str(images_dir),
               "--output", str(target), "--overwrite"]
    print("Converting cameras:", " ".join(command))
    done = subprocess.run(command, cwd=str(V3_DIR))
    if done.returncode != 0 or not target.is_file():
        raise SceneError("convert_cameras.py failed (see messages above).")
    return target





# -----------------------------------------------------------------------------
# Checks and logging helpers
# -----------------------------------------------------------------------------

def check_clone_commit():
    """
    Warn when the InfiniSplat clone is not at the pinned commit.

    Reads InfiniSplat/.git/HEAD, a detached HEAD holding the sha. Never fails the run,
    because run_offload mirrors that commit and only the user can judge a difference.

    Returns:
        True when the clone is at PINNED_COMMIT, False otherwise
    """
    try:
        head = (CLONE_GIT_DIR / "HEAD").read_text().strip()
    except OSError as exc:
        print("WARNING: could not read %s (%s). run_offload mirrors commit %s." % (
            CLONE_GIT_DIR / "HEAD", exc, PINNED_COMMIT[:7]))
        return False
    if head == PINNED_COMMIT:
        return True
    print("WARNING: InfiniSplat is at %s, not the pinned %s. run_offload mirrors inference.main "
          "at the pinned commit and may be out of date." % (head[:40], PINNED_COMMIT[:7]))
    return False



def ply_vertex_count(path):
    """
    Number of Gaussians in a splat ply, read from its header.

    Each Gaussian is one vertex, so the header line "element vertex N" gives the count
    without reading the rest of the file.

    Args:
        path: ply file written by InfiniSplat

    Returns:
        the vertex count, or None when the header has no vertex element
    """
    with open(path, "rb") as f:
        header = f.read(PLY_HEADER_MAX_BYTES)
    header = header.split(b"end_header", 1)[0].decode("ascii", "replace")
    m = re.search(r"^element vertex (\d+)", header, re.MULTILINE)
    return int(m.group(1)) if m else None



def start_gpu_log(log_path):
    """
    Start nvidia-smi writing a csv log of memory, temperature, power and utilization.

    The output goes to the file through stdout. nvidia-smi's own -f option buffers and writes
    nothing when the process is terminated, which would leave the log empty.

    Args:
        log_path: Path of the csv file to write

    Returns:
        the Popen of nvidia-smi, or None when nvidia-smi is not found
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [exe, "--query-gpu=" + GPU_LOG_QUERY, "--format=csv", "-l", str(GPU_LOG_INTERVAL_S)]
    handle = open(log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.DEVNULL)
    except OSError:
        handle.close()
        return None
    proc.log_handle = handle
    return proc



def stop_gpu_log(proc):
    """
    Stop the nvidia-smi process started by start_gpu_log.

    Args:
        proc: Popen from start_gpu_log, or None

    Returns:
        None
    """
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=GPU_LOG_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
    proc.log_handle.close()



def set_memory_guard(torch):
    """
    Cap PyTorch's GPU allocations at GPU_MEMORY_FRACTION of the card and report free memory.

    Why: with the cap, PyTorch stops with a clean out of memory error at 90 percent of the
    card. Without it the Windows NVIDIA driver spills into system RAM, which slows everything
    to a crawl and, together with the earlier blue screens on this PC under full GPU load, is
    what we want to avoid.

    Args:
        torch: the imported torch module

    Returns:
        True when the guard was set, False when CUDA is not available
    """
    if not torch.cuda.is_available():
        return False
    free, total = torch.cuda.mem_get_info()
    print("GPU memory: %.2f GiB free of %.2f GiB, PyTorch capped at %.0f%%" % (
        free / GIB, total / GIB, GPU_MEMORY_FRACTION * 100))
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    torch.cuda.reset_peak_memory_stats()
    return True



def is_oom(exc):
    """
    Whether an exception is a CUDA out of memory error.

    Args:
        exc: the exception

    Returns:
        True for torch.cuda.OutOfMemoryError or a RuntimeError saying out of memory
    """
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()





# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

def enter_v3():
    """
    Make InfiniSplat's imports and relative YAML paths work.

    Their imports are "from src..." and the YAML paths are relative to the repo root, so V3_DIR
    goes first on sys.path and becomes the working directory.

    Returns:
        None
    """
    if str(V3_DIR) not in sys.path:
        sys.path.insert(0, str(V3_DIR))
    os.chdir(V3_DIR)



def run_offload(argv):
    """
    Inference with one model on the GPU at a time.

    Mirrors inference.main at InfiniSplat commit f2140ec (lines 122 to 179) step by step and
    changes two things. First, where the two models live: with both resident, the weights
    alone take about 8.9 GB of the 12 GB card (3.8 GB Gaussian model, 5.1 GB WorldMirror 2.0).
    Here both models are built and loaded on the CPU, the geometry model (teacher) goes to the
    GPU for its forward pass and back, then the Gaussian model goes to the GPU. The views and
    the geometry stay on the GPU. Second, both decoders turn OFFLOAD_DECODE_FRAMES images into
    full resolution output at a time instead of 8, through the settings their own code takes.

    Args:
        argv: command line for inference.parse_args, without --device

    Returns:
        None, results are written to the output folder like inference.main does
    """
    import inference
    args = inference.parse_args(argv)
    cfg = inference.load_config(args)
    import torch
    from omegaconf import OmegaConf
    from src.models.models.gaussian_worldmirror import GaussianWorldMirror
    from src.models.models.geometry_worldmirror import GeometryWorldMirror
    from src.utils.checkpoints import load_model_weights
    from src.utils.scene_io import build_views, export_gaussians, export_points, export_depth_rgb
    from src.utils.sky_segmentation import SkySegmenter

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use a working CUDA/PyTorch environment, or --device cpu for a slow smoke run.')
    segmenter = SkySegmenter(args.sky_onnx_path, device=args.device) if args.sky_seg else None
    views, scene_name, img_paths = build_views(
        args.images_path, args.cameras, tuple(args.resolution), device,
        cam_align=not args.no_cam_align,
        sky_segmenter=segmenter, sky_max_fraction=args.sky_max_fraction)
    cond_flags = [1, 0, 1] if cfg.wrapper.enable_cond else [0, 0, 0]
    # Both models are built and loaded on the CPU, then moved to the GPU one at a time
    model = GaussianWorldMirror(**OmegaConf.to_container(cfg.wrapper.model),
                                enable_cond=bool(cfg.wrapper.enable_cond),
                                filter_sky_infer=args.sky_seg,
                                decode_chunk_size=OFFLOAD_DECODE_FRAMES)
    teacher = GeometryWorldMirror(enable_cond=bool(cfg.wrapper.enable_cond))
    # GeometryWorldMirror.forward calls its depth head with the default of 8 images at a time
    teacher.depth_head.forward = functools.partial(
        teacher.depth_head.forward, frames_chunk_size=OFFLOAD_DECODE_FRAMES)
    load_model_weights(model, cfg.wrapper.SplatAnything_pretrained)
    load_model_weights(teacher, cfg.wrapper.worldmirror2_pretrained)
    model.eval().requires_grad_(False)
    teacher.eval().requires_grad_(False)
    print(f'[Input] {scene_name}: {views["img"].shape[1]} images, resolution={args.resolution}, '
          f'enable_cond={cfg.wrapper.enable_cond}, use_gt_camera={cfg.wrapper.model.use_gt_camera}, sky_seg={args.sky_seg}')
    started = time.perf_counter()
    with torch.inference_mode():
        teacher.to(device)
        geometry = teacher(views, cond_flags=cond_flags)
        teacher.to('cpu')
        torch.cuda.empty_cache()
        model.to(device)
        preds = model(views, geometry, cond_flags=cond_flags)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f'[Inference] {time.perf_counter() - started:.2f}s')
    outdir = Path(args.output_path) / scene_name
    outdir.mkdir(parents=True, exist_ok=True)
    # An empty filtered scene must not be mistaken for a successful reconstruction.
    if not len(preds['splats']['means'][0]):
        raise RuntimeError('All Gaussians were filtered. Check the input, sky masks, and scale threshold.')
    if not (preds['splats']['opacities'][0] >= cfg.wrapper.infer_opacity_thresh).any():
        raise RuntimeError('All Gaussians are below opacity_thresh. Lower the threshold and check the checkpoint.')
    export_gaussians(preds, outdir, cfg.wrapper.infer_opacity_thresh, None)
    export_points(preds, outdir)
    if not args.no_depth_rgb:
        export_depth_rgb(preds, views, img_paths, outdir)
    cameras = {'extrinsics': [], 'intrinsics': []}
    for index, image in enumerate(img_paths):
        camera_id = Path(image).stem
        for name, key in [('extrinsics', 'export_camera_poses'), ('intrinsics', 'export_camera_intrs')]:
            cameras[name].append({'camera_id': camera_id, 'matrix': preds[key][0, index].cpu().tolist()})
    (outdir / 'cameras.json').write_text(json.dumps(cameras, indent=2) + '\n')
    OmegaConf.save(cfg, outdir / 'config.yaml', resolve=True)
    (outdir / 'run.json').write_text(json.dumps({**vars(args), 'images': img_paths, 'cond_flags': cond_flags}, indent=2) + '\n')
    print(f'[Done] {outdir}')



def build_inference_argv(cli, scene_dir, root, resolution):
    """
    Command line for InfiniSplat's inference.main.

    Args:
        cli: dict from parse_cli
        scene_dir: Path of the scene folder
        root: Path of the output root
        resolution: (W, H) tuple

    Returns:
        list of argument strings; --device cpu is added for the cpu option
    """
    argv = ["--input_path", str(scene_dir), "--mode", cli["mode"], "--output_path", str(root),
            "--resolution", str(resolution[0]), str(resolution[1])]
    if cli["sky"]:
        argv.append("--sky_seg")
    if cli["cpu"]:
        argv += ["--device", "cpu"]
    return argv



def run_inference(cli, argv):
    """
    Run the inference, guarded against GPU memory exhaustion, and measure time and peak memory.

    The time covers the whole call, including image loading and model loading.

    Args:
        cli: dict from parse_cli
        argv: command line from build_inference_argv

    Returns:
        dict with status (ok, oom or error), seconds, peak_gib (None on cpu) and error text
    """
    result = {"status": STATUS_ERROR, "seconds": None, "peak_gib": None, "error": ""}
    enter_v3()
    torch = None
    started = time.perf_counter()
    try:
        if not cli["cpu"]:
            import torch
            set_memory_guard(torch)
            started = time.perf_counter()
        if cli["offload"]:
            run_offload(argv)
        else:
            import inference
            inference.main(argv)
        result["status"] = STATUS_OK
    except Exception as exc:
        if is_oom(exc):
            result["status"] = STATUS_OOM
            result["error"] = "Out of GPU memory"
            print("\nERROR: out of GPU memory at %s views. Try fewer views, offload, or cpu." % (
                cli["views_label"]))
        else:
            result["error"] = "%s: %s" % (type(exc).__name__, exc)
            traceback.print_exc()
    finally:
        result["seconds"] = time.perf_counter() - started
        if torch is not None and torch.cuda.is_available():
            result["peak_gib"] = torch.cuda.max_memory_allocated() / GIB
    return result





# -----------------------------------------------------------------------------
# Summary and run log
# -----------------------------------------------------------------------------

def ply_info(out_dir):
    """
    Gaussian count and size of the gaussians.ply in an output folder.

    Args:
        out_dir: Path of the run's output folder

    Returns:
        (count, size in MB), each None when the file is missing or unreadable
    """
    path = out_dir / "gaussians.ply"
    if not path.is_file():
        return None, None
    try:
        return ply_vertex_count(path), path.stat().st_size / MIB
    except OSError:
        return None, path.stat().st_size / MIB



def print_summary(info, result, gaussians, ply_mb):
    """
    Print the run summary.

    Args:
        info: dict with scene, mode, views, total, resolution, options and out_dir
        result: dict from run_inference
        gaussians: Gaussian count or None
        ply_mb: size of gaussians.ply in MB or None

    Returns:
        None
    """
    shown = os.path.relpath(info["out_dir"], REPO_DIR)
    seconds = "n/a" if result["seconds"] is None else "%.1f s" % result["seconds"]
    peak = "n/a" if result["peak_gib"] is None else "%.2f GiB" % result["peak_gib"]
    print()
    print("Run summary: %s" % shown)
    print("  scene %s, mode %s, views %d of %d, resolution %dx%d, options %s" % (
        info["scene"], info["mode"], info["views"], info["total"],
        info["resolution"][0], info["resolution"][1], info["options"]))
    print("  time %s, peak GPU memory %s" % (seconds, peak))
    if gaussians is None and ply_mb is None:
        print("  gaussians.ply: not written")
    else:
        count = "n/a" if gaussians is None else format(gaussians, ",d")
        print("  gaussians.ply: %s Gaussians, %.0f MB" % (count, ply_mb))
    if result["status"] != STATUS_OK:
        print("  status %s: %s" % (result["status"], result["error"]))



def append_run_row(info, result, gaussians, ply_mb):
    """
    Append one row to outputs/runs.csv, creating it with a header row if needed.

    Args:
        info: dict with scene, mode, views, total, resolution, options and out_dir
        result: dict from run_inference
        gaussians: Gaussian count or None
        ply_mb: size of gaussians.ply in MB or None

    Returns:
        None
    """
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RUNS_CSV.is_file()
    row = [
        datetime.datetime.now().isoformat(timespec="seconds"), info["scene"], info["mode"],
        info["views"], info["total"], "%dx%d" % tuple(info["resolution"]), info["options"],
        result["status"],
        "" if result["seconds"] is None else "%.1f" % result["seconds"],
        "" if result["peak_gib"] is None else "%.2f" % result["peak_gib"],
        "" if gaussians is None else gaussians,
        "" if ply_mb is None else "%.1f" % ply_mb,
        os.path.relpath(info["out_dir"], REPO_DIR),
    ]
    with open(RUNS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(CSV_HEADER)
        writer.writerow(row)





# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv=None):
    """
    Build the scene, run InfiniSplat, print the summary and log the run.

    Args:
        argv: arguments after the script name, defaults to sys.argv[1:]

    Returns:
        process exit code, 0 for a finished run and 1 otherwise
    """
    argv = sys.argv[1:] if argv is None else argv
    try:
        cli = parse_cli(argv)
    except UsageError as exc:
        print("ERROR: %s\n" % exc)
        print(USAGE)
        return 1
    scene, mode = cli["scene"], cli["mode"]
    if not (V3_DIR / "inference.py").is_file():
        print("ERROR: %s not found. Run setup.bat first." % (V3_DIR / "inference.py"))
        return 1
    try:
        scene_dir, names, total = build_scene(scene, cli["views"])
        if mode == "pose":
            prepare_cameras(scene, scene_dir)
    except SceneError as exc:
        print("ERROR: %s" % exc)
        return 1
    check_clone_commit()

    views = len(names)
    cli["views_label"] = str(views)
    resolution = SCENES[scene]["resolution"]
    root = output_root(cli)
    out_dir = root / scene_dir.name
    options = "+".join(options_label(cli)) or "none"
    info = {"scene": scene, "mode": mode, "views": views, "total": total,
            "resolution": resolution, "options": options, "out_dir": out_dir}
    suffix = root.name
    log_path = LOGS_DIR / ("%s_%s_gpu.csv" % (scene_dir.name, suffix))

    print("Scene %s, mode %s, %d of %d images, resolution %dx%d, options %s" % (
        scene, mode, views, total, resolution[0], resolution[1], options))
    gpu_log = start_gpu_log(log_path)
    try:
        result = run_inference(cli, build_inference_argv(cli, scene_dir, root, resolution))
    finally:
        stop_gpu_log(gpu_log)

    gaussians, ply_mb = ply_info(out_dir) if result["status"] == STATUS_OK else (None, None)
    print_summary(info, result, gaussians, ply_mb)
    append_run_row(info, result, gaussians, ply_mb)
    return 0 if result["status"] == STATUS_OK else 1



if __name__ == "__main__":
    sys.exit(main())
