# InfiniSplat Test

Runs InfiniSplat v3 (SplatAnything) from the official repo on the Windows lab PC (RTX 5070 12 GB, no admin rights) inside conda, on the same stump and Amorphophallus data as `gsplat-experiment`. InfiniSplat predicts a Gaussian splat from photos in one forward pass of a trained network, with no per-scene training, so this compares it against gsplat's trained splats.

This document covers:

* Setting up the lab PC with one click
* The scenes and the run command
* Staying within GPU memory
* The order to run the experiment in
* Reviewing the results
* Findings so far
* Deviations from the official InfiniSplat docs



## Related

* Official repo: https://github.com/zju3dv/InfiniSplat (v3 in `InfiniSplat_v3`)
* v3 docs: `InfiniSplat\InfiniSplat_v3\README.md`, `docs\inference.md`, `docs\prepare_data.md`
* Paper: SplatAnything, https://arxiv.org/abs/2611.08264
* Weights: https://huggingface.co/ritianyu/SplatAnything
* The gsplat runs to compare against: `gsplat-experiment`, which must sit next to this folder
* SuperSplat, for viewing the splats: https://superspl.at/editor



## Versions

* InfiniSplat v3 (SplatAnything), pinned at commit f2140ec (2026-11-14), clone `InfiniSplat`, env `infinisplat3`
* Weights pinned at Hugging Face revision cea5713 (2026-11-08)
* Python 3.11 and torch 2.9.1 cu128, the same torch as `gsplat-experiment`

v1 takes a single image and v2 at most 50 views, so neither fits a 125 image 360. v2's optional refiner is a possible second phase.



## Setup

1. Clone this repo next to `gsplat-experiment`, for example both on the Desktop
2. Double click `setup.bat`. It creates the conda env, clones InfiniSplat at the pinned commit, installs torch and `requirements.txt`, downloads the weights and runs InfiniSplat's own unit tests
3. The weights are about 16.6 GB and go to `InfiniSplat\InfiniSplat_v3\checkpoints`. Each file is checked against the hash Hugging Face lists


### Notes

* Run `setup.bat` again after a reboot. It only installs or downloads what is missing
* The test step prints unittest's own `FAILED (errors=1)` for the one known out of date test (see Findings). The line after it, `failed outside the known upstream ones: 0`, is the one that counts
* A download that stops halfway resumes on the next run
* The weight files:
  * `SplatAnything.ckpt` (11.3 GB): the Gaussian model. Only its 3.8 GB of weights are loaded, the rest is training state
  * `worldmirror2.safetensors` (5.1 GB): WorldMirror 2.0, which predicts depth and cameras for the Gaussian model
  * `skyseg.onnx` (176 MB): optional sky segmentation, used only with the `sky` option



## Scenes and commands

The scenes:

* `object`: InfiniSplat's own example (23 images, with cameras), to check the install
* `stump`: the Mip-NeRF 360 stump at full size, 125 images, with its official COLMAP poses for pose mode
* `amorphophallus`: the 240 frames of the frontend's 360, RGB mode only
* `stump_rs`, `amorphophallus_rs`: the RealityScan exports from `gsplat-experiment` (run `run_sfm.bat` there first), with RealityScan poses for pose mode

The command:

```bat
run_scene.bat <scene> <rgb|pose> [views] [offload|cpu] [sky]
```

* `rgb` lets InfiniSplat predict the cameras, `pose` gives it the known ones
* `views` picks that many images evenly around the capture, `all` by default
* `offload` and `cpu` lower GPU memory use, see Staying within GPU memory
* `sky` removes sky pixels before the Gaussians are merged, with InfiniSplat's own `--sky_seg`. Without it, open sky can turn into a far away shell of Gaussians


### Notes

* Each run builds `scenes\<scene>_v<views>\images`, plus `prior_camera.json` for pose mode, written by InfiniSplat's own `scripts\convert_cameras.py`
* Results go to `outputs\<mode>\<scene>_v<views>`, with `_offload`, `_cpu` or `_sky` added to the mode folder when used, for example `outputs\rgb_offload\stump_v32`. Each holds `gaussians.ply`, `points.ply`, `cameras.json`, and the processed images and depth
* Every run adds a line to `outputs\runs.csv`: scene, mode, views, resolution, options, status, time, peak GPU memory and the number of Gaussians
* Resolution follows each scene's aspect ratio at about the pixel count of InfiniSplat's default 756x756: 924x616 for stump, 1022x574 for Amorphophallus, 756x756 for `object`



## Staying within GPU memory

InfiniSplat loads two models: the Gaussian model (3.8 GB) and WorldMirror 2.0 (5.1 GB). The official `inference.py` keeps both on the GPU, about 8.9 GB before any image, and sends all views through the network at once. On the 12 GB card that leaves about 3 GB for the views.

Every run has these guards:

* PyTorch may use at most 90% of the GPU. A run that needs more stops with an out of memory error and the summary says so, instead of the driver spilling into system RAM, which slows the whole PC to a crawl
* The GPU is logged every 2 seconds to `outputs\logs\<scene>_v<views>_<mode folder>_gpu.csv` while a run is going, for example `stump_v32_rgb_offload_gpu.csv`

Two options lower memory use:

* `offload`: only one model sits on the GPU at a time. WorldMirror 2.0 runs first and goes back to RAM, then the Gaussian model runs. This frees at least 3.8 GB for the views. Both models also turn their results into full resolution output 1 image at a time instead of 8, which at 756x756 in FP32 saves about 4 GB in the depth step alone. Each image is decoded on its own, so the output is the same up to float rounding (largest difference 3e-6, checked with InfiniSplat's own decoder classes)
* `cpu`: runs everything on the CPU with InfiniSplat's own `--device cpu`, in system RAM. InfiniSplat calls this a slow smoke run, so expect minutes or more, but the GPU is not used at all


### Notes

* If a run stops with out of memory, try fewer views, then `offload`, then `cpu`
* An out of memory error stops only the run. If the PC still restarts, check the GPU log and see the crash steps in `gsplat-experiment`'s README
* `offload` reproduces the official `inference.main` step by step, changing only where the two models are kept and how many images the decoders take at once, both through settings InfiniSplat's own code accepts. It mirrors commit f2140ec, and `run_scene.py` warns if the clone is at another commit



## Running the experiment

Run one at a time. Stop going up in views at the first out of memory error.

1. Check the install on InfiniSplat's own example:

```bat
run_scene.bat object rgb offload
```

2. Find how many stump views fit, going up:

```bat
run_scene.bat stump rgb 16 offload
```

```bat
run_scene.bat stump rgb 32 offload
```

```bat
run_scene.bat stump rgb 64 offload
```

```bat
run_scene.bat stump rgb all offload
```

3. At the highest view count that fit, give it the official poses:

```bat
run_scene.bat stump pose 32 offload
```

4. Amorphophallus, the same way:

```bat
run_scene.bat amorphophallus rgb 16 offload
```

```bat
run_scene.bat amorphophallus rgb 32 offload
```

5. Optional, with RealityScan poses or sky removal:

```bat
run_scene.bat stump_rs pose 32 offload
```

```bat
run_scene.bat stump rgb 32 offload sky
```


### Notes

* Use `offload`. The official path (no option) keeps both models on the GPU and decodes 8 images at a time, which needs more than this card has even for the bundled example
* `rgb` against `pose` on stump shows whether InfiniSplat's own camera estimate holds up against the official poses
* Amorphophallus is where InfiniSplat could differ most from RealityScan and gsplat, since it does not need the frames to align strictly before it starts



## Reviewing results

Every run ends with a summary like this:

```
Run summary: outputs\rgb_offload\stump_v32
  scene stump, mode rgb, views 32 of 125, resolution 924x616, options offload
  time 41.2 s, peak GPU memory 7.10 GiB
  gaussians.ply: 2,345,678 Gaussians, 154 MB
```

Open `gaussians.ply` in SuperSplat next to the gsplat splat of the same scene, for example `gsplat-experiment\results\153\30k_mcmc\stump\ply\point_cloud_29999.ply`.


### Notes

* InfiniSplat stores colour only, with no view-dependent shading, so shiny surfaces look flatter than in gsplat
* The splat is in InfiniSplat's own normalised scale, anchored to the first camera, so it sits at a different size and position from the gsplat splat
* InfiniSplat has no PSNR, SSIM or LPIPS script, so the comparison is visual plus time, memory and Gaussian count



## Findings

* InfiniSplat v3's unit tests pass on Windows, with a CPU build of torch on a laptop, apart from two that are out of date upstream on every OS: `tests\test_inference.py` imports `validation`, the old name of `inference.py`, and one test reads `prepare_data.md` from the v3 folder, which now sits in `docs`. With `validation` pointing at `inference.py`, 44 of 45 tests pass
* On the lab PC two more tests failed, because the user name is longer than 8 characters. `%TEMP%` then holds the Windows short name (`C:\Users\102789~1\...`), `inference.py` expands it to the long one, and the tests compare the two spellings of the same folder. With temp folders at their long path, the result matches the laptop
* The first lab PC runs with `offload` ran out of memory while the decoders took 8 images at a time: `object` at 23 views (peak 9.84 GiB) and stump at 16 views (peak 10.57 GiB), against the 10.7 GiB cap. The decoders now take 1 image at a time



## Deviations

* conda env `infinisplat3` with Python 3.11, because the docs only ask for Python 3.10+ and no environment tool
* torch 2.9.1 cu128 from the PyTorch index, because the docs ask for "a CUDA build compatible with your NVIDIA driver" and the RTX 5070 needs CUDA 12.8
* The repo is cloned with `core.longpaths=true`, because some of its paths are too long for Windows' default limit
* Weights are downloaded with `curl` from Hugging Face file URLs at a pinned revision and checked against the hashes Hugging Face lists, because the docs only link the Hugging Face page
* The unit tests run with `validation` pointing at `inference.py`, and `test_documented_camera_json_matches_loader` may fail, because both are out of date upstream (see Findings). Temp folders for the tests use their long path, so a short Windows user folder name does not fail two of them
* `--resolution` matches each scene's aspect ratio at about the pixel count of the default 756x756, because the default crops every image to a square
* View subsets, because all views go through the network at once and the 12 GB GPU holds about 3 GB besides the weights
* `run_scene.py` calls InfiniSplat's own `inference.main` and records time and peak GPU memory. InfiniSplat's code is unchanged
* PyTorch is capped at 90% of GPU memory and the GPU is logged during each run, so an out of memory case stops cleanly instead of spilling into system RAM
* `offload` reproduces `inference.main` with only one model on the GPU at a time, because both together take about 8.9 GB of the 12 GB, and decodes 1 image at a time instead of 8 (`decode_chunk_size` of the Gaussian model, `frames_chunk_size` of the depth head), because 8 at full resolution ran out of memory on the lab PC
* Stump pose mode uses the full size `images`, because the official COLMAP cameras are calibrated at full size and the converter rejects resized images
