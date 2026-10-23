# arrow-swift

[ARROW](https://github.com/ifradlin/arrow) ([paper](https://arxiv.org/abs/2610.01314), [weights](https://huggingface.co/ifradlin/arrow)) running natively on Apple silicon with [MLX](https://github.com/ml-explore/mlx-swift): 3D reconstruction, 3D point tracking, camera poses and depth from a video, a photo set, or several synchronized videos of one scene. The repository also holds **Quiver**, a Mac app for running the model and looking at what it predicts.

<p align="center">
  <img width="480" alt="quiver" src="https://github.com/user-attachments/assets/f211a940-3745-409b-8717-636abbbb2fa5" />
</p>

- **`packages/ArrowKit`** is the model as a Swift package. Its outputs match the PyTorch reference: the encoder and every head reach a cosine similarity of 0.9999 or better against outputs captured from upstream on CPU in fp32. It runs in fp16 by default.
- **`apps/Quiver`** is the app. You open a clip, pick frames and seeds, and run. The result appears in a 3D view with the input frame and its tracks below. It can save results to reopen without the model, and export them to USD.

This is an independent port. The model, its weights and the method are the work of the ARROW authors (see [Credits](#credits)).

## Requirements

- A Mac with Apple silicon and macOS 15 or later; Xcode 16 or later to build from source.
- About 16 GB of memory for clips of up to a few dozen frames. See [Long clips](#long-clips) for more.
- The weights, [`ifradlin/arrow`](https://huggingface.co/ifradlin/arrow) (5.3 GB). They download into the Hugging Face cache on first run, and an existing cache from the Python tools is reused.

## Demo App (Quiver)
<p align="center">
  <img width="128" height="128" alt="the quiver mac demo app icon" src="https://github.com/user-attachments/assets/bc7f9ceb-b7bf-4000-b9f2-abd679c8667e" />
</p>

<img width="2052" height="1374" alt="demo app screenshot: moving vs separate-tracks"  src="https://github.com/user-attachments/assets/91f9159f-5f82-4f21-ada0-e9fa10cefb30" />

**[Download the latest Quiver for Mac](https://github.com/melMass/arrow-swift/releases/latest)**: signed and notarized, for Apple silicon on macOS 15 or later. Unzip and open it; the weights download on the first run.

To build it yourself instead:

```sh
scripts/build-app.sh --open
```

This builds `build/Quiver.app` and opens it. The build goes through `xcodebuild`, because SwiftPM alone does not compile MLX's Metal shaders. The app is ad-hoc signed; set `CODESIGN_IDENTITY` to sign it with your own certificate.

**Input.** Open a video, a folder of images or a single image, or drop one on the window. There are two ways to pick frames:
- **Spread** takes N frames evenly across the clip.
- **Range** takes every k-th frame from one frame to another, up to a count: upstream's start, stride and max frames (24 by default), plus an end. With a step of 1 and a high enough count it reads every frame of a clip.

**Add Camera…** adds another video of the same scene for multi-view input (see [Multi-view](#multi-view)).

**Tasks.**
- **3D tracking**: points from one frame, followed through every frame. **All pixels** tracks every N-th pixel of the clip once instead (upstream's all-pixel tracking): the start frame's whole grid, then, frame by frame, the pixels that no earlier track was seen at.
- **3D reconstruction**: every frame's pixels in the first frame's camera.
- **Camera poses**: a rigid fit and intrinsics per frame.
- **Video depth**: each frame in its own camera.

**Skip sky** (off by default) leaves sky pixels unasked, as upstream does, so outdoor clips lose their stray far points. The segmenter is upstream's [skyseg](https://huggingface.co/JianyuanWang/skyseg) converted to Core ML ([melmass/skyseg-coreml](https://huggingface.co/melmass/skyseg-coreml), 84 MB, downloaded on first use) and runs on the Neural Engine at about 17 ms per frame.

**Seeds.** Drag a region on the frame to bound the seed grid, or click points to track exactly those pixels. Seeds belong to the frame they were placed on, and tracks start from that frame. **Inspect…** zooms in to 1600% to check the exact query pixels. With **All pixels**, the region and clicked points are set aside and tracks start from the frame shown. Each frame's seeds depend on where the earlier tracks went, so its tracks are decoded before the next frame is seeded, and they appear frame by frame as the run streams in.

**Display.** These follow the upstream viewer:
- a confidence filter, starting at upstream's "recommended" cut;
- point colours by RGB, confidence, visibility, or predicted or estimated normals;
- track colours by seed position, RGB, time or confidence;
- trails, and hidden points shown, dimmed or dropped;
- still and moving tracks split apart;
- frusta along the camera path, with the current frame on its frustum;
- a depth panel with depth, normals and visibility.

**Output.**
- **Save Result** (⌘S) writes a `.quiver` file: the frames, every query and prediction, and the cameras. It reopens without the model.
- **Export USD** (⌘E) writes a `.usda` stage of what is shown: the cloud, per-frame depth, tracks animated over time, their paths as curves, and the cameras with their intrinsics. The stage passes `usdchecker`. Its axes are Y up, and the scale is the model's own, which is relative and not metres.

## ArrowKit

```swift
import ArrowKit

let frames = try await FrameSet.load(.video(url), selection: .spread(16), maxSide: 504)
let runner = try await ArrowRunner.load()          // downloads the weights on first use
var settings = TaskSettings()
settings.trackStride = 12                          // a seed every 12 px
// or: settings.trackAllPixels = true; settings.allPixelsStride = 4
let result = try await runner.run(frames, tasks: [.track, .reconstruct, .pose], settings: settings)

let tracks = TrackSet(result, result.group(.track)!)
let point = tracks.points[tracks.index(5, 0)]      // seed 0 at frame 5, in frame 0's camera
try USDExport.write(result, to: URL(filePath: "scene.usda"))
try ResultArchive.write(result, sources: [url], to: URL(filePath: "scene.quiver"))
```

- ARROW encodes a set of frames once, then answers *queries* against that encoding: the 3D point seen at a pixel of a source frame, located in a target frame and expressed in a reference camera.
- `ArrowRunner` keeps the last encoding, so asking new questions about the same frames skips the encoder. The decoder's keys and values depend only on the frames, so they are computed once per encoding and shared by every chunk of queries.
- Each task is a block of queries (`TaskSettings.queries`), and `ArrowResult.groups` says where each block sits. All-pixel tracking is the exception: its block grows frame by frame (`PixelCoverage`), and ends up laid out like any other track group.
- `run` streams its progress: `started` gives the queries before any answer, then `answered` chunks fill them in. All-pixel tracking adds `grew`, the run laid out anew with another frame's tracks, and `placed` chunks for them.
- Cameras come back in the frame that tracks and reconstruction are expressed in: frame 0, which is the identity.

## Multi-view

Several synchronized videos of one scene are one frame set. Frame `t · views + v` is camera v at timestep t, the same order upstream uses:

```swift
let frames = try await FrameSet.load(views: [.video(left), .video(right)],
                                     selection: .range(start: 0, end: 48, step: 2, limit: 24))
```

- The streams must already be aligned frame for frame; nothing estimates a time offset. Cameras of different lengths are read over the shortest.
- Without a drawn region, every camera's frame at the seed timestep gets the seed grid, as upstream seeds a rig. A drawn region and clicked points stay on the camera they were drawn on.
- `TrackSet` folds the cameras' answers into one track per timestep, in one of two modes:
  - **Camera median**: the per-axis median of the cameras that see the point.
  - **Source camera**: the prediction from the camera the track was seeded in.
- Quiver shows one camera's frame below the 3D view and one camera path per view. USD export writes one animated camera per view.

## Long clips

ARROW's encoder attends across every frame, and every query reads every frame's patches. Memory and time therefore grow with the number of frames, and a query costs more the more frames there are.

On an M1 Pro (16 GB), 8 frames at 504 × 280 encode in about 8 s, and 95,000 queries take about 16 s.

The decoder's keys and values take about 30 MB per frame at 504 px, on top of 2.6 GB of fp16 weights. Up to 2 GB of them (about 70 frames at 504 px) are computed once and kept. Past that, only the projected patches are kept, a sixteenth of the size, and each batch of queries recomputes the keys and values one block at a time. The answers are identical, and each batch takes a little longer. `ArrowModel.encode(_:memoryBudget:)` sets the limit. Time still grows with the number of frames, because every query reads all of them. Neither the model nor the app caps the frame count; Quiver warns past 48 images.

All-pixel tracking adds a block of tracks per frame, each asked in every frame. On the same M1 Pro, 8 frames at 504 × 280 with a pixel every 4 px gave 18,000 tracks (145,000 queries); with reconstruction and poses, the run took 45 s to decode.

## Tests

```sh
packages/ArrowKit/test.sh                        # queries, grids, poses, frame selection, multi-view folding, all-pixel seeding
ARROW_MODEL_TESTS=1 packages/ArrowKit/test.sh    # + parity against PyTorch (needs the weights)
ARROW_MODEL_TESTS=1 ARROW_CLIP=/path/to/railway.mp4 packages/ArrowKit/test.sh   # + end to end: one and two cameras, all pixels
(cd apps/Quiver && swift test)                   # scene building, streaming, USD export, archives, seeds
```

- `test.sh` wraps `swift test` and puts MLX's shaders where MLX looks for them.
- The parity fixtures in `packages/ArrowKit/Fixtures` were captured from upstream at commit `6fca6f8` with `capture.py`. Its docstring says how to rerun it.
- `railway.mp4` is the sample clip in upstream's `assets/videos`.

## Credits

ARROW is by Ilya Fradlin, Christian Schmidt, Jens Piekenbrinck, Karim Knaebel, Gonzalo Martin Garcia and Bastian Leibe:

```bibtex
@article{fradlin2026arrow,
    title   = {{ARROW}: Arbitrary Reconstruction and Tracking of 4D Observations in the Wild},
    author  = {Fradlin, Ilya and Schmidt, Christian and Piekenbrinck, Jens and Knaebel, Karim and Martin Garcia, Gonzalo and Leibe, Bastian},
    year    = 2026,
    journal = {arXiv preprint arXiv:2610.01314},
}
```

The encoder is Depth Anything 3's ViT-Giant (DINOv2). Upstream's code is MIT-licensed, and its model layers are Apache-2.0. The weights are for non-commercial use and are also subject to Waymo's terms, as upstream's [LICENSE](https://github.com/ifradlin/arrow/blob/main/LICENSE) states. They are downloaded from Hugging Face, not redistributed here.

## License

[MPL-2.0](LICENSE). Upstream's notices are in [NOTICE](NOTICE).
