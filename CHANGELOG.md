# Changelog

All notable changes to arrow-swift: ArrowKit and Quiver.

## [0.2.0] - 2026-10-23

### Added
- **quiver:** Let range sampling pick its own frame count instead of a fixed 300 cap
- **quiver:** Add all pixels tracking mode that covers the whole clip
- **arrow-kit:** Add all-pixel tracking, seeded frame by frame from uncovered pixels
- **quiver:** Show only the current frame's cloud unless accumulate is on
- **quiver:** Add skip sky toggle backed by the sky segmenter
- **arrow-kit:** Add optional sky skipping via a core ml port of upstream's skyseg

### Documentation
- **readme:** Add weights link and quiver download section

### Tests
- **quiver:** Accumulate cloud in streaming fill-in test

## [0.1.0] - 2026-10-22

### Added
- **quiver:** Add clear scene button to empty the 3d view
- **quiver:** Stream partial results into the scene while a run decodes
- **arrow-kit:** Stream partial predictions while decoding
- **quiver:** Add app icon
- **arrow-kit:** Recompute decoder keys and values per batch when they exceed a memory budget
- **arrow-kit:** Add parity fixtures captured from the reference pytorch model
- Add readme and app build script
- **quiver:** Add macos app to view arrow results in 3d
- **arrow-kit:** Add swift package running ARROW on MLX

### Fixed
- Replace demo images in README

### Documentation
- Add gif
