# Lightning reference fixtures

These one-second MJPEG/AVI clips are public regression fixtures extracted from
the project owner's source footage with explicit permission. They are scaled
to 960×540, retain the original 100 fps cadence, and contain no audio.

`ground-truth.json` records the expected winning frame and minimum visible
channel measurements. Proposed winners were visually inspected before being
added. More positive channel shapes and negative examples such as cloud-only
illumination, camera motion, and exposure changes should be added over time.

The fixtures are test material, not representative production output. Their
license follows the repository license.
