# animate_native_sin.py Notes

This script renders GPU vendor comparisons for `native_sin(x)` using Matplotlib. It loads chunked float32 dumps (path suffix `_native_sin`) via `NativeSinReader`, then animates each vendor's current value in a shared scatter plot.

Key behaviours:
1. Frame generation uses `build_x_sequence` to sweep an initial range before focusing near the configured `--target-x`. For legacy runs you can still pass `--skip-multiple k` to target `k*pi`, but the default spotlight now sits around raw x ~ 32768 so large-argument behaviour is easier to inspect.
2. Visual effects include a fading intro title, highlight shading once the threshold is reached, and a trailing history rendered with decaying alpha. Alerts blink when `x` exceeds the current focus threshold label.
3. PNG export is optional. When `--png-dir` is supplied the script iterates frames manually, saving `png_prefix + index` images with the requested zero padding and DPI. Use `--skip-video` if you only want the PNG sequence. After frame export `init()` is called again so the Matplotlib animation writer starts from a clean state.
4. `animate_native_sin_multi.py` wraps the same pipeline and introduces `--multiple` (default 1). It multiplies the base focus value 32768 by `n` so you can jump straight to areas like 65536 without recalculating arguments.

Operational tips:
- The scripts assume `numpy` and `matplotlib` are installed. Run `python -m pip install numpy matplotlib` inside `2_????/` if imports fail.
- Dataset arguments default to sibling directories (`intelUHD770_win`, `gfx1036_win`, `RTX5090_win`). Adjust them when comparing other captures.
- Generating PNG frames keeps the figure open in memory; consider exporting in batches if you encounter memory pressure. The loop currently runs single-threaded.

Future tweaks should keep CLI flags PEP 8 aligned, avoid breaking existing defaults, and prefer vectorised math in new effects. Update this note whenever you add significant behaviours (new overlays, camera paths, etc.).
