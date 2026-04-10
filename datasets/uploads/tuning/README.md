# Session Tuning

This folder holds clip-specific tuning files for uploaded sessions that need help beyond the default segmentation stack.

Supported fields:

- `replace_auto_shots`
  When `true`, manual seeds replace the auto-segmented shot windows.
- `manual_shots`
  List of shot windows with:
  - `shot_id`
  - `shot_start_frame`
  - `set_point_frame`
  - `release_frame`
  - `shot_end_frame`
  - optional `apex_frame`

These files are intended for weak sessions where the teacher signals are usable but the temporal segmentation still needs explicit help.

