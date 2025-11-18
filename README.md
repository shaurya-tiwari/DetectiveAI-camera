# DetectiveAI

DetectiveAI — Real-time intelligent surveillance & threat detection prototype.

## Overview
DetectiveAI analyzes video (CCTV/demo) in real time to detect:
- Weapons (guns, knives)
- Abandoned / stationary bags
- Crowd formation

Tech stack: YOLOv8 (detection), DeepSORT (tracking), Streamlit (dashboard), OpenCV (video handling).

## Repo structure
- app/       : production scripts (inference, utils)
- notebooks/ : Colab notebooks (experiments)
- models/    : model config files (do NOT commit large weights)
- videos/    : small demo videos (keep out of git)
- requirements.txt
- README.md

## Notes
- Keep large model weights in Google Drive (for Colab) or use Git LFS.
- Do not commit `models/*.pt` to GitHub.
