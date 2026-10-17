# Pixal3D UI (image -> textured GLB)

Gradio app around TencentARC/Pixal3D, meant to run on a Kaggle T4 (~29 GB RAM).
Triangle target (default 12,000), texture size, seed and background removal are
exposed in the UI; the result is a PNG-textured GLB with a download button.

- `app.py` UI + idle auto-shutdown (`IDLE_MINUTES`, default 30) + "Band karo" button
- `backend.py` real Pixal3D backend and a CPU mock (`PIXAL3D_MOCK=1`) for UI tests
- `patches.py` T4 memory fixes + runtime knobs applied to a Pixal3D checkout

Launch on Kaggle (from a machine that can reach Kaggle):

    export KAGGLE_API_TOKEN=...
    python scripts/push_pixal3d_ui_to_kaggle.py --kernel-id <user>/voidvideo-3d-ui

Then open the kernel's log and look for the `public URL` (gradio.live) line.
Local UI test without a GPU: `PIXAL3D_MOCK=1 PIXAL3D_OUT=/tmp/out python pixal3d_ui/app.py`.
