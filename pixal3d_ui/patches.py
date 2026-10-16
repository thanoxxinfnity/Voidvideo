"""Source patches applied to a fresh TencentARC/Pixal3D checkout so it runs on a
Kaggle T4 (14.5 GB) and exposes the mesh budget / texture size as runtime knobs.

Every edit is anchored on an exact string and fails loudly if upstream moved,
because a silent no-op costs an hour of GPU time before it surfaces.
"""
import os

SHIM = '''from PIL import Image


class BiRefNet:
    """Chomugiri stand-in for Pixal3D background remover.

    The real one loads briaai/RMBG-2.0, which is gated on HuggingFace, and it
    is constructed *eagerly* in Pipeline.from_pretrained, before any image is
    looked at. So cutting the image out beforehand is not enough on its own:
    the run dies loading a model it was never going to need.

    This satisfies the construction and nothing else. Every image that reaches
    the pipeline already has an alpha channel, so preprocess_image takes the
    has_alpha branch and never calls this.
    """

    def __init__(self, model_name: str = ""):
        self.model = None
        self.model_name = model_name

    def to(self, *args, **kwargs):
        # The pipeline moves it to the GPU whether or not it is ever used.
        return self

    def __call__(self, image):
        raise RuntimeError(
            "The background remover was called, which means an image arrived "
            "without an alpha channel. Chomugiri cuts them out before sending."
        )
'''

SDPA_SHIM = '''import torch
from torch.nn.functional import scaled_dot_product_attention as _torch_sdpa

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older torch
    SDPBackend = sdpa_kernel = None

# Biggest score matrix the math backend may hold at once, in bytes.
BUDGET = 1 << 30


def chunked_sdpa(q, k, v):
    """Torch SDPA that cannot allocate an L-by-L score matrix on a T4.

    Pixal3D asks for full attention over every sparse token. At resolution
    1024 the HR shape stage reaches roughly twenty-two thousand of them, and
    torch's *math* backend materialises the whole [1, H, Lq, Lkv] matrix:
    7.17 GiB in a single allocation, on a card with 14.56 GiB and the model
    already resident. That is the out-of-memory error, and it arrives an hour
    in, after the weights have downloaded and three stages have run.

    The memory-efficient CUTLASS kernel never builds that matrix and it does
    support sm_75, so ask for it by name instead of hoping torch picks it.

    When it is unavailable for these shapes torch raises rather than quietly
    falling back, so the query is split instead. Attention over a slice of Q
    equals attention over the whole of Q for those rows -- softmax is taken
    across the key axis, independently per query row -- so the slices are
    concatenated, not combined. Same numbers, bounded peak.
    """
    if sdpa_kernel is not None:
        try:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                return _torch_sdpa(q, k, v)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            pass

    heads, lq, lkv = q.shape[1], q.shape[-2], k.shape[-2]
    per_row = max(1, heads * lkv * q.element_size())
    rows = max(1, min(lq, BUDGET // per_row))
    if rows >= lq:
        return _torch_sdpa(q, k, v)
    return torch.cat(
        [_torch_sdpa(q[:, :, i:i + rows], k, v) for i in range(0, lq, rows)],
        dim=-2,
    )
'''

MEMO_SHIM = '''_CHOMUGIRI_CACHE = {}


def init_pipeline(model_path=MODEL_PATH, device="cuda", low_vram=False):
    """Build the pipeline once per process instead of once per call.

    Measured on a real run: 230 minutes end to end, of which 28 were the four
    sampling stages and the GLB extraction. The rest went into this function --
    the Pixal3D checkpoints, four DINOv3 ViT-L feature extractors, the NAF
    upsampler and MoGe, downloaded and constructed from nothing.

    Nothing upstream caches it, so a server that answers two requests pays that
    twice and is no faster than two separate runs. This is what makes holding
    the process open worth anything.
    """
    key = ("pipeline", model_path, device, low_vram)
    if key not in _CHOMUGIRI_CACHE:
        _CHOMUGIRI_CACHE[key] = _chomugiri_init_pipeline(model_path, device, low_vram)
    return _CHOMUGIRI_CACHE[key]


def load_moge_model(device="cuda", model_name=MOGE_MODEL_NAME):
    """Same, for the depth pass.

    run_inference moves MoGe to the CPU and drops its reference once the camera
    is estimated. That frees the VRAM, which is the point, and it does not
    destroy the weights -- the cache still holds them, and the original moves
    them back to the GPU on the next call.
    """
    key = ("moge", model_name)
    if key not in _CHOMUGIRI_CACHE:
        _CHOMUGIRI_CACHE[key] = _chomugiri_load_moge_model(device, model_name)
        return _CHOMUGIRI_CACHE[key]
    return _CHOMUGIRI_CACHE[key].to(device)
'''

# Anchors in upstream inference.py.
_NAF_BIG = '"naf_target_size": 1024,'
_GLB_HEAVY = "decimation_target=1000000, texture_size=4096,"
_GLB_EXPORT = "glb.export(output_path, extension_webp=True)"
_SDPA_IMPORT = "from torch.nn.functional import scaled_dot_product_attention as _sdpa"
_LOADERS = (
    'init_pipeline(model_path=MODEL_PATH, device="cuda", low_vram=False)',
    'load_moge_model(device="cuda", model_name=MOGE_MODEL_NAME)',
)

# Module-level knobs the UI sets before each call (read at call time).
KNOBS = "TARGET_FACES = 12000\nTEXTURE_SIZE = 1024\n"


class PatchError(RuntimeError):
    pass


def _once(text, needle, what):
    n = text.count(needle)
    if n != 1:
        raise PatchError("%s moved upstream (found %d matches)" % (what, n))


def patch_inference(inf):
    """Pure string -> string so it can be tested without a GPU."""
    _once(inf, _NAF_BIG, "NAF target size")
    inf = inf.replace(_NAF_BIG, '"naf_target_size": 512,')  # T4 OOM fix
    _once(inf, _GLB_HEAVY, "GLB extraction settings")
    inf = inf.replace(_GLB_HEAVY, "decimation_target=TARGET_FACES, texture_size=TEXTURE_SIZE,")
    # Plain PNG textures: opens in Blender/Unity/Godot/Windows viewer, not only webp-aware ones.
    _once(inf, _GLB_EXPORT, "GLB export call")
    inf = inf.replace(_GLB_EXPORT, "glb.export(output_path, extension_webp=False)")
    for name in _LOADERS:
        head = "def " + name + ":"
        _once(inf, head, name.split("(")[0])
        inf = inf.replace(head, "def _chomugiri_" + name + ":")
    inf = inf.replace("def _chomugiri_load_moge_model", MEMO_SHIM + "\n\ndef _chomugiri_load_moge_model", 1)
    # Constants must exist before run_inference executes; put them next to MODEL_PATH.
    _once(inf, 'MODEL_PATH = "TencentARC/Pixal3D"', "MODEL_PATH constant")
    inf = inf.replace('MODEL_PATH = "TencentARC/Pixal3D"', 'MODEL_PATH = "TencentARC/Pixal3D"\n' + KNOBS, 1)
    return inf


def apply_repo_patches(repo):
    """Patch a Pixal3D checkout in place."""
    with open(os.path.join(repo, "pixal3d/pipelines/rembg/__init__.py"), "w") as fh:
        fh.write(SHIM)
    with open(os.path.join(repo, "chomugiri_sdpa.py"), "w") as fh:
        fh.write(SDPA_SHIM)
    attn = os.path.join(repo, "pixal3d/modules/sparse/attention/full_attn.py")
    with open(attn) as fh:
        body = fh.read()
    _once(body, _SDPA_IMPORT, "sdpa import")
    with open(attn, "w") as fh:
        fh.write(body.replace(_SDPA_IMPORT, "from chomugiri_sdpa import chunked_sdpa as _sdpa"))
    path = os.path.join(repo, "inference.py")
    with open(path) as fh:
        inf = fh.read()
    with open(path, "w") as fh:
        fh.write(patch_inference(inf))
