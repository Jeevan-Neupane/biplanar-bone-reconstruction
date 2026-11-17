"""Tests for the Swin window attention in models/swinX2S/swin_transformer.py (CPU only).

1. window_partition keeps whole pixels in each token, and window_reverse inverts it.
2. A SwinTransformerBlock equals an independent reference that loops over
   windows explicitly (cyclic shift, attention mask, residuals), for shifted and unshifted blocks,
   including padding (H, W not multiples of the window).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.swinX2S import swin_transformer as st  # noqa: E402


def test_partition():
    B, H, W, C, ws = 2, 14, 21, 5, (7, 7)
    h = torch.arange(H).view(1, H, 1, 1); w = torch.arange(W).view(1, 1, W, 1); c = torch.arange(C).view(1, 1, 1, C)
    b = torch.arange(B).view(B, 1, 1, 1)
    x = (b * 1e6 + h * 1000 + w * 10 + c).double()
    win = st.window_partition(x, ws)
    tokens = win.view(-1, ws[0] * ws[1], C)
    pix_ok = bool(((tokens - tokens[..., :1]) == torch.arange(C).double()).all())
    rt_ok = torch.equal(st.window_reverse(win, ws, H, W), x)
    print(f"1. each token = one pixel's channels: {pix_ok}; partition+reverse returns the input: {rt_ok}")
    return pix_ok and rt_ok


def reference_block(blk, x):
    """Standard Swin block written with explicit loops over windows (independent of reshapes)."""
    H, W = blk.input_resolution
    B, L, C = x.shape
    wh, ww = blk.window_size
    sh, sw = blk.shift_size
    shortcut = x
    y = blk.norm1(x).view(B, H, W, C)
    pb, pr = (wh - H % wh) % wh, (ww - W % ww) % ww
    y = torch.nn.functional.pad(y, (0, 0, 0, pr, 0, pb))
    Hp, Wp = H + pb, W + pr
    if sh > 0 or sw > 0:
        y = torch.roll(y, shifts=(-sh, -sw), dims=(1, 2))
    out = torch.zeros_like(y)
    for bi in range(B):
        widx = 0
        for i in range(0, Hp, wh):
            for j in range(0, Wp, ww):
                tok = y[bi, i:i + wh, j:j + ww, :].reshape(1, wh * ww, C)
                mask = None if blk.attn_mask is None else blk.attn_mask[widx:widx + 1]
                out[bi, i:i + wh, j:j + ww, :] = blk.attn(tok, mask=mask).reshape(wh, ww, C)
                widx += 1
    if sh > 0 or sw > 0:
        out = torch.roll(out, shifts=(sh, sw), dims=(1, 2))
    out = out[:, :H, :W, :].reshape(B, H * W, C)
    y = shortcut + out
    return y + blk.mlp(blk.norm2(y))


def test_block_vs_reference():
    torch.manual_seed(0)
    ok_all = True
    for res, shift in [((14, 21), (0, 0)), ((14, 21), (3, 3)), ((12, 17), (3, 3)), ((24, 32), (3, 3))]:
        blk = st.SwinTransformerBlock(dim=16, input_resolution=res, num_heads=2, window_size=(7, 7),
                                      shift_size=list(shift), drop_path=0.1).eval()
        x = torch.randn(2, res[0] * res[1], 16)
        with torch.no_grad():
            a, r = blk(x), reference_block(blk, x)
        ok = torch.allclose(a, r, atol=1e-5)
        ok_all &= ok
        print(f"2. block {res} shift {tuple(blk.shift_size)}: max|block - reference| = {float((a - r).abs().max()):.2e}  -> {ok}")
    return ok_all


if __name__ == "__main__":
    results = [test_partition(), test_block_vs_reference()]
    print("ALL PASSED" if all(results) else "SOME TESTS FAILED")
