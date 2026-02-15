"""PESNet 网络前向传播形状测试."""

import pytest


def test_pesnet_forward_shapes():
    """验证输入 (B,1,64,180,180) -> 输出 (B,1,1,180,180)."""
    torch = pytest.importorskip("torch")

    from pesnet.models.pesnet import PESNet, PESNetConfig

    cfg = PESNetConfig(R=64, U=180, V=180, stride_uv=3, upsample_uv=True, debug_shapes=True)
    model = PESNet(cfg)
    gs = torch.zeros((2, 1, 64, 180, 180))
    gt = torch.zeros((2, 1, 64, 180, 180))

    y, shapes = model(gs, gt, return_shapes=True)

    assert y.shape == (2, 1, 1, 180, 180)
    assert shapes["s5"][2] == 1   # RB5 后 R 维降到 1
    assert shapes["a5"][2] == 1
