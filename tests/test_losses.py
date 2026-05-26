import torch

from stad.losses import DirectionOnlyLoss, MseDirectionLoss, MseLoss


def _make_features(n_layers=13, b=2, c=8, h=4, w=4):
    return [torch.randn(b, c, h, w, requires_grad=True) for _ in range(n_layers)]


def test_mse_direction_loss_is_zero_when_identical():
    feats = _make_features()
    same = [f.detach().clone() for f in feats]
    loss = MseDirectionLoss(lamda=0.5, layer_indices=[3, 6, 9, 12])(same, same)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_mse_direction_loss_is_positive_when_different():
    a = _make_features()
    b = _make_features()
    loss = MseDirectionLoss(lamda=0.5, layer_indices=[3, 6, 9, 12])(a, b)
    assert loss.item() > 0


def test_loss_is_differentiable():
    a = _make_features()
    b = [f.detach() for f in _make_features()]
    loss = MseDirectionLoss(lamda=0.5, layer_indices=[3, 6, 9, 12])(a, b)
    loss.backward()
    assert any(f.grad is not None for f in a)


def test_direction_only_uses_no_lambda():
    a = _make_features()
    b = _make_features()
    loss = DirectionOnlyLoss(layer_indices=[3, 6, 9, 12])(a, b)
    assert loss.item() >= 0


def test_mse_only_loss():
    a = _make_features()
    b = _make_features()
    loss = MseLoss(lamda=0.5, layer_indices=[3, 6, 9, 12])(a, b)
    assert loss.item() > 0
