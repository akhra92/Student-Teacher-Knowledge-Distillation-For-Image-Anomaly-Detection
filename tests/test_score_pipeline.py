import torch

from stad.eval import score_batch
from stad.models import TimmTeacher, ViTStudent

_MODEL = "vit_tiny_patch16_224"
_LAYERS = [2, 5, 8, 11]


def _build_pair():
    teacher = TimmTeacher(_MODEL, pretrained=False).eval()
    student = ViTStudent(
        embed_dim=teacher.embed_dim,
        patch_size=teacher.patch_size,
        num_prefix_tokens=teacher.num_prefix_tokens,
        depth=teacher.depth,
        num_heads=teacher.num_heads,
        mlp_ratio=2.0,
    ).eval()
    return teacher, student


def test_score_batch_returns_per_image_scalar():
    teacher, student = _build_pair()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        s = score_batch(student, teacher, x, layer_indices=_LAYERS, lamda=0.05)
    assert s.shape == (4,)
    assert torch.isfinite(s).all()


def test_grayscale_input_is_handled():
    teacher, student = _build_pair()
    x = torch.randn(2, 1, 224, 224)
    with torch.no_grad():
        s = score_batch(student, teacher, x, layer_indices=_LAYERS, lamda=0.05)
    assert s.shape == (2,)
