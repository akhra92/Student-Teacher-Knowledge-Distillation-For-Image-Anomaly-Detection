import torch

from stad.models import TimmTeacher, ViTStudent

_MODEL = "vit_tiny_patch16_224"


def _build_pair():
    teacher = TimmTeacher(_MODEL, pretrained=False)
    student = ViTStudent(
        embed_dim=teacher.embed_dim,
        patch_size=teacher.patch_size,
        num_prefix_tokens=teacher.num_prefix_tokens,
        depth=teacher.depth,
        num_heads=teacher.num_heads,
        mlp_ratio=2.0,
    )
    return teacher, student


def test_teacher_is_frozen():
    teacher = TimmTeacher(_MODEL, pretrained=False)
    n_trainable = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    assert n_trainable == 0


def test_teacher_returns_one_feature_per_block():
    teacher = TimmTeacher(_MODEL, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feats = teacher(x)
    assert len(feats) == teacher.depth
    for f in feats:
        assert f.dim() == 3 and f.shape[0] == 2  # (B, N, D) tokens


def test_student_is_trainable():
    _, student = _build_pair()
    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    assert n_trainable > 0


def test_teacher_and_student_aligned_per_block():
    """The student's per-block token tensors must match the teacher's shapes so
    the distillation loss can compare them directly."""
    teacher, student = _build_pair()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ts = teacher(x)
        ss = student(x)
    assert len(ts) == len(ss) == teacher.depth
    for t, s in zip(ts, ss):
        assert t.shape == s.shape, f"{t.shape} vs {s.shape}"
