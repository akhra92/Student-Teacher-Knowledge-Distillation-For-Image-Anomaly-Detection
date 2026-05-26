import torch

from stad.models import Student, Vgg16Teacher


def test_student_returns_thirteen_relu_outputs():
    student = Student(equal_size=False, use_bias=False, batch_norm=True)
    x = torch.randn(2, 3, 64, 64)
    feats = student(x)
    assert len(feats) == 13
    for f in feats:
        assert f.dim() == 4 and f.shape[0] == 2


def test_student_is_trainable():
    student = Student()
    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    assert n_trainable > 0


def test_vgg16_teacher_is_frozen():
    teacher = Vgg16Teacher(pretrained=False)
    n_trainable = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    assert n_trainable == 0


def test_teacher_returns_thirteen_relu_outputs():
    teacher = Vgg16Teacher(pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        feats = teacher(x)
    assert len(feats) == 13


def test_teacher_and_student_aligned_at_loss_layers():
    """At the 4 layers used by the loss, student/teacher feature maps should
    have identical shapes (config 'B' is engineered for this)."""
    teacher = Vgg16Teacher(pretrained=False)
    student = Student()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        ts = teacher(x)
        ss = student(x)
    assert len(ts) == len(ss) == 13
    for k in (3, 6, 9, 12):
        assert ts[k].shape == ss[k].shape, f"layer {k}: {ts[k].shape} vs {ss[k].shape}"


def test_teacher_and_student_full_size_align_everywhere():
    teacher = Vgg16Teacher(pretrained=False)
    student = Student(equal_size=True)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        ts = teacher(x)
        ss = student(x)
    for t, s in zip(ts, ss):
        assert t.shape == s.shape, f"{t.shape} vs {s.shape}"
