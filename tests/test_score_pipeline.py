import torch

from stad.eval import score_batch
from stad.models import Student, Vgg16Teacher


def test_score_batch_returns_per_image_scalar():
    teacher = Vgg16Teacher(pretrained=False).eval()
    student = Student().eval()
    x = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        s = score_batch(student, teacher, x, layer_indices=[3, 6, 9, 12], lamda=0.05)
    assert s.shape == (4,)
    assert torch.isfinite(s).all()


def test_grayscale_input_is_handled():
    teacher = Vgg16Teacher(pretrained=False).eval()
    student = Student().eval()
    x = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        s = score_batch(student, teacher, x, layer_indices=[3, 6, 9, 12], lamda=0.05)
    assert s.shape == (2,)
