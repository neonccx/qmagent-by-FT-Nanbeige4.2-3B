import unittest
import pytest
torch = pytest.importorskip("torch")
import torch.nn.functional as F
from assistant_loss import assistant_loss

class AssistantLossTests(unittest.TestCase):
    def test_exact_masked_objective_and_gradient(self):
        torch.manual_seed(31)
        head = torch.nn.Linear(8, 23)
        hidden = torch.randn(2, 9, 8, requires_grad=True)
        labels = torch.randint(0, 23, (2, 9))
        labels[0, :6] = -100
        labels[1, :4] = -100
        labels[1, 7:] = -100  # right-padding must also remain masked
        reference = F.cross_entropy(head(hidden)[:, :-1].reshape(-1, 23), labels[:, 1:].reshape(-1))
        expected = torch.autograd.grad(reference, (hidden, head.weight, head.bias))
        actual = assistant_loss(hidden, labels, head)
        observed = torch.autograd.grad(actual, (hidden, head.weight, head.bias))
        torch.testing.assert_close(actual, reference)
        for a, b in zip(observed, expected):
            torch.testing.assert_close(a, b)

    def test_empty_target_rejected(self):
        with self.assertRaises(ValueError):
            assistant_loss(torch.zeros(1, 3, 8), torch.full((1, 3), -100), torch.nn.Linear(8, 23))
