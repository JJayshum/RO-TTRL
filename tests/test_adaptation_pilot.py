import torch

from relational_orbit_ttrl.adaptation_pilot import sequence_log_probs


def test_sequence_log_probs_ignore_prompt_tokens():
    logits = torch.zeros((1, 4, 5))
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 3, 4]])
    sequence, tokens, mask = sequence_log_probs(logits, ids, labels)
    assert mask.tolist() == [[False, True, True]]
    assert torch.isclose(sequence[0], -2 * torch.log(torch.tensor(5.0)))
