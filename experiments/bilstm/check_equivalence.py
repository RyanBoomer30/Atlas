"""Correctness check: our BiLSTM vs nn.LSTM(batch_first=True, bidirectional=True).

Loaded with the same weights, both must give the same output (torch.allclose).
"""

import torch
from bilstm import BiLSTM


def compare(model, x, split_bias=False):
    ref = model.to_torch_lstm()
    if split_bias:
        # nn.LSTM's two biases only ever appear as a sum: move a random share of
        # bias_ih into bias_hh and the output must not change.
        with torch.no_grad():
            for suffix in ("l0", "l0_reverse"):
                b_ih, b_hh = getattr(ref, f"bias_ih_{suffix}"), getattr(ref, f"bias_hh_{suffix}")
                share = torch.randn_like(b_ih)
                b_ih -= share
                b_hh += share
    with torch.no_grad():
        _, (h_n, _) = ref(x)  # h_n: (2, batch, H) -> fwd final, bwd final
        expected = model.head(torch.cat([h_n[0], h_n[1]], dim=1))
        actual = model(x)
    return actual, expected


def run(model=None, dtype=torch.float32, seed=42):
    torch.manual_seed(seed)
    model = (model or BiLSTM(input_size=10, hidden_size=64)).to(dtype).eval()
    atol = 1e-6 if dtype == torch.float32 else 1e-12
    results = {}
    for name, (batch, seq_len) in {"short": (8, 1), "typical": (32, 12), "long": (4, 200)}.items():
        x = torch.randn(batch, seq_len, model.hparams.input_size, dtype=dtype)
        for split_bias in (False, True):
            actual, expected = compare(model, x, split_bias)
            ok = torch.allclose(actual, expected, atol=atol, rtol=1e-5)
            diff = (actual - expected).abs().max().item()
            results[(name, split_bias)] = (ok, diff)
            print(
                f"{dtype!s:14} seq={seq_len:<4} split_bias={split_bias!s:5}  "
                f"allclose={ok}  max|diff|={diff:.2e}"
            )
    return results


if __name__ == "__main__":
    all_ok = all(ok for d in (torch.float32, torch.float64) for ok, _ in run(dtype=d).values())
    print("PASS" if all_ok else "FAIL")
    raise SystemExit(0 if all_ok else 1)
