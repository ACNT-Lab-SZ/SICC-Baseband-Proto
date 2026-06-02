from pathlib import Path

import torch


class SmokeDetectionModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        det = x.new_zeros((1, 6))
        det[0, 0] = 8.0
        det[0, 1] = 8.0
        det[0, 2] = 56.0
        det[0, 3] = 40.0
        det[0, 4] = 0.90
        det[0, 5] = 0.0
        return det.repeat(batch, 1, 1)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "models" / "libtorch_smoke_detection.ts"
    out.parent.mkdir(parents=True, exist_ok=True)
    model = SmokeDetectionModel().eval()
    example = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    traced = torch.jit.trace(model, example)
    traced.save(str(out))
    print(out)


if __name__ == "__main__":
    main()
