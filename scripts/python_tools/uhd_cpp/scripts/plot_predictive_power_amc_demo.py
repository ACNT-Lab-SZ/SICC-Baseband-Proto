import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    with open(args.curves, newline="", encoding="utf-8-sig") as f:
        curves = list(csv.DictReader(f))
    with open(args.summary, newline="", encoding="utf-8-sig") as f:
        summary = list(csv.DictReader(f))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[PLOT] matplotlib unavailable: {exc}")
        return 0

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {
        "fixed_qpsk": "#2563EB",
        "fixed_16qam": "#DC2626",
        "predictive_amc": "#16A34A",
    }
    labels = {
        "fixed_qpsk": "固定 QPSK",
        "fixed_16qam": "固定 16QAM",
        "predictive_amc": "预测功率驱动 AMC",
    }

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    for case in dict.fromkeys(row["case"] for row in curves):
        g = [row for row in curves if row["case"] == case]
        x = [float(row["logical_frame"]) for row in g]
        y = [max(float(row["cum_fer"]), 1e-6) for row in g]
        ax.semilogy(x, y, linewidth=2.0, color=colors.get(case), label=labels.get(case, case))
    ax.axhline(1e-3, color="black", linestyle="--", linewidth=1.0, label="FER=1e-3")
    ax.set_xlabel("帧序号")
    ax.set_ylabel("累计 FER")
    ax.set_title("预测相对功率驱动自适应方案的可靠性对比")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "predictive_power_amc_fer.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    for case in dict.fromkeys(row["case"] for row in curves):
        g = [row for row in curves if row["case"] == case]
        ax.plot([float(row["logical_frame"]) for row in g],
                [float(row["cum_goodput_mbps"]) for row in g],
                linewidth=2.0,
                color=colors.get(case), label=labels.get(case, case))
    ax.set_xlabel("帧序号")
    ax.set_ylabel("累计有效吞吐率 (Mbps)")
    ax.set_title("预测相对功率驱动自适应方案的吞吐量对比")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "predictive_power_amc_goodput.png")

    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=150)
    case_names = [row["case"] for row in summary]
    names = [labels.get(x, x) for x in case_names]
    ax.bar(names, [float(row["final_goodput_mbps"]) for row in summary],
           color=[colors.get(x, "#64748B") for x in case_names])
    ax.set_ylabel("最终有效吞吐率 (Mbps)")
    ax.set_title("最终吞吐率对比")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.autofmt_xdate(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "predictive_power_amc_summary_goodput.png")

    print(f"[PLOT] wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
