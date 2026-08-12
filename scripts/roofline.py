import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PEAK_GFLOPS = 9030.0
PEAK_BANDWIDTH_GBS = 256.0
RIDGE = PEAK_GFLOPS / PEAK_BANDWIDTH_GBS  # 35.3

# (label, arithmetic intensity in flops/byte, measured GFLOP/s)
KERNELS = [
    ("v1 naive", 0.061, 290.0),
    ("v2 coalesced", 0.40, 1912.0),
]

intensity = np.logspace(-2, 4, 400)
ceiling = np.minimum(PEAK_GFLOPS, intensity * PEAK_BANDWIDTH_GBS)

figure, axes = plt.subplots(figsize=(8, 5.5))
axes.loglog(
    intensity,
    ceiling,
    color="black",
    linewidth=2,
    label="roofline"
)

axes.axvline(
    RIDGE,
    color="grey",
    linestyle=":",
    linewidth=1
)

axes.text(
    RIDGE * 1.15,
    PEAK_GFLOPS * 0.3,
    f"ridge {RIDGE:.1f} flops/byte",
    rotation=90,
    fontsize=9,
    color="grey"
)

for label, ai, measured in KERNELS:
    axes.plot(ai, measured, "o", markersize=8)
    axes.annotate(
        label,
        (ai, measured),
        textcoords="offset points",
        xytext=(8, -4),
        fontsize=9
    )

axes.set_xlabel("arithmetic intensity (flops per byte)")
axes.set_ylabel("attainable performance (GFLOP/s)")
axes.set_title("Roofline, RTX 5070, single precision")
axes.grid(True, which="both", alpha=0.3)
axes.legend(loc="lower right")

figure.tight_layout()
figure.savefig("results/roofline.png", dpi=150)
