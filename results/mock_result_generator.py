import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# self made database
from crystaldatabase import CRYSTALS
from crystaldatabase import *


def generate_maker_fringe_csv(
    wl1_nm=1064.0,
    L_mm=4.4758,
    pos_range=(-30, 30),
    step=0.2,
    noise_level=0.05,
    offset_x = 0.8,
    offset_y = 0.5,
    out_csv="results/20250820_1412_fitting_test/original/in0_out0.csv",
):
    """Generate synthetic SHG Maker fringe data (position, ch1, ch2)"""

    wl1 = wl1_nm * 1e-6  # convert to mm
    wl2 = wl1 * 0.5
    L = L_mm

    # angle array
    theta_deg = np.arange(pos_range[0] - offset_x,
                        pos_range[1] + step - offset_x,
                        step)
    theta = np.deg2rad(theta_deg)

    # refractive indices (ordinary)
    crystal = SiO2()
    no_w = crystal.get_n(wl1_nm, polarization="o")
    no_2w = crystal.get_n(wl1_nm / 2, polarization="o")

    thetap_w = np.arcsin(np.sin(theta) / no_w)
    thetap_2w = np.arcsin(np.sin(theta) / no_2w)

    # parameters
    p = 1.0
    B = 1.0

    Psi = (np.pi * L / 2) * (4 / wl1) * (
        no_w * np.cos(thetap_w) - no_2w * np.cos(thetap_2w)
    )
    P_envelope = (
        (no_w * np.cos(thetap_w) + no_2w * np.cos(thetap_2w))
        * np.cos(theta) ** 4
        * np.cos(thetap_2w)
        * (no_w + 1) ** 3
        * (no_2w + 1) ** 3
        / (
            (no_w * np.cos(thetap_w) + np.cos(theta)) ** 3
            * (no_2w * np.cos(thetap_2w) + np.cos(theta)) ** 3
            * (no_w + no_2w)
        )
        * B
        * p
    )
    P_N = P_envelope * np.sin(Psi) ** 2

    # scale to ~3.0 max
    P_N = 3.0 * P_N / np.max(P_N)

    # add Gaussian noise
    noise = np.random.normal(scale=noise_level, size=len(P_N))
    ch2 = P_N + noise + offset_y
    ch1 = np.ones_like(ch2)

    # save CSV
    csv_path = Path(out_csv)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "ch1", "ch2"])
        for pos, c1, c2 in zip(theta_deg + offset_x, ch1, ch2):
            writer.writerow([f"{pos:.2f}", f"{c1:.3f}", f"{c2:.6f}"])

    # quick plot
    plt.plot(theta_deg + offset_x, ch2, label="ch2 (signal)")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend()
    plt.show()

    print(f"CSV saved to {csv_path.absolute()}")


if __name__ == "__main__":
    generate_maker_fringe_csv()
