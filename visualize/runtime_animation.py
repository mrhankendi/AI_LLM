"""
Deepseek – Runtime Throughput Tracking (animated) + Terminal Panel (fixed layout)

- Target throughput step schedule (tokens/s)
- Observed throughput (synthetic, noisy, fast tracking)
- Runtime decisions: power cap (W) + batch size (step) on twin axis
- Vertical dashed lines at target-change times
- Moving cursor and time label
- Terminal panel at bottom showing commands + runtime reactions (NOT cut off)

Save as: runtime_animation_deepseek.py
Run:     python runtime_animation_deepseek.py

Optional:
- Save MP4: uncomment anim.save(...) and ensure ffmpeg is installed
- Save GIF: uncomment anim.save(...) and ensure pillow is installed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# ----------------------------
# 1) Values extracted/approximated from your slide
# ----------------------------
T_END = 360
t = np.arange(0, T_END + 1, 1)  # 0..360, 1 Hz

# Target change times (vertical dashed lines in the slide)
change_times = [70, 140, 220, 300]

# Target throughput (tokens/s): ~330 → 600 → 920 → 650 → 370
target_tps = np.piecewise(
    t,
    [t < 70, (t >= 70) & (t < 140), (t >= 140) & (t < 220), (t >= 220) & (t < 300), t >= 300],
    [330, 600, 920, 650, 370],
).astype(float)

# Power cap (W): ~150 → 200 → 300 → 150 → 150
power_cap = np.piecewise(
    t,
    [t < 70, (t >= 70) & (t < 140), (t >= 140) & (t < 220), (t >= 220) & (t < 300), t >= 300],
    [150, 200, 300, 150, 150],
).astype(float)

# Batch size (dashed, right axis): ~8 → 16 → 32 → 32 → 8
batch_size = np.piecewise(
    t,
    [t < 70, (t >= 70) & (t < 140), (t >= 140) & (t < 220), (t >= 220) & (t < 300), t >= 300],
    [8, 16, 32, 32, 8],
).astype(float)

# ----------------------------
# 2) Observed throughput (synthetic, noisy + fast tracking)
# ----------------------------
def simulate_observed_throughput(
    t,
    target,
    seed=7,
    tau_up=1.1,             # fast ramp up
    tau_down=1.8,           # fast ramp down but slightly slower than up
    noise_frac=0.03,        # noise as fraction of target (3%)
    jitter_frac=0.015,      # extra micro-jitter fraction (1.5%)
    overshoot_gain=0.06,    # overshoot on step-ups
    ring_amp=0.04,          # ringing amplitude fraction of step
    ring_tau=5.0,           # ringing decay (s)
    ring_freq=1.4,          # ringing frequency
    dip_prob=0.02,          # dips per sample (after warmup)
    dip_mag_frac=(0.05, 0.18),  # dip size as fraction of target
    dip_len=(2, 8),         # dip length in samples
    warmup_s=60,            # keep first 60s well-behaved
    clamp_low=0.92,         # never go below 92% of target
    clamp_high=1.12,        # never go above 112% of target
):
    """
    Realistic throughput telemetry with:
    - fast closed-loop tracking
    - noise scaled to operating point (target)
    - occasional dips after warmup
    - band-clamp around target to avoid "way below target" artifacts
    """
    rng = np.random.default_rng(seed)
    y = np.zeros_like(target, dtype=float)

    # Start essentially at target (slightly above)
    y[0] = float(target[0] * 1.01)

    last_target = float(target[0])
    step_size = 0.0

    # Ringing state
    ring = 0.0
    ring_phase = 0.0

    # Dip state
    dip_remaining = 0
    dip_amount = 0.0

    # dt support
    dt_arr = np.diff(t, prepend=t[0])
    dt_arr[0] = dt_arr[1] if len(dt_arr) > 1 else 1.0

    for i in range(1, len(t)):
        dt = float(dt_arr[i])
        tgt = float(target[i])

        # Step detection
        if tgt != last_target:
            step_size = tgt - last_target
            if step_size > 0:
                ring = ring_amp * step_size
                ring_phase = 0.0
            else:
                ring = 0.0
            last_target = tgt

        # Ringing (decaying sinusoid)
        ring_phase += 2.0 * np.pi * ring_freq * dt
        ring *= np.exp(-dt / ring_tau)
        ring_term = ring * np.sin(ring_phase)

        # Overshoot on upward step (decays quickly)
        overshoot_term = 0.0
        if step_size > 0:
            overshoot_term = overshoot_gain * step_size * np.exp(-dt / 2.5)

        # Tracking update (fast)
        tau = tau_up if (tgt + ring_term + overshoot_term) > y[i - 1] else tau_down
        alpha = 1.0 - np.exp(-dt / tau)
        desired = tgt + ring_term + overshoot_term
        y[i] = y[i - 1] + alpha * (desired - y[i - 1])

        # Scale noise to target magnitude (prevents low-target segments dipping too far)
        sigma = max(6.0, noise_frac * tgt)
        jitter = max(2.0, jitter_frac * tgt)

        # Dips only after warmup, and sized relative to target
        if t[i] > warmup_s:
            if dip_remaining <= 0 and rng.random() < dip_prob:
                dip_remaining = int(rng.integers(dip_len[0], dip_len[1] + 1))
                dip_amount = float(rng.uniform(dip_mag_frac[0], dip_mag_frac[1]) * tgt)

        if dip_remaining > 0:
            y[i] -= dip_amount * (0.7 + 0.3 * (dip_remaining / max(dip_len[1], 1)))
            dip_remaining -= 1

        # Add noise + jitter
        y[i] += rng.normal(0.0, sigma) + rng.normal(0.0, jitter)

        # Clamp around target to avoid "unphysical" long under-target artifacts
        lo = clamp_low * tgt
        hi = clamp_high * tgt
        y[i] = np.clip(y[i], lo, hi)

    # Also clip to overall plot bounds safety (matches your axis range)
    y = np.clip(y, 300, 1020)
    return y

observed_tps = simulate_observed_throughput(t, target_tps)

# ----------------------------
# 3) Figure setup (3 panels: top, bottom, terminal)
# ----------------------------
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.85], hspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax_term = fig.add_subplot(gs[2, 0])
ax2b = ax2.twinx()

fig.suptitle("Deepseek – Runtime Throughput Tracking", fontsize=18)

# Manual layout (prevents terminal cut-off on Windows/TkAgg)
fig.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.08, hspace=0.35)

# Top plot
ax1.set_title("Deepseek- Throughput Tracking")
ax1.set_ylabel("Tokens/s")
ax1.grid(True, alpha=0.3)

# Bottom plot
ax2.set_title("Deepseek- Runtime Decisions")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Power cap (W)")
ax2.grid(True, alpha=0.3)
ax2b.set_ylabel("Batch size")

# Limits (avoid autoscale jitter)
ax1.set_xlim(t.min(), t.max())
ax1.set_ylim(300, 1020)
ax2.set_ylim(140, 310)
ax2b.set_ylim(6, 34)

# Vertical target-change markers
for ct in change_times:
    ax1.axvline(ct, linestyle="--", linewidth=1)
    ax2.axvline(ct, linestyle="--", linewidth=1)

# Terminal styling
ax_term.set_facecolor("black")
ax_term.set_xticks([])
ax_term.set_yticks([])
for spine in ax_term.spines.values():
    spine.set_visible(False)

term_text = ax_term.text(
    0.01, 0.97, "",
    transform=ax_term.transAxes,
    va="top", ha="left",
    family="monospace",
    fontsize=9,
    linespacing=1.15,
    color="#00ff66",
)

TERM_MAX_LINES = 10
term_lines = deque(maxlen=TERM_MAX_LINES)

def term_print(line: str):
    term_lines.append(line)
    term_text.set_text("\n".join(term_lines))

# Map change times to "user command" strings (customize)
user_cmd = {
    70:  "set_target 600",
    140: "set_target 920",
    220: "set_target 650",
    300: "set_target 370",
}

def runtime_decision_str(idx: int) -> str:
    cap = int(power_cap[idx])
    bs = int(batch_size[idx])
    return f"controller: power_cap -> {cap}W | batch -> {bs}"

# ----------------------------
# 4) Artists
# ----------------------------
(line_target,) = ax1.plot([], [], drawstyle="steps-post", linewidth=2, label="Target throughput (tokens/s)")
(line_obs,) = ax1.plot([], [], linewidth=2, label="Observed throughput (tokens/s)")
ax1.legend(loc="upper right")

(line_cap,) = ax2.plot([], [], drawstyle="steps-post", linewidth=2, label="Chosen power cap (W)")
(line_bs,) = ax2b.plot([], [], drawstyle="steps-post", linewidth=2, linestyle="--", label="Chosen batch size")

ax2.legend(loc="upper right")
ax2b.legend(loc="upper right", bbox_to_anchor=(1.0, 0.85))

# Moving cursor lines (IMPORTANT: xdata must be a sequence [x,x])
cursor1 = ax1.axvline(t[0], color="k", alpha=0.25, linewidth=1)
cursor2 = ax2.axvline(t[0], color="k", alpha=0.25, linewidth=1)

# Time label
time_text = ax1.text(0.02, 0.92, "", transform=ax1.transAxes)

# Optional: annotate the cap/batch choices near the step segments
ax2.text(74, 205, "target_change\ncap=200, b=16", fontsize=9)
ax2.text(145, 303, "cap=300, b=32", fontsize=9)
ax2.text(224, 152, "target_change\ncap=150, b=32", fontsize=9)
ax2.text(304, 152, "target_change\ncap=150, b=8", fontsize=9)

# ----------------------------
# 5) Animation
# ----------------------------
N = len(t)
stride = 1
interval_ms = 30
frames = int(np.ceil(N / stride))

printed_changes = set()

def init():
    line_target.set_data([], [])
    line_obs.set_data([], [])
    line_cap.set_data([], [])
    line_bs.set_data([], [])

    cursor1.set_xdata([t[0], t[0]])
    cursor2.set_xdata([t[0], t[0]])
    time_text.set_text("")

    term_lines.clear()
    term_print("runtime@node0:~$ ./power_aware_runtime --model deepseek-moe")
    term_print("loading policy... OK")
    term_print(f"initial target = {int(target_tps[0])} tok/s")
    term_print(runtime_decision_str(0))
    term_print("--------------------------------------------------")
    term_print("ready: type `set_target <tok/s>` to change target")

    printed_changes.clear()

    return (line_target, line_obs, line_cap, line_bs, cursor1, cursor2, time_text, term_text)

def update(frame_idx):
    i = frame_idx * stride
    if i >= N:
        i = N - 1

    x = t[: i + 1]
    line_target.set_data(x, target_tps[: i + 1])
    line_obs.set_data(x, observed_tps[: i + 1])

    line_cap.set_data(x, power_cap[: i + 1])
    line_bs.set_data(x, batch_size[: i + 1])

    cursor1.set_xdata([t[i], t[i]])
    cursor2.set_xdata([t[i], t[i]])
    time_text.set_text(f"t = {t[i]} s")

    ti = int(t[i])

    # Print once at each change time
    if ti in user_cmd and ti not in printed_changes:
        printed_changes.add(ti)
        term_print(f"runtime@node0:~$ {user_cmd[ti]}")
        term_print(f"target updated -> {int(target_tps[i])} tok/s")
        term_print(runtime_decision_str(i))
        term_print("tracking...")

    # Periodic status line
    if ti % 10 == 0:
        term_print(
            f"status: t={ti:3d}s | obs={observed_tps[i]:7.1f} tok/s | "
            f"cap={int(power_cap[i])}W | b={int(batch_size[i])}"
        )

    return (line_target, line_obs, line_cap, line_bs, cursor1, cursor2, time_text, term_text)

# If Windows/TkAgg gives blit issues with twinx + terminal text, set blit=False.
anim = FuncAnimation(
    fig, update, frames=frames, init_func=init,
    interval=interval_ms, blit=True
)

plt.show()

# ----------------------------
# 6) Optional saving
# ----------------------------
# MP4 (requires ffmpeg installed and on PATH)
# anim.save("deepseek_runtime.mp4", fps=30, dpi=150)

# GIF (requires pillow)
anim.save("deepseek_runtime.gif", fps=20, dpi=120)
