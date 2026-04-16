import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# DeepSeek-style runtime control sim (feedforward batch + PID power trim)
# Top plot: Target + OBSERVED throughput (and optional model prediction dashed)
# Bottom plot: Power cap + Batch size
#
# Key points:
# - Model is optimistic; true plant is 10–15% worse (mismatch)
# - Batch is chosen by feedforward (model inversion), optionally held within each target segment
# - PID adjusts ONLY power cap incrementally (true convergence behavior)
# - Plant has lag + measurement delay to create oscillation and settling
# ----------------------------

rng = np.random.default_rng(4)

# ----------------------------
# Time base
# ----------------------------
T_END = 360
dt = 1.0
t = np.arange(0, T_END + 1, dt)

# ----------------------------
# Target schedule (tokens/s)
# ----------------------------
change_times = [0, 70, 140, 220, 300]
target_levels = [350, 600, 920, 650, 380]

target = np.zeros_like(t, dtype=float)
for i in range(len(change_times)):
    start = change_times[i]
    end = change_times[i + 1] if i + 1 < len(change_times) else (T_END + 1)
    target[(t >= start) & (t < end)] = target_levels[i]

# ----------------------------
# Knob bounds
# ----------------------------
B_MIN, B_MAX = 1, 32
P_MIN, P_MAX = 150, 300
P_STEP = 5

# Controller update cadence
CTRL_PERIOD = 2  # seconds

# ----------------------------
# Model: predicted throughput from knobs
# ----------------------------
def model_tput(batch, power_w):
    b = float(batch)
    p = float(power_w)
    batch_term = 1.0 - np.exp(-b / 10.0)
    power_term = 1.0 - np.exp(-(p - 120.0) / 80.0)
    return 260.0 + 850.0 * batch_term * power_term  # ~300..~1000 range

def pick_knobs_from_model(desired_tput):
    """
    Simple inversion via grid search.
    Returns (batch, power) that barely meets desired_tput under the model,
    preferring lower power then lower batch.
    """
    best = None
    for b in range(B_MIN, B_MAX + 1):
        for p in range(P_MIN, P_MAX + 1, P_STEP):
            pred = model_tput(b, p)
            if pred >= desired_tput:
                cand = (p, b, pred)
                if best is None or (cand[0] < best[0]) or (cand[0] == best[0] and cand[1] < best[1]):
                    best = cand
                break
    if best is None:
        return B_MAX, P_MAX
    return best[1], best[0]  # (batch, power)

# ----------------------------
# True plant mismatch: model overestimates by 10–15%
# ----------------------------
mismatch_scale = 1.0 - rng.uniform(0.10, 0.15)  # 0.85..0.90
mismatch_bias = -rng.uniform(0.0, 15.0)

def true_tput_steady(batch, power_w):
    return mismatch_scale * model_tput(batch, power_w) + mismatch_bias

# Plant dynamics + measurement delay
tau = 9.0            # seconds (bigger -> slower -> more overshoot)
meas_delay_s = 3     # seconds feedback delay
noise_sigma = 2.0    # measurement noise (tokens/s)
delay_steps = int(meas_delay_s / dt)

# ----------------------------
# PID (power-cap-only)
# ----------------------------
# Slightly aggressive to show ringing + settling; derivative damps
Kp = 0.18
Ki = 0.030
Kd = 0.10

integral = 0.0
prev_err = 0.0
I_CLAMP = 3000.0

# Rate limit power change per control update (W/update)
MAX_DP_PER_UPDATE = 15

# Map controller output to dp (W). Smaller denom => more aggressive power moves.
U_TO_W_SCALE = 7.0

# Anti-windup: if power saturates and controller wants to push further, stop integrating
ANTI_WINDUP = True

# Feedforward batch handling
HOLD_BATCH_BETWEEN_TARGET_CHANGES = True

# Optional: on target change, reset integral to avoid carryover windup
RESET_INTEGRAL_ON_TARGET_CHANGE = True

# ----------------------------
# Run simulation
# ----------------------------
batch = np.zeros_like(t, dtype=int)
power = np.zeros_like(t, dtype=float)

pred_tput = np.zeros_like(t, dtype=float)  # model prediction for current knobs
obs_tput  = np.zeros_like(t, dtype=float)  # observed (true plant output)

# Initialize
b_ff0, p_ff0 = pick_knobs_from_model(target[0])
batch[0] = b_ff0
power[0] = p_ff0
pred_tput[0] = model_tput(batch[0], power[0])
obs_tput[0] = true_tput_steady(batch[0], power[0])

obs_hist = [obs_tput[0]]
current_segment_batch = b_ff0
last_target = target[0]

ctrl_steps = int(CTRL_PERIOD / dt)

for k in range(1, len(t)):
    # default: hold previous knobs
    batch[k] = batch[k - 1]
    power[k] = power[k - 1]

    # plant evolves each second
    y_ss = true_tput_steady(batch[k], power[k])
    obs_tput[k] = obs_tput[k - 1] + (dt / tau) * (y_ss - obs_tput[k - 1]) + rng.normal(0, noise_sigma)
    obs_hist.append(obs_tput[k])

    # model prediction for plotting/reference
    pred_tput[k] = model_tput(batch[k], power[k])

    # control tick
    if (k % ctrl_steps) != 0:
        continue

    # delayed measurement
    k_meas = max(0, k - delay_steps)
    y_meas = obs_hist[k_meas]

    # detect target change
    target_changed = (target[k] != last_target)
    if target_changed:
        last_target = target[k]
        if RESET_INTEGRAL_ON_TARGET_CHANGE:
            integral = 0.0
            prev_err = 0.0

    # feedforward (batch + baseline power from model)
    b_ff, p_ff = pick_knobs_from_model(target[k])

    # choose batch (feedforward only)
    if HOLD_BATCH_BETWEEN_TARGET_CHANGES:
        if target_changed:
            current_segment_batch = b_ff
        b_use = current_segment_batch
    else:
        b_use = b_ff

    # PID on throughput error (observed)
    err = target[k] - y_meas

    # candidate integral update
    integral_candidate = np.clip(integral + err * CTRL_PERIOD, -I_CLAMP, I_CLAMP)

    deriv = (err - prev_err) / CTRL_PERIOD
    prev_err = err

    u = Kp * err + Ki * integral + Kd * deriv

    # convert to delta power
    dp = int(np.round(u / U_TO_W_SCALE))
    dp = int(np.clip(dp, -MAX_DP_PER_UPDATE, MAX_DP_PER_UPDATE))

    # incremental PID power update (true convergence behavior)
    p_unc = power[k] + dp
    p_new = float(np.clip(p_unc, P_MIN, P_MAX))
    p_new = P_STEP * round(p_new / P_STEP)

    # anti-windup: only accept integral update if we're not saturating "against" the control direction
    if ANTI_WINDUP:
        pushing_up = (dp > 0)
        pushing_down = (dp < 0)
        sat_high = (p_new >= P_MAX - 1e-9)
        sat_low = (p_new <= P_MIN + 1e-9)

        if (sat_high and pushing_up) or (sat_low and pushing_down):
            # don't integrate further into saturation
            pass
        else:
            integral = integral_candidate
    else:
        integral = integral_candidate

    # commit knobs
    batch[k] = int(np.clip(b_use, B_MIN, B_MAX))
    power[k] = p_new

    # update model prediction after decision (optional)
    pred_tput[k] = model_tput(batch[k], power[k])

# ----------------------------
# Plot (reference layout)
# ----------------------------
fig = plt.figure(figsize=(14, 7))

# Top: Target + Observed (and dashed model prediction for context)
ax1 = plt.subplot(2, 1, 1)
ax1.step(t, target, where="post", label="Target throughput (tokens/s)")
ax1.plot(t, obs_tput, label="Observed throughput (tokens/s)")
ax1.plot(t, pred_tput, linestyle="--", label="Predicted throughput (model)")

for ct in change_times[1:]:
    ax1.axvline(ct, linestyle="--", linewidth=1)

ax1.set_title("Deepseek – Throughput Tracking (Target vs Observed)")
ax1.set_ylabel("Tokens/s")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right")

# Bottom: Power cap + Batch size
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax2.step(t, power, where="post", label="Chosen power cap (W)")
ax2.set_ylabel("Power cap (W)")
ax2.grid(True, alpha=0.3)

for ct in change_times[1:]:
    ax2.axvline(ct, linestyle="--", linewidth=1)

ax2b = ax2.twinx()
ax2b.step(t, batch, where="post", linestyle="--", label="Chosen batch size")
ax2b.set_ylabel("Batch size")

ax2.set_title("Deepseek – Runtime Decisions (Feedforward batch, PID trims power only)")
ax2.set_xlabel("Time (s)")

# combined legend
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
ax2b.legend(h1 + h2, l1 + l2, loc="upper right")

plt.tight_layout()
plt.show()