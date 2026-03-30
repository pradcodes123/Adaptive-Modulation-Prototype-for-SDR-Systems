"""
Adaptive Modulation Communication System Simulator
====================================================
Demonstrates a transmitter-receiver pipeline with:
- Bit generation
- Adaptive modulation selection (BPSK, QPSK, 16-QAM)
- Packet / header creation
- AWGN channel model
- Receiver with dynamic demodulation
- BER computation
- Visualizations (constellation diagrams, BER vs SNR, switching behaviour)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend → saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1.  MODULATION SCHEMES
# ─────────────────────────────────────────────────────────────

class BPSK:
    """Binary Phase Shift Keying  –  1 bit per symbol."""
    name   = "BPSK"
    mod_id = 0
    bps    = 1                         # bits per symbol

    @staticmethod
    def modulate(bits: np.ndarray) -> np.ndarray:
        """Map bits {0,1} → complex symbols {-1, +1}."""
        return (2 * bits - 1).astype(complex)

    @staticmethod
    def demodulate(symbols: np.ndarray) -> np.ndarray:
        """Hard decision: real part > 0 → 1, else → 0."""
        return (symbols.real > 0).astype(int)


class QPSK:
    """Quadrature Phase Shift Keying  –  2 bits per symbol."""
    name   = "QPSK"
    mod_id = 1
    bps    = 2

    # Gray-coded constellation: 00→(+1+j), 01→(-1+j), 11→(-1-j), 10→(+1-j)
    _MAP = {(0, 0):  1+1j,
            (0, 1): -1+1j,
            (1, 1): -1-1j,
            (1, 0):  1-1j}
    _MAP_INV = {v: k for k, v in _MAP.items()}

    @classmethod
    def modulate(cls, bits: np.ndarray) -> np.ndarray:
        """Pack bits in pairs and map to QPSK constellation."""
        bits = bits[: len(bits) - len(bits) % 2]   # ensure even length
        pairs = bits.reshape(-1, 2)
        symbols = np.array([cls._MAP[tuple(p)] for p in pairs])
        return symbols / np.sqrt(2)                  # normalise power

    @classmethod
    def demodulate(cls, symbols: np.ndarray) -> np.ndarray:
        """Nearest-neighbour decision in the normalised constellation."""
        symbols_scaled = symbols * np.sqrt(2)
        recovered = []
        for s in symbols_scaled:
            # closest point in I/Q quadrants
            i_bit = 0 if s.real >= 0 else 1
            q_bit = 0 if s.imag >= 0 else 0
            # re-derive from quadrant
            real_d = 1 if s.real >= 0 else -1
            imag_d = 1 if s.imag >= 0 else -1
            closest = real_d + 1j * imag_d
            bits_out = cls._MAP_INV.get(closest, (0, 0))
            recovered.extend(bits_out)
        return np.array(recovered, dtype=int)


class QAM16:
    """16-QAM  –  4 bits per symbol (Gray-coded)."""
    name   = "16-QAM"
    mod_id = 2
    bps    = 4

    # Gray-coded 4-level PAM mapping for I and Q axes independently
    # 2-bit Gray code → amplitude level
    _GRAY2AMP = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
    _AMP2GRAY = {v: k for k, v in _GRAY2AMP.items()}
    _SCALE    = 1 / np.sqrt(10)         # normalise average power to 1

    @classmethod
    def modulate(cls, bits: np.ndarray) -> np.ndarray:
        """Pack bits in groups of 4 and map to 16-QAM symbols."""
        bits = bits[: len(bits) - len(bits) % 4]
        groups = bits.reshape(-1, 4)
        symbols = []
        for g in groups:
            i_amp = cls._GRAY2AMP[tuple(g[:2])]
            q_amp = cls._GRAY2AMP[tuple(g[2:])]
            symbols.append(i_amp + 1j * q_amp)
        return np.array(symbols) * cls._SCALE

    @classmethod
    def demodulate(cls, symbols: np.ndarray) -> np.ndarray:
        """Slice each I/Q component to nearest PAM-4 level."""
        levels   = np.array([-3, -1, 1, 3])
        recovered = []
        for s in symbols:
            s_unscaled = s / cls._SCALE
            i_level = levels[np.argmin(np.abs(s_unscaled.real - levels))]
            q_level = levels[np.argmin(np.abs(s_unscaled.imag - levels))]
            i_bits  = cls._AMP2GRAY[i_level]
            q_bits  = cls._AMP2GRAY[q_level]
            recovered.extend(i_bits + q_bits)
        return np.array(recovered, dtype=int)


# ─────────────────────────────────────────────────────────────
# 2.  ADAPTIVE MODULATION CONTROLLER
# ─────────────────────────────────────────────────────────────

# SNR thresholds (dB) for upward and downward switching
# Hysteresis: upgrade at THRESHOLD, downgrade 2 dB below
MODULATION_SCHEMES = [BPSK, QPSK, QAM16]   # ordered lowest → highest spectral efficiency

SNR_THRESHOLDS_UP   = [0.0,  7.0, 13.0]    # switch UP  to scheme[i] above this SNR
SNR_THRESHOLDS_DOWN = [0.0,  5.0, 11.0]    # switch DOWN from scheme[i] below this SNR


class AdaptiveController:
    """
    Maintains the current modulation index and decides when to switch.
    A simple exponential moving average (EMA) smooths the SNR estimate
    to prevent ping-pong switching (hysteresis is also encoded in the
    UP/DOWN threshold gap).
    """
    def __init__(self, alpha: float = 0.3):
        self.current_idx = 0            # start with BPSK (most robust)
        self.alpha       = alpha        # EMA smoothing factor  (0 < α ≤ 1)
        self._ema_snr    = None         # smoothed SNR

    @property
    def current_scheme(self):
        return MODULATION_SCHEMES[self.current_idx]

    def update(self, snr_db: float) -> object:
        """
        Accept a new raw SNR measurement, smooth it, and decide whether
        to switch modulation.  Returns the (possibly new) scheme object.
        """
        # --- EMA smoothing -------------------------------------------------
        if self._ema_snr is None:
            self._ema_snr = snr_db
        else:
            self._ema_snr = self.alpha * snr_db + (1 - self.alpha) * self._ema_snr

        snr = self._ema_snr

        # --- Upgrade check (try moving to a higher-order scheme) -----------
        next_idx = self.current_idx + 1
        if next_idx < len(MODULATION_SCHEMES):
            if snr >= SNR_THRESHOLDS_UP[next_idx]:
                self.current_idx = next_idx

        # --- Downgrade check (protect link quality) ------------------------
        if self.current_idx > 0:
            if snr < SNR_THRESHOLDS_DOWN[self.current_idx]:
                self.current_idx -= 1

        return self.current_scheme


# ─────────────────────────────────────────────────────────────
# 3.  PACKET STRUCTURE
# ─────────────────────────────────────────────────────────────

HEADER_BITS = 4       # enough to encode mod_id (0, 1, 2) in 2 bits + 2 parity


def build_packet(payload_bits: np.ndarray, mod_id: int) -> np.ndarray:
    """
    Packet layout (bits):
      [ mod_id_bit1 | mod_id_bit0 | parity | parity | payload … ]
    The two parity bits are the XOR of the mod_id bits (simple error detect).
    """
    id_bits = np.array([(mod_id >> 1) & 1, mod_id & 1], dtype=int)
    parity  = np.array([id_bits[0] ^ id_bits[1], id_bits[0] | id_bits[1]], dtype=int)
    header  = np.concatenate([id_bits, parity])
    return np.concatenate([header, payload_bits])


def parse_packet(packet_bits: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Extract mod_id from header and return (mod_id, payload_bits).
    If header parity fails, fall back to BPSK (mod_id = 0).
    """
    header  = packet_bits[:HEADER_BITS]
    payload = packet_bits[HEADER_BITS:]
    mod_id  = (int(header[0]) << 1) | int(header[1])
    # clamp to valid range
    mod_id  = max(0, min(mod_id, len(MODULATION_SCHEMES) - 1))
    return mod_id, payload


# ─────────────────────────────────────────────────────────────
# 4.  CHANNEL MODEL  (AWGN)
# ─────────────────────────────────────────────────────────────

def add_awgn(symbols: np.ndarray, snr_db: float, bps: int) -> np.ndarray:
    """
    Add complex AWGN noise to a symbol stream.

    Es/N0 = SNR_db  (per symbol)  →  σ² = 1 / (2 * Es/N0_linear)
    Symbols are assumed to have unit average energy after normalisation.
    """
    snr_linear = 10 ** (snr_db / 10)
    # Convert symbol SNR to bit SNR for noise calculation
    eb_n0 = snr_linear / bps
    noise_var = 1 / (2 * eb_n0 * bps)
    noise = (np.random.randn(*symbols.shape) +
             1j * np.random.randn(*symbols.shape)) * np.sqrt(noise_var)
    return symbols + noise


# ─────────────────────────────────────────────────────────────
# 5.  TRANSMITTER & RECEIVER PIPELINE
# ─────────────────────────────────────────────────────────────

def transmit_packet(bits: np.ndarray, scheme, snr_db: float) -> np.ndarray:
    """
    Full transmitter pipeline:
      1. Build packet (header + payload)
      2. Pad so length is divisible by bps
      3. Modulate
      4. Pass through AWGN channel
    Returns received *symbols* (complex array).
    """
    packet = build_packet(bits, scheme.mod_id)

    # Pad to multiple of bps
    remainder = len(packet) % scheme.bps
    if remainder:
        packet = np.concatenate([packet, np.zeros(scheme.bps - remainder, dtype=int)])

    tx_symbols  = scheme.modulate(packet)
    rx_symbols  = add_awgn(tx_symbols, snr_db, scheme.bps)
    return rx_symbols, len(packet)


def receive_packet(rx_symbols: np.ndarray, n_bits: int) -> tuple[int, np.ndarray]:
    """
    Full receiver pipeline:
      1. Demodulate using BPSK (the header is always BPSK-like in position)
         Actually: we first try BPSK to read header, then re-demodulate payload
         with the correct scheme.
      NOTE: In this simulation the entire packet (including header) is modulated
            with the selected scheme. The receiver knows the scheme from the header,
            so we demodulate everything with BPSK first, extract the header to find
            the true scheme, then re-demodulate with that scheme.
      2. Extract mod_id from header
      3. Demodulate full packet with correct scheme
      4. Return (mod_id, recovered_payload_bits)
    """
    # Step A: decode with BPSK to read header (coarse, enough for 4-bit header)
    # For simplicity we just look at real part of first 4 symbols (BPSK decision)
    header_bits_raw = (rx_symbols[:HEADER_BITS].real > 0).astype(int)
    mod_id_raw = (int(header_bits_raw[0]) << 1) | int(header_bits_raw[1])
    mod_id     = max(0, min(mod_id_raw, len(MODULATION_SCHEMES) - 1))
    scheme     = MODULATION_SCHEMES[mod_id]

    # Step B: demodulate entire packet with the identified scheme
    recovered_bits = scheme.demodulate(rx_symbols)[:n_bits]

    # Step C: parse packet to extract payload
    _, payload = parse_packet(recovered_bits)
    return mod_id, payload


# ─────────────────────────────────────────────────────────────
# 6.  BER COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_ber(original: np.ndarray, recovered: np.ndarray) -> float:
    """Bit Error Rate = number of bit errors / total bits compared."""
    n = min(len(original), len(recovered))
    if n == 0:
        return 1.0
    return float(np.sum(original[:n] != recovered[:n])) / n


# ─────────────────────────────────────────────────────────────
# 7.  SIMULATION RUNNER
# ─────────────────────────────────────────────────────────────

def run_ber_vs_snr_simulation(snr_range_db, n_bits_per_packet=512, n_packets=20):
    """
    Sweep SNR values, transmit packets, and record BER per modulation scheme.
    Returns dict: { scheme_name → (snr_list, ber_list) }
    """
    results = {s.name: {"snr": [], "ber": []} for s in MODULATION_SCHEMES}

    for snr in snr_range_db:
        for scheme in MODULATION_SCHEMES:
            bers = []
            for _ in range(n_packets):
                bits       = np.random.randint(0, 2, n_bits_per_packet)
                rx_sym, nb = transmit_packet(bits, scheme, snr)
                _, payload = receive_packet(rx_sym, nb)
                ber        = compute_ber(bits, payload)
                bers.append(ber)
            results[scheme.name]["snr"].append(snr)
            results[scheme.name]["ber"].append(np.mean(bers))

    return results


def run_adaptive_simulation(snr_values, n_bits_per_packet=256):
    """
    Simulate a series of transmissions where the SNR varies over time.
    The adaptive controller picks the modulation scheme for each packet.
    Returns lists of (snr, scheme_name, ber) for each packet.
    """
    controller = AdaptiveController(alpha=0.4)
    log = []   # (snr_db, scheme_name, ber)

    for snr in snr_values:
        scheme = controller.update(snr)
        bits   = np.random.randint(0, 2, n_bits_per_packet)
        rx_sym, nb = transmit_packet(bits, scheme, snr)
        _, payload = receive_packet(rx_sym, nb)
        ber    = compute_ber(bits, payload)

        print(f"  SNR={snr:5.1f} dB  →  {scheme.name:8s}  |  BER={ber:.4f}")
        log.append((snr, scheme.name, ber))

    return log


# ─────────────────────────────────────────────────────────────
# 8.  CONSTELLATION SAMPLE GENERATOR
# ─────────────────────────────────────────────────────────────

def get_constellation_samples(scheme, snr_db=25.0, n_symbols=800):
    """Generate noisy constellation samples for plotting."""
    bits   = np.random.randint(0, 2, n_symbols * scheme.bps)
    bits   = bits[: len(bits) - len(bits) % scheme.bps]
    clean  = scheme.modulate(bits)
    noisy  = add_awgn(clean, snr_db, scheme.bps)
    return clean, noisy


# ─────────────────────────────────────────────────────────────
# 9.  VISUALISATION
# ─────────────────────────────────────────────────────────────

COLORS = {
    "BPSK":   "#4C72B0",
    "QPSK":   "#DD8452",
    "16-QAM": "#55A868",
}
SCHEME_IDX = {"BPSK": 0, "QPSK": 1, "16-QAM": 2}

def plot_constellations(snr_db=20.0, save_path="constellation_diagrams.png"):
    """Plot clean + noisy constellation diagrams for all three schemes."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.patch.set_facecolor("#0F1117")
    fig.suptitle("Constellation Diagrams", fontsize=16, color="white",
                 fontweight="bold", y=0.98)

    row_labels = ["Clean", f"Noisy  (SNR = {snr_db} dB)"]
    for col, scheme in enumerate(MODULATION_SCHEMES):
        clean, noisy = get_constellation_samples(scheme, snr_db)
        color = COLORS[scheme.name]

        for row, (syms, label) in enumerate([(clean, "Clean"), (noisy, "Noisy")]):
            ax = axes[row][col]
            ax.set_facecolor("#1A1D27")
            ax.scatter(syms.real, syms.imag, s=10, alpha=0.5, color=color,
                       linewidths=0)
            # ideal constellation points (clean)
            ax.scatter(clean.real, clean.imag, s=80, color="white",
                       edgecolors=color, linewidths=1.5, zorder=5)
            ax.axhline(0, color="#555", linewidth=0.5)
            ax.axvline(0, color="#555", linewidth=0.5)
            ax.set_title(f"{scheme.name}  —  {label}", color="white", fontsize=11)
            ax.tick_params(colors="gray", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            ax.set_xlabel("In-phase (I)", color="gray", fontsize=8)
            ax.set_ylabel("Quadrature (Q)", color="gray", fontsize=8)

        # Row label on first column
        if col == 0:
            axes[0][0].text(-0.25, 0.5, row_labels[0], transform=axes[0][0].transAxes,
                            color="white", va="center", rotation=90, fontsize=9)
            axes[1][0].text(-0.25, 0.5, row_labels[1], transform=axes[1][0].transAxes,
                            color="white", va="center", rotation=90, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓  Saved: {save_path}")


def plot_ber_vs_snr(ber_results, save_path="ber_vs_snr.png"):
    """
    BER vs SNR curves for each modulation scheme on a log-linear scale.
    Overlays the SNR switching thresholds.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D27")
    ax.set_title("BER vs SNR  —  Per Modulation Scheme", color="white",
                 fontsize=14, fontweight="bold")

    for scheme_name, data in ber_results.items():
        snr_arr = np.array(data["snr"])
        ber_arr = np.array(data["ber"])
        # clip zeros for log scale
        ber_arr = np.clip(ber_arr, 1e-6, 1.0)
        color = COLORS[scheme_name]
        ax.semilogy(snr_arr, ber_arr, "-o", color=color, label=scheme_name,
                    linewidth=2, markersize=5)

    # Switch thresholds
    for i, (t_up, t_down) in enumerate(zip(SNR_THRESHOLDS_UP[1:],
                                            SNR_THRESHOLDS_DOWN[1:]), start=1):
        ax.axvline(t_up, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(t_up + 0.2, 0.3, f"↑{MODULATION_SCHEMES[i].name}", color="gray",
                fontsize=8, rotation=90, va="top")

    ax.set_xlabel("SNR (dB)", color="white", fontsize=12)
    ax.set_ylabel("Bit Error Rate (BER)", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1A1D27", labelcolor="white", edgecolor="#555")
    ax.grid(True, which="both", color="#333", linestyle="--", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.set_ylim(1e-6, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓  Saved: {save_path}")


def plot_adaptive_behaviour(adaptive_log, save_path="adaptive_switching.png"):
    """
    Two-panel figure:
      Top:    SNR over time with colour-coded modulation regions
      Bottom: Modulation index over time (staircase) + BER
    """
    snr_vals     = [r[0] for r in adaptive_log]
    scheme_names = [r[1] for r in adaptive_log]
    ber_vals     = [r[2] for r in adaptive_log]
    scheme_idx   = [SCHEME_IDX[n] for n in scheme_names]
    packets      = np.arange(len(adaptive_log))

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor("#0F1117")
    gs = gridspec.GridSpec(3, 1, hspace=0.45, figure=fig)

    # ── Panel 1: SNR timeline ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#1A1D27")
    ax1.plot(packets, snr_vals, color="#8BE3FF", linewidth=2, label="SNR (dB)")
    # shade regions by active scheme
    prev_scheme = scheme_names[0]
    seg_start   = 0
    for i, name in enumerate(scheme_names):
        if name != prev_scheme or i == len(scheme_names) - 1:
            ax1.axvspan(seg_start, i, alpha=0.15, color=COLORS[prev_scheme],
                        label=prev_scheme if seg_start == 0 else "")
            seg_start   = i
            prev_scheme = name
    ax1.set_title("SNR Over Packets (colour = active scheme)", color="white",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("SNR (dB)", color="white")
    ax1.tick_params(colors="white")
    for t in SNR_THRESHOLDS_UP[1:]:
        ax1.axhline(t, color="yellow", linewidth=0.8, linestyle=":")
    ax1.grid(color="#333", linestyle="--", linewidth=0.5)
    for sp in ax1.spines.values():
        sp.set_edgecolor("#333")

    # ── Panel 2: Modulation index staircase ───────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#1A1D27")
    ax2.step(packets, scheme_idx, where="post", color="#FFB347", linewidth=2)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["BPSK", "QPSK", "16-QAM"], color="white", fontsize=9)
    ax2.set_title("Selected Modulation Scheme per Packet", color="white",
                  fontsize=12, fontweight="bold")
    ax2.tick_params(colors="white", axis="x")
    ax2.grid(color="#333", linestyle="--", linewidth=0.5)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333")
    # colour-fill each step
    for i, (name, idx) in enumerate(zip(scheme_names, scheme_idx)):
        ax2.fill_between([packets[i], packets[i] + 1], idx - 0.4, idx + 0.4,
                          color=COLORS[name], alpha=0.35, step="post")

    # ── Panel 3: BER timeline ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#1A1D27")
    ber_clipped = np.clip(ber_vals, 1e-6, 1.0)
    ax3.semilogy(packets, ber_clipped, color="#FF6B6B", linewidth=1.5,
                 label="BER per packet")
    ax3.axhline(0.01, color="yellow", linestyle="--", linewidth=1,
                label="BER = 1%")
    ax3.set_title("BER per Packet", color="white", fontsize=12,
                  fontweight="bold")
    ax3.set_xlabel("Packet Index", color="white")
    ax3.set_ylabel("BER", color="white")
    ax3.tick_params(colors="white")
    ax3.legend(facecolor="#1A1D27", labelcolor="white", edgecolor="#555",
               fontsize=9)
    ax3.grid(color="#333", which="both", linestyle="--", linewidth=0.5)
    for sp in ax3.spines.values():
        sp.set_edgecolor("#333")

    fig.suptitle("Adaptive Modulation Switching Behaviour", fontsize=15,
                 color="white", fontweight="bold", y=0.99)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓  Saved: {save_path}")


def plot_system_overview(save_path="system_overview.png"):
    """Simple block-diagram overview of the system."""
    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")
    ax.set_xlim(0, 13); ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Adaptive Modulation System – Block Diagram", color="white",
                 fontsize=13, fontweight="bold", pad=10)

    blocks = [
        (0.7,  2, "Bit\nGenerator"),
        (2.5,  2, "Adaptive\nController"),
        (4.5,  2, "Modulator\n(BPSK/QPSK\n/16-QAM)"),
        (6.7,  2, "Packet\nBuilder"),
        (8.7,  2, "AWGN\nChannel"),
        (10.7, 2, "Demodulator\n& Header\nParser"),
        (12.3, 2, "BER\nComputer"),
    ]
    block_colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B2",
                    "#C44E52", "#64B5CD", "#8C8C8C"]

    for (x, y, label), color in zip(blocks, block_colors):
        rect = plt.Rectangle((x - 0.6, y - 0.6), 1.2, 1.2,
                               facecolor=color, edgecolor="white",
                               linewidth=1.2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", color="white",
                fontsize=7.5, fontweight="bold")

    # arrows
    for i in range(len(blocks) - 1):
        x1 = blocks[i][0]   + 0.6
        x2 = blocks[i+1][0] - 0.6
        y  = blocks[i][1]
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="white",
                                    lw=1.5))

    # SNR feedback arrow
    ax.annotate("", xy=(2.5, 3.2), xytext=(8.7, 3.2),
                arrowprops=dict(arrowstyle="->", color="#FFB347",
                                lw=1.5, connectionstyle="arc3,rad=0"))
    ax.annotate("", xy=(2.5, 3.2), xytext=(2.5, 2.6),
                arrowprops=dict(arrowstyle="->", color="#FFB347", lw=1.5))
    ax.text(5.6, 3.35, "SNR Feedback", color="#FFB347", fontsize=9,
            ha="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# 10.  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("  Adaptive Modulation Communication System Simulator")
    print("=" * 60)

    # ── A. System overview block diagram ──────────────────────
    print("\n[1/4]  Generating system overview diagram …")
    plot_system_overview("system_overview.png")

    # ── B. Constellation diagrams ─────────────────────────────
    print("\n[2/4]  Generating constellation diagrams …")
    plot_constellations(snr_db=15.0, save_path="constellation_diagrams.png")

    # ── C. BER vs SNR sweep ───────────────────────────────────
    print("\n[3/4]  Running BER vs SNR sweep (this may take a moment) …")
    snr_range = np.arange(-2, 22, 1)
    ber_results = run_ber_vs_snr_simulation(snr_range, n_bits_per_packet=512,
                                             n_packets=30)
    plot_ber_vs_snr(ber_results, save_path="ber_vs_snr.png")

    # ── D. Adaptive simulation ────────────────────────────────
    print("\n[4/4]  Running adaptive modulation simulation …")
    # SNR profile: ramp up, plateau, ramp down, noisy region
    snr_profile = np.concatenate([
        np.linspace(-2, 5,  20),    # slow ramp up into BPSK territory
        np.linspace(5,  15, 20),    # transition through QPSK
        np.linspace(15, 20, 10),    # 16-QAM territory
        np.linspace(20, 5,  20),    # degradation
        np.random.uniform(3, 16, 30),  # noisy / rapidly varying
    ])
    adaptive_log = run_adaptive_simulation(snr_profile, n_bits_per_packet=256)
    plot_adaptive_behaviour(adaptive_log, save_path="adaptive_switching.png")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for scheme in MODULATION_SCHEMES:
        data   = ber_results[scheme.name]
        mid_i  = len(data["snr"]) // 2
        print(f"  {scheme.name:8s}  |  BER @ {data['snr'][mid_i]:.0f} dB = "
              f"{data['ber'][mid_i]:.4f}")

    used = {r[1] for r in adaptive_log}
    print(f"\n  Modulation schemes used in adaptive run: {', '.join(sorted(used))}")
    avg_ber = np.mean([r[2] for r in adaptive_log])
    print(f"  Average BER across all adaptive packets : {avg_ber:.4f}")
    print("\n  Output files:")
    for f in ["system_overview.png", "constellation_diagrams.png",
              "ber_vs_snr.png", "adaptive_switching.png"]:
        print(f"    • {f}")
    print("\nDone.\n")
