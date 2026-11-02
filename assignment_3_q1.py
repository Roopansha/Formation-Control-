import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from scipy.optimize import linear_sum_assignment

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
NAME = "ROOPANSHA"          # name to display (capital letters A-Z)
N = 20                  # number of agents
P_EDGE = 0.3            # Erdos-Renyi edge probability p in (0, 1)
SEED = 7                # random seed for reproducibility

K_GAIN = 1.0            # formation control gain
DT = 0.01               # integration time step
TOL = 1e-3              # formation error at which a letter counts as formed
MAX_T_PER_LETTER = 15.0 # safety cap on simulated time per letter
HOLD_FRAMES = 25        # frames to hold each completed letter
STEPS_PER_FRAME = 4     # simulation steps per animation frame
FPS = 25

LETTER_HEIGHT = 8.0     # size of the letters in R^2
INIT_BOX = 10.0         # initial positions are uniform in [-INIT_BOX, INIT_BOX]^2

VIDEO_FILE = "name_formation.mp4"
GIF_FILE = "name_formation.gif"

rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------
# Connected Erdos-Renyi graph
# ----------------------------------------------------------------------------
def erdos_renyi(n, p):
    upper = np.triu(rng.random((n, n)) < p, k=1)
    return (upper | upper.T).astype(float)


def laplacian(A):
    return np.diag(A.sum(axis=1)) - A


def connected_erdos_renyi(n, p):
    """Resample G(n, p) until the graph is connected (lambda_2(L) > 0)."""
    attempts = 0
    while True:
        attempts += 1
        A = erdos_renyi(n, p)
        lam = np.sort(np.linalg.eigvalsh(laplacian(A)))
        if lam[1] > 1e-9:
            return A, lam[1], attempts


# ----------------------------------------------------------------------------
# Letter shapes: each letter is a list of polylines in a 1.0 x 1.4 box
# ----------------------------------------------------------------------------
def arc(cx, cy, rx, ry, a0, a1, n=40):
    t = np.radians(np.linspace(a0, a1, n))
    return list(zip(cx + rx * np.cos(t), cy + ry * np.sin(t)))


P_STROKES = [[(0, 0), (0, 1.4), (0.6, 1.4)] + arc(0.6, 1.05, 0.35, 0.35, 90, -90) + [(0, 0.7)]]

LETTERS = {
    "A": [[(0, 0), (0.5, 1.4), (1, 0)], [(0.25, 0.7), (0.75, 0.7)]],
    "B": [[(0, 0), (0, 1.4)],
          [(0, 1.4), (0.6, 1.4)] + arc(0.6, 1.05, 0.35, 0.35, 90, -90) + [(0, 0.7)],
          [(0, 0.7), (0.6, 0.7)] + arc(0.6, 0.35, 0.4, 0.35, 90, -90) + [(0, 0)]],
    "C": [arc(0.5, 0.7, 0.5, 0.7, 45, 315)],
    "D": [[(0, 0), (0, 1.4), (0.3, 1.4)] + arc(0.3, 0.7, 0.7, 0.7, 90, -90) + [(0, 0)]],
    "E": [[(1, 1.4), (0, 1.4), (0, 0), (1, 0)], [(0, 0.7), (0.75, 0.7)]],
    "F": [[(1, 1.4), (0, 1.4), (0, 0)], [(0, 0.7), (0.75, 0.7)]],
    "G": [arc(0.5, 0.7, 0.5, 0.7, 45, 360) + [(0.55, 0.7)]],
    "H": [[(0, 0), (0, 1.4)], [(1, 0), (1, 1.4)], [(0, 0.7), (1, 0.7)]],
    "I": [[(0.5, 0), (0.5, 1.4)], [(0.2, 1.4), (0.8, 1.4)], [(0.2, 0), (0.8, 0)]],
    "J": [[(0.2, 1.4), (1, 1.4)], [(0.75, 1.4), (0.75, 0.35)] + arc(0.4, 0.35, 0.35, 0.35, 0, -180)],
    "K": [[(0, 0), (0, 1.4)], [(1, 1.4), (0, 0.6)], [(0.35, 0.85), (1, 0)]],
    "L": [[(0, 1.4), (0, 0), (1, 0)]],
    "M": [[(0, 0), (0, 1.4), (0.5, 0.7), (1, 1.4), (1, 0)]],
    "N": [[(0, 0), (0, 1.4), (1, 0), (1, 1.4)]],
    "O": [arc(0.5, 0.7, 0.5, 0.7, 0, 360)],
    "P": P_STROKES,
    "Q": [arc(0.5, 0.7, 0.5, 0.7, 0, 360), [(0.6, 0.35), (1, 0)]],
    "R": P_STROKES + [[(0.4, 0.7), (1, 0)]],
    "S": [arc(0.5, 1.05, 0.5, 0.35, 20, 270) + arc(0.5, 0.35, 0.5, 0.35, 90, -160)],
    "T": [[(0, 1.4), (1, 1.4)], [(0.5, 1.4), (0.5, 0)]],
    "U": [[(0, 1.4), (0, 0.5)] + arc(0.5, 0.5, 0.5, 0.5, 180, 360) + [(1, 1.4)]],
    "V": [[(0, 1.4), (0.5, 0), (1, 1.4)]],
    "W": [[(0, 1.4), (0.25, 0), (0.5, 0.9), (0.75, 0), (1, 1.4)]],
    "X": [[(0, 0), (1, 1.4)], [(0, 1.4), (1, 0)]],
    "Y": [[(0, 1.4), (0.5, 0.7), (1, 1.4)], [(0.5, 0.7), (0.5, 0)]],
    "Z": [[(0, 1.4), (1, 1.4), (0, 0), (1, 0)]],
}


def densify(polyline, spacing=0.005):
    pts = np.asarray(polyline, dtype=float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        m = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        out.extend(a + (b - a) * t for t in np.linspace(0, 1, m + 1)[1:])
    return np.array(out)


def letter_formation(ch, n):
    """n well-spread points on the letter (farthest point sampling), centred at
    the origin and scaled to LETTER_HEIGHT."""
    dense = np.vstack([densify(s) for s in LETTERS[ch]])
    chosen = [0]
    dist = np.linalg.norm(dense - dense[0], axis=1)
    for _ in range(n - 1):
        idx = int(np.argmax(dist))
        chosen.append(idx)
        dist = np.minimum(dist, np.linalg.norm(dense - dense[idx], axis=1))
    pts = dense[chosen]
    pts = pts - pts.mean(axis=0)
    return pts * (LETTER_HEIGHT / 1.4)


# ----------------------------------------------------------------------------
# Formation control
#   x_i' = K * sum_j a_ij [ (x_j - x_i) - (p_j* - p_i*) ]
#   i.e.  x' = -K L (x - p*)   (applied to each coordinate)
# Since 1^T L = 0, the centroid of x is invariant, and for a connected graph
# x(t) -> p* + centroid(x[0]).
# ----------------------------------------------------------------------------
def assign_targets(x, p):
    """Match agents to the letter's points so that the total travel is minimal."""
    rel = x - x.mean(axis=0)
    cost = np.linalg.norm(rel[:, None, :] - p[None, :, :], axis=2) ** 2
    rows, cols = linear_sum_assignment(cost)
    p_star = np.empty_like(p)
    p_star[rows] = p[cols]
    return p_star


def formation_error(x, p_star):
    d = (x - x.mean(axis=0)) - p_star
    return np.linalg.norm(d)


def simulate(x0, L):
    """Returns per-frame positions, targets, letter index and time."""
    frames = []
    x = x0.copy()
    t = 0.0
    for _ in range(HOLD_FRAMES):
        frames.append((x.copy(), None, -1, t))

    for k, ch in enumerate(NAME):
        p_star = assign_targets(x, letter_formation(ch, len(x)))
        step = 0
        t_letter = 0.0
        while formation_error(x, p_star) > TOL and t_letter < MAX_T_PER_LETTER:
            x = x + DT * (-K_GAIN * L @ (x - p_star))
            t += DT
            t_letter += DT
            step += 1
            if step % STEPS_PER_FRAME == 0:
                frames.append((x.copy(), p_star, k, t))
        print(f"Letter {ch}: formed in {t_letter:.2f} s, "
              f"error = {formation_error(x, p_star):.2e}")
        for _ in range(HOLD_FRAMES):
            frames.append((x.copy(), p_star, k, t))
    return frames


# ----------------------------------------------------------------------------
# Animation
# ----------------------------------------------------------------------------
def animate(frames, A, centroid):
    edges = np.argwhere(np.triu(A) > 0)

    all_x = np.vstack([f[0] for f in frames])
    lo, hi = all_x.min(axis=0) - 1.0, all_x.max(axis=0) + 1.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    edge_lines = LineCollection([], colors="tab:gray", linewidths=0.6, alpha=0.35)
    ax.add_collection(edge_lines)
    targets = ax.scatter([], [], s=120, facecolors="none", edgecolors="tab:red",
                         linewidths=1.0, alpha=0.6, label="target formation")
    agents = ax.scatter([], [], s=60, c="tab:blue", zorder=3, label="agents")
    title = ax.set_title("", fontsize=14, family="monospace")
    info = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9)
    ax.legend(loc="upper right", fontsize=9)

    def update(i):
        x, p_star, k, t = frames[i]
        edge_lines.set_segments(x[edges])
        agents.set_offsets(x)
        if p_star is None:
            targets.set_offsets(np.empty((0, 2)))
            title.set_text("Initial positions x[0]")
        else:
            targets.set_offsets(p_star + centroid)
            shown = "".join(c if j <= k else "_" for j, c in enumerate(NAME))
            title.set_text(f"Forming '{NAME[k]}'   [{' '.join(shown)}]")
        info.set_text(f"N = {N}, Erdos-Renyi p = {P_EDGE},  t = {t:5.2f} s")
        return edge_lines, agents, targets, title, info

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000 / FPS, blit=False)
    return fig, anim


def save(anim):
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    if animation.writers.is_available("ffmpeg"):
        anim.save(VIDEO_FILE, writer=animation.FFMpegWriter(fps=FPS, bitrate=2000))
        print(f"Saved {VIDEO_FILE}")
    else:
        print("ffmpeg not found (pip install imageio-ffmpeg), skipping MP4")
    anim.save(GIF_FILE, writer=animation.PillowWriter(fps=FPS))
    print(f"Saved {GIF_FILE}")


def main():
    A, lam2, attempts = connected_erdos_renyi(N, P_EDGE)
    L = laplacian(A)
    print(f"Connected G({N}, {P_EDGE}) after {attempts} draw(s): "
          f"{int(A.sum() / 2)} edges, lambda_2(L) = {lam2:.3f}")

    x0 = rng.uniform(-INIT_BOX, INIT_BOX, size=(N, 2))
    frames = simulate(x0, L)

    fig, anim = animate(frames, A, x0.mean(axis=0))
    save(anim)
    plt.close(fig)


if __name__ == "__main__":
    main()
