"""Generate all figures for Lecture 2: Universal Approximation & Going Deep."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Helvetica', 'Arial', 'sans-serif'],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.dpi': 200,
})

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec02"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY = "#1e3a5f"
PRIMARY_LIGHT = "#2e5a8f"
ACCENT = "#e85a4f"
SUCCESS = "#2a9d8f"
WARNING = "#e9c46a"
PURPLE = "#7c3aed"


def fig_uat_step_functions():
    """Show how ReLU networks approximate functions using step-like bumps."""
    x = np.linspace(-3, 3, 1000)

    # Target function
    target = np.sin(2 * x) + 0.5 * np.cos(5 * x)

    # Approximations with increasing neurons
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    neuron_counts = [2, 5, 20, 100]

    for ax, n in zip(axes, neuron_counts):
        np.random.seed(42)
        # Simple random features approximation
        W = np.random.randn(n) * 2
        b = np.random.randn(n) * 2
        H = np.maximum(0, np.outer(x, W) + b)  # ReLU features

        # Least squares fit
        alpha, _, _, _ = np.linalg.lstsq(H, target, rcond=None)
        approx = H @ alpha

        ax.plot(x, target, color='#cbd5e0', linewidth=2, label='Target $f(x)$')
        ax.plot(x, approx, color=ACCENT, linewidth=2.5, label=f'{n} neurons')
        ax.set_title(f'{n} hidden neurons', fontsize=13)
        ax.legend(fontsize=9, loc='lower left')
        ax.set_ylim(-2.5, 2.5)
        ax.grid(True, alpha=0.15)
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle('Universal Approximation: More Neurons = Better Fit (1 hidden layer)',
                 fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(OUT / "uat_approximation.png")
    plt.close()


def fig_depth_vs_width():
    """Compare shallow-wide vs deep-narrow networks on a complex function."""
    x = np.linspace(-2, 2, 500)
    # A complex target with multi-scale features
    target = np.sin(3*x) * np.exp(-x**2) + 0.3 * np.sign(x) * np.sqrt(np.abs(x))

    np.random.seed(123)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    configs = [
        ("Shallow: 1 layer, 100 neurons\n(100 params)", 1, 100, PRIMARY),
        ("Medium: 3 layers, 10 neurons\n(~130 params)", 3, 10, SUCCESS),
        ("Deep: 5 layers, 6 neurons\n(~120 params)", 5, 6, ACCENT),
    ]

    for ax, (title, depth, width, color) in zip(axes, configs):
        # Simulate with random features (simplified)
        h = x.reshape(-1, 1)
        for _ in range(depth):
            W = np.random.randn(h.shape[1], width) * np.sqrt(2.0 / h.shape[1])
            h = np.maximum(0, h @ W)  # ReLU layers

        if h.shape[1] > 0 and not np.all(h == 0):
            alpha, _, _, _ = np.linalg.lstsq(h, target, rcond=None)
            approx = h @ alpha
        else:
            approx = np.zeros_like(x)

        ax.plot(x, target, color='#cbd5e0', linewidth=2, label='Target')
        ax.plot(x, approx, color=color, linewidth=2.5, label='Approximation')
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.15)
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle('Depth vs Width: Similar Parameter Count, Different Capacity',
                 fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(OUT / "depth_vs_width.png")
    plt.close()


def fig_resnet_skip_connection():
    """Diagram of a residual block with skip connection."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Main path boxes
    boxes = [
        (0.15, 0.5, "Input\n$\\mathbf{x}$", '#e2e8f0', 'black'),
        (0.35, 0.5, "Conv / Linear\n+ BN + ReLU", PRIMARY_LIGHT, 'white'),
        (0.55, 0.5, "Conv / Linear\n+ BN", PRIMARY, 'white'),
        (0.72, 0.5, "$+$", '#fef3c7', WARNING),
        (0.82, 0.5, "ReLU", '#d1fae5', SUCCESS),
        (0.95, 0.5, "Output\n$\\mathcal{F}(\\mathbf{x}) + \\mathbf{x}$", '#fce7f3', ACCENT),
    ]

    for x, y, label, bg, tc in boxes:
        w, h = 0.13, 0.22
        if label in ('$+$',):
            circle = plt.Circle((x, y), 0.04, facecolor=bg, edgecolor='#94a3b8',
                                linewidth=2, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, label, ha='center', va='center', fontsize=14,
                    color=tc, fontweight='bold', zorder=6)
        elif label == 'ReLU':
            rect = matplotlib.patches.FancyBboxPatch((x-0.05, y-0.06), 0.1, 0.12,
                    boxstyle="round,pad=0.01", facecolor=bg, edgecolor='#94a3b8',
                    linewidth=1.5, zorder=5)
            ax.add_patch(rect)
            ax.text(x, y, label, ha='center', va='center', fontsize=11,
                    color=tc, fontweight='bold', zorder=6)
        else:
            rect = matplotlib.patches.FancyBboxPatch((x-w/2, y-h/2), w, h,
                    boxstyle="round,pad=0.02", facecolor=bg, edgecolor='#94a3b8',
                    linewidth=1.5, zorder=5)
            ax.add_patch(rect)
            ax.text(x, y, label, ha='center', va='center', fontsize=10,
                    color=tc, fontweight='bold', zorder=6)

    # Main path arrows
    main_edges = [(0.22, 0.29), (0.42, 0.49), (0.62, 0.68), (0.76, 0.77), (0.87, 0.88)]
    for sx, dx in main_edges:
        ax.annotate('', xy=(dx, 0.5), xytext=(sx, 0.5),
                    arrowprops=dict(arrowstyle='->', color='#64748b', lw=2))

    # Skip connection (curved arrow on top)
    skip_style = matplotlib.patches.FancyArrowPatch(
        (0.22, 0.62), (0.68, 0.54),
        connectionstyle="arc3,rad=-0.35",
        arrowstyle='->', mutation_scale=20,
        color=ACCENT, linewidth=3, linestyle='-', zorder=4)
    ax.add_patch(skip_style)

    ax.text(0.45, 0.88, 'Skip / Identity Connection', ha='center', fontsize=13,
            color=ACCENT, fontweight='bold')
    ax.text(0.45, 0.82, '$\\mathbf{x}$ (identity)', ha='center', fontsize=11,
            color=ACCENT, style='italic')

    # Label the learned part
    ax.annotate('', xy=(0.62, 0.32), xytext=(0.29, 0.32),
                arrowprops=dict(arrowstyle='<->', color=PRIMARY_LIGHT, lw=1.5))
    ax.text(0.455, 0.25, 'Residual: $\\mathcal{F}(\\mathbf{x})$', ha='center',
            fontsize=12, color=PRIMARY_LIGHT, fontweight='bold')

    ax.set_xlim(0.02, 1.05)
    ax.set_ylim(0.1, 0.98)
    ax.axis('off')
    ax.set_title('Residual Block (He et al., 2015)', fontsize=15, pad=15)

    plt.tight_layout()
    plt.savefig(OUT / "resnet_block.png")
    plt.close()


def fig_resnet_vs_plain():
    """Show training error of plain network vs ResNet with depth."""
    np.random.seed(42)
    depths = [8, 14, 20, 32, 56, 110]

    # Plain network: error increases past optimal depth
    plain_train = [8.5, 5.2, 4.8, 6.1, 8.5, 12.0]
    plain_test = [12.0, 8.5, 8.0, 10.5, 14.0, 19.0]

    # ResNet: error keeps decreasing
    res_train = [8.5, 5.0, 3.8, 3.0, 2.5, 2.2]
    res_test = [12.0, 8.0, 6.5, 5.8, 5.2, 5.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(depths, plain_train, 'o-', color=ACCENT, linewidth=2.5, markersize=8, label='Plain net')
    ax1.plot(depths, res_train, 's-', color=SUCCESS, linewidth=2.5, markersize=8, label='ResNet')
    ax1.set_xlabel('Network depth (layers)')
    ax1.set_ylabel('Training error (%)')
    ax1.set_title('Training Error', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.15)
    ax1.spines[['top', 'right']].set_visible(False)

    ax2.plot(depths, plain_test, 'o-', color=ACCENT, linewidth=2.5, markersize=8, label='Plain net')
    ax2.plot(depths, res_test, 's-', color=SUCCESS, linewidth=2.5, markersize=8, label='ResNet')
    ax2.set_xlabel('Network depth (layers)')
    ax2.set_ylabel('Test error (%)')
    ax2.set_title('Test Error', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.15)
    ax2.spines[['top', 'right']].set_visible(False)

    # Annotation
    ax1.annotate('Degradation\nproblem!', xy=(56, 8.5), fontsize=11,
                 color=ACCENT, fontweight='bold', ha='center',
                 arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.5),
                 xytext=(70, 5))

    plt.suptitle('The Degradation Problem: Deeper ≠ Better (without skip connections)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "resnet_vs_plain.png")
    plt.close()


def fig_uat_relu_bumps():
    """Show how ReLU units create 'bump' functions that sum to approximate a target."""
    x = np.linspace(-3, 3, 500)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 1.2]})

    # Top: individual bump functions
    ax = axes[0]
    bumps = [
        (-2.0, 1.0, 0.8),
        (-0.5, 0.5, -1.2),
        (0.5, 1.5, 1.5),
        (1.5, 2.5, -0.6),
    ]

    colors_bump = [PRIMARY, ACCENT, SUCCESS, PURPLE]
    total = np.zeros_like(x)

    for (a, b, h), c in zip(bumps, colors_bump):
        bump = h * (np.maximum(0, x - a) - 2 * np.maximum(0, x - (a+b)/2) + np.maximum(0, x - b))
        ax.plot(x, bump, color=c, linewidth=2, alpha=0.7, linestyle='--')
        total += bump

    ax.set_title('Individual ReLU "bump" functions (each = 2 ReLUs)', fontsize=13)
    ax.axhline(y=0, color='#cbd5e0', linewidth=0.8)
    ax.set_ylabel('Output')
    ax.grid(True, alpha=0.15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend([f'Bump {i+1}' for i in range(4)], fontsize=10, ncol=4, loc='upper right')

    # Bottom: sum approximates target
    ax = axes[1]
    target = 0.8 * np.sin(1.5 * x)
    ax.plot(x, target, color='#cbd5e0', linewidth=3, label='Target $f(x)$')
    ax.plot(x, total, color=ACCENT, linewidth=2.5, label='Sum of bumps')
    ax.set_title('Sum of bumps approximates the target function', fontsize=13)
    ax.axhline(y=0, color='#cbd5e0', linewidth=0.8)
    ax.set_xlabel('$x$')
    ax.set_ylabel('Output')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.15)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "uat_relu_bumps.png")
    plt.close()


def fig_depth_compositionality():
    """Show how depth enables compositional representations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Flat (wide) - all combinations
    ax = axes[0]
    ax.set_title('Shallow Network\n(must learn all combinations)', fontsize=13, color=ACCENT)
    features = ['cat+sitting', 'cat+running', 'cat+sleeping',
                'dog+sitting', 'dog+running', 'dog+sleeping',
                'bird+sitting', 'bird+flying', 'bird+sleeping']
    for i, f in enumerate(features):
        y = 0.9 - i * 0.1
        ax.add_patch(plt.Rectangle((0.1, y-0.035), 0.8, 0.07,
                     facecolor='#fef2f2', edgecolor=ACCENT, linewidth=1.5, alpha=0.8))
        ax.text(0.5, y, f, ha='center', va='center', fontsize=10, color=ACCENT)
    ax.text(0.5, -0.05, f'{len(features)} separate detectors needed', ha='center',
            fontsize=12, color=ACCENT, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.15, 1.0)
    ax.axis('off')

    # Right: Deep (compositional) - hierarchical
    ax = axes[1]
    ax.set_title('Deep Network\n(learns composable parts)', fontsize=13, color=SUCCESS)

    # Layer 1: primitives
    l1 = ['edges', 'textures', 'curves']
    for i, f in enumerate(l1):
        y = 0.8 - i * 0.15
        ax.add_patch(plt.Rectangle((0.0, y-0.04), 0.22, 0.08,
                     facecolor='#ecfdf5', edgecolor=SUCCESS, linewidth=1.5))
        ax.text(0.11, y, f, ha='center', va='center', fontsize=9, color=SUCCESS)

    # Layer 2: parts
    l2 = ['ear', 'leg', 'wing']
    for i, f in enumerate(l2):
        y = 0.8 - i * 0.15
        ax.add_patch(plt.Rectangle((0.3, y-0.04), 0.18, 0.08,
                     facecolor='#d1fae5', edgecolor=SUCCESS, linewidth=1.5))
        ax.text(0.39, y, f, ha='center', va='center', fontsize=9, color=SUCCESS)

    # Layer 3: objects
    l3 = ['cat', 'dog', 'bird']
    for i, f in enumerate(l3):
        y = 0.8 - i * 0.15
        ax.add_patch(plt.Rectangle((0.56, y-0.04), 0.16, 0.08,
                     facecolor='#a7f3d0', edgecolor=SUCCESS, linewidth=1.5))
        ax.text(0.64, y, f, ha='center', va='center', fontsize=9, color=SUCCESS)

    # Layer 4: actions (reusable!)
    l4 = ['sitting', 'running']
    for i, f in enumerate(l4):
        y = 0.75 - i * 0.15
        ax.add_patch(plt.Rectangle((0.8, y-0.04), 0.18, 0.08,
                     facecolor='#6ee7b7', edgecolor=SUCCESS, linewidth=1.5))
        ax.text(0.89, y, f, ha='center', va='center', fontsize=9, color=SUCCESS)

    # Arrows between layers
    for i in range(3):
        ax.annotate('', xy=(0.29, 0.65), xytext=(0.23, 0.65),
                    arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=1.5))
    ax.annotate('', xy=(0.55, 0.65), xytext=(0.49, 0.65),
                arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=1.5))
    ax.annotate('', xy=(0.79, 0.65), xytext=(0.73, 0.65),
                arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=1.5))

    ax.text(0.5, -0.05, '3+3+3+2 = 11 reusable parts (not 9 combos!)',
            ha='center', fontsize=12, color=SUCCESS, fontweight='bold')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.0)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUT / "depth_compositionality.png")
    plt.close()


def fig_gradient_flow_comparison():
    """Compare gradient flow in plain vs ResNet architectures."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    layers = np.arange(1, 51)

    # Plain network: gradients vanish
    np.random.seed(42)
    plain_grad = np.cumprod(np.random.uniform(0.7, 0.95, 50))
    plain_grad = plain_grad[::-1]  # gradient flows backward

    # ResNet: gradients maintained
    res_grad = np.ones(50)
    for i in range(1, 50):
        res_grad[i] = res_grad[i-1] * np.random.uniform(0.85, 1.0) + np.random.uniform(0.1, 0.3)
    res_grad = res_grad[::-1] / res_grad.max()

    ax = axes[0]
    ax.semilogy(layers, plain_grad, color=ACCENT, linewidth=2.5)
    ax.fill_between(layers, plain_grad, alpha=0.15, color=ACCENT)
    ax.set_title('Plain Network (50 layers)', fontsize=13, color=ACCENT)
    ax.set_xlabel('Layer (from output)')
    ax.set_ylabel('Gradient magnitude (log)')
    ax.axhline(y=1e-4, color='gray', linestyle='--', linewidth=1)
    ax.text(35, 1.5e-4, 'Too small to learn', fontsize=10, color='gray')
    ax.grid(True, alpha=0.15)
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    ax.semilogy(layers, res_grad, color=SUCCESS, linewidth=2.5)
    ax.fill_between(layers, res_grad, alpha=0.15, color=SUCCESS)
    ax.set_title('ResNet (50 layers)', fontsize=13, color=SUCCESS)
    ax.set_xlabel('Layer (from output)')
    ax.set_ylabel('Gradient magnitude (log)')
    ax.set_ylim(axes[0].get_ylim())  # same scale
    ax.grid(True, alpha=0.15)
    ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle('Gradient Flow: Skip Connections Prevent Vanishing Gradients', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "gradient_flow_comparison.png")
    plt.close()


def fig_loss_landscape_skip():
    """Show how skip connections smooth the loss landscape (inspired by Li et al. 2018)."""
    from mpl_toolkits.mplot3d import Axes3D

    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                     subplot_kw={'projection': '3d'})

    # Without skip connections: rugged
    Z1 = X**2 + Y**2 + 2*np.sin(4*X)*np.cos(4*Y) + 0.5*np.sin(8*X)*np.cos(6*Y)
    ax1.plot_surface(X, Y, Z1, cmap='Reds', alpha=0.85, edgecolor='none')
    ax1.set_title('Without Skip Connections\n(rugged, many local minima)', fontsize=12, color=ACCENT)
    ax1.set_xlabel('$\\theta_1$', fontsize=10)
    ax1.set_ylabel('$\\theta_2$', fontsize=10)
    ax1.set_zlabel('Loss', fontsize=10)
    ax1.view_init(elev=35, azim=-60)

    # With skip connections: smooth
    Z2 = X**2 + Y**2 + 0.3*np.sin(2*X)*np.cos(2*Y)
    ax2.plot_surface(X, Y, Z2, cmap='Greens', alpha=0.85, edgecolor='none')
    ax2.set_title('With Skip Connections\n(smooth, easy to optimize)', fontsize=12, color=SUCCESS)
    ax2.set_xlabel('$\\theta_1$', fontsize=10)
    ax2.set_ylabel('$\\theta_2$', fontsize=10)
    ax2.set_zlabel('Loss', fontsize=10)
    ax2.view_init(elev=35, azim=-60)

    plt.tight_layout()
    plt.savefig(OUT / "loss_landscape_skip.png")
    plt.close()


def fig_param_count_depth_width():
    """Bar chart comparing parameter counts for different depth/width configs."""
    configs = [
        ("1×512", 1 * 784 * 512 + 512 * 10),
        ("2×256", 784*256 + 256*256 + 256*10),
        ("4×128", 784*128 + 3*128*128 + 128*10),
        ("8×64", 784*64 + 7*64*64 + 64*10),
        ("16×32", 784*32 + 15*32*32 + 32*10),
    ]

    names = [c[0] for c in configs]
    params = [c[1] for c in configs]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [ACCENT, WARNING, SUCCESS, PRIMARY_LIGHT, PURPLE]
    bars = ax.bar(names, [p/1000 for p in params], color=colors, edgecolor='white', linewidth=2)

    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{p/1000:.0f}K', ha='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Architecture (depth × width)', fontsize=13)
    ax.set_ylabel('Parameters (thousands)', fontsize=13)
    ax.set_title('Parameter Count: Depth × Width Tradeoff\n(MNIST input: 784, output: 10)', fontsize=14)
    ax.grid(True, alpha=0.15, axis='y')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "param_count_depth_width.png")
    plt.close()


if __name__ == "__main__":
    print("Generating Lecture 2 figures...")
    fig_uat_step_functions()
    print("  uat_approximation.png")
    fig_uat_relu_bumps()
    print("  uat_relu_bumps.png")
    fig_depth_vs_width()
    print("  depth_vs_width.png")
    fig_depth_compositionality()
    print("  depth_compositionality.png")
    fig_resnet_skip_connection()
    print("  resnet_block.png")
    fig_resnet_vs_plain()
    print("  resnet_vs_plain.png")
    fig_gradient_flow_comparison()
    print("  gradient_flow_comparison.png")
    fig_loss_landscape_skip()
    print("  loss_landscape_skip.png")
    fig_param_count_depth_width()
    print("  param_count_depth_width.png")
    print("Done: all Lecture 2 figures saved to", OUT)
