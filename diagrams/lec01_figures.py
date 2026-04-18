"""Generate all figures for Lecture 1: Why Deep Learning + MLP Recap."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif'],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.dpi': 200,
})

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec01"
OUT.mkdir(parents=True, exist_ok=True)

# Color palette matching dl-theme.css
PRIMARY = "#1e3a5f"
PRIMARY_LIGHT = "#2e5a8f"
ACCENT = "#e85a4f"
SUCCESS = "#2a9d8f"
WARNING = "#e9c46a"
PURPLE = "#7c3aed"
COLORS = [PRIMARY, ACCENT, SUCCESS, PURPLE, WARNING, PRIMARY_LIGHT]


def fig_activation_functions():
    """Plot common activation functions side by side."""
    z = np.linspace(-4, 4, 500)

    sigmoid = 1 / (1 + np.exp(-z))
    tanh = np.tanh(z)
    relu = np.maximum(0, z)
    leaky_relu = np.where(z > 0, z, 0.1 * z)
    gelu = z * 0.5 * (1 + np.vectorize(lambda x: np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))(z))
    swish = z * sigmoid

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    funcs = [
        (sigmoid, "Sigmoid", r"$\sigma(z) = \frac{1}{1+e^{-z}}$"),
        (tanh, "Tanh", r"$\tanh(z)$"),
        (relu, "ReLU", r"$\max(0, z)$"),
        (leaky_relu, "Leaky ReLU", r"$\max(0.1z, z)$"),
        (gelu, "GELU", r"$z \cdot \Phi(z)$"),
        (swish, "SiLU / Swish", r"$z \cdot \sigma(z)$"),
    ]

    for ax, (y, name, formula), color in zip(axes.flat, funcs, COLORS):
        ax.plot(z, y, color=color, linewidth=2.5)
        ax.axhline(y=0, color='#cbd5e0', linewidth=0.8, zorder=0)
        ax.axvline(x=0, color='#cbd5e0', linewidth=0.8, zorder=0)
        ax.set_title(f"{name}\n{formula}", fontsize=13)
        ax.set_xlim(-4, 4)
        ax.grid(True, alpha=0.15)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "activation_functions.png")
    plt.close()


def fig_activation_gradients():
    """Plot gradients of activation functions to show vanishing gradient problem."""
    z = np.linspace(-4, 4, 500)

    sigmoid = 1 / (1 + np.exp(-z))
    grad_sigmoid = sigmoid * (1 - sigmoid)
    grad_tanh = 1 - np.tanh(z)**2
    grad_relu = np.where(z > 0, 1.0, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (y, name, color, maxval) in zip(axes, [
        (grad_sigmoid, "Sigmoid gradient", ACCENT, 0.25),
        (grad_tanh, "Tanh gradient", PRIMARY, 1.0),
        (grad_relu, "ReLU gradient", SUCCESS, 1.0),
    ]):
        ax.plot(z, y, color=color, linewidth=2.5)
        ax.axhline(y=maxval, color=color, linewidth=1, linestyle='--', alpha=0.5)
        ax.annotate(f"max = {maxval}", xy=(2, maxval), fontsize=11,
                    color=color, va='bottom')
        ax.axhline(y=0, color='#cbd5e0', linewidth=0.8, zorder=0)
        ax.axvline(x=0, color='#cbd5e0', linewidth=0.8, zorder=0)
        ax.set_title(name, fontsize=13)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-0.1, 1.3)
        ax.grid(True, alpha=0.15)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "activation_gradients.png")
    plt.close()


def fig_loss_surface_3d():
    """3D visualization of a non-convex loss surface."""
    from mpl_toolkits.mplot3d import Axes3D

    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)

    # Interesting non-convex surface with local minima and saddle points
    Z = (1 - X)**2 + 3 * (Y - X**2)**2 + 0.5 * np.sin(3*X) * np.cos(3*Y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.85,
                           edgecolor='none', antialiased=True)

    # Mark some points
    ax.scatter([1], [1], [0.5*np.sin(3)*np.cos(3)], color=SUCCESS, s=100,
               zorder=10, edgecolor='white', linewidth=2)
    ax.scatter([0], [0], [1 + 0.5*np.sin(0)], color=ACCENT, s=100,
               zorder=10, marker='^', edgecolor='white', linewidth=2)

    ax.set_xlabel('Weight 1', fontsize=12)
    ax.set_ylabel('Weight 2', fontsize=12)
    ax.set_zlabel('Loss', fontsize=12)
    ax.set_title('Non-Convex Loss Surface', fontsize=15)
    ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    plt.savefig(OUT / "loss_surface_3d.png")
    plt.close()


def fig_loss_surface_contour():
    """Contour plot showing local minima, saddle points."""
    x = np.linspace(-2, 2, 300)
    y = np.linspace(-2, 2, 300)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 3 * (Y - X**2)**2 + 0.5 * np.sin(3*X) * np.cos(3*Y)

    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contourf(X, Y, Z, levels=30, cmap='coolwarm', alpha=0.8)
    ax.contour(X, Y, Z, levels=15, colors='white', linewidths=0.3, alpha=0.5)
    plt.colorbar(cs, ax=ax, label='Loss')

    # Find actual critical points numerically
    from scipy.optimize import minimize
    res = minimize(lambda p: (1-p[0])**2 + 3*(p[1]-p[0]**2)**2 + 0.5*np.sin(3*p[0])*np.cos(3*p[1]),
                   x0=[0.9, 0.9], method='Nelder-Mead')
    gmin = res.x

    ax.plot(gmin[0], gmin[1], 'o', color=SUCCESS, markersize=12, markeredgecolor='white',
            markeredgewidth=2, zorder=5)
    ax.annotate('Local min', (gmin[0], gmin[1]), (gmin[0]+0.4, gmin[1]+0.4), fontsize=11,
                color='white', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    ax.plot(-0.5, 0.5, '^', color=WARNING, markersize=12, markeredgecolor='white',
            markeredgewidth=2, zorder=5)
    ax.annotate('Another\nlocal min', (-0.5, 0.5), (-1.5, 1.0), fontsize=11,
                color='white', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    ax.set_xlabel('Weight 1')
    ax.set_ylabel('Weight 2')
    ax.set_title('Non-Convex Loss Landscape (illustrative)')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "loss_surface_contour.png")
    plt.close()


def fig_vanishing_gradient():
    """Show how gradient magnitude decreases with depth for sigmoid vs ReLU."""
    layers = np.arange(1, 21)

    # Sigmoid: max gradient ~0.25 per layer
    grad_sigmoid = 0.25 ** layers
    # Tanh: max gradient ~1.0 but typically ~0.6 in practice
    grad_tanh = 0.65 ** layers
    # ReLU: gradient is 1 (ideally)
    grad_relu = np.ones_like(layers, dtype=float) * 0.5  # ~50% active neurons

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(layers, grad_sigmoid, 'o-', color=ACCENT, linewidth=2.5,
                markersize=6, label='Sigmoid (max grad = 0.25)')
    ax.semilogy(layers, grad_tanh, 's-', color=WARNING, linewidth=2.5,
                markersize=6, label='Tanh (effective grad ~ 0.65)')
    ax.semilogy(layers, grad_relu, 'D-', color=SUCCESS, linewidth=2.5,
                markersize=6, label='ReLU (grad ~ 0.5, half active)')

    ax.axhline(y=1e-6, color='#cbd5e0', linewidth=1, linestyle='--')
    ax.annotate('Effectively zero', xy=(15, 1e-6), fontsize=11, color='gray', va='bottom')

    ax.set_xlabel('Layer depth (from output)')
    ax.set_ylabel('Gradient magnitude (log scale)')
    ax.set_title('Vanishing Gradients: Signal Decay Through Layers')
    ax.legend(fontsize=11, loc='center right')
    ax.set_xlim(0.5, 20.5)
    ax.grid(True, alpha=0.15)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "vanishing_gradient.png")
    plt.close()


def fig_training_curves():
    """Show typical training and validation loss/accuracy curves with overfitting."""
    epochs = np.arange(1, 51)
    np.random.seed(42)

    # Training loss: steadily decreases
    train_loss = 2.3 * np.exp(-0.08 * epochs) + 0.05 + np.random.normal(0, 0.02, len(epochs))
    # Validation loss: decreases then increases (overfitting)
    val_loss = 2.3 * np.exp(-0.06 * epochs) + 0.15 + 0.003 * (epochs - 25)**2 * (epochs > 25) + np.random.normal(0, 0.03, len(epochs))
    val_loss = np.clip(val_loss, 0.1, 3)

    # Accuracy curves
    train_acc = 100 * (1 - 0.8 * np.exp(-0.1 * epochs)) + np.random.normal(0, 0.3, len(epochs))
    val_acc = 100 * (1 - 0.85 * np.exp(-0.07 * epochs)) - 0.15 * (epochs - 25)**2 * (epochs > 30) / 50 + np.random.normal(0, 0.5, len(epochs))
    val_acc = np.clip(val_acc, 50, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, train_loss, color=PRIMARY, linewidth=2.5, label='Train loss')
    ax1.plot(epochs, val_loss, color=ACCENT, linewidth=2.5, label='Val loss')
    ax1.axvline(x=28, color='gray', linewidth=1, linestyle='--', alpha=0.6)
    ax1.annotate('Early stopping\npoint', xy=(28, 0.3), fontsize=10,
                 color='gray', ha='center')
    ax1.fill_betweenx([0, 3], 28, 50, alpha=0.05, color=ACCENT)
    ax1.annotate('Overfitting\nregion', xy=(39, 1.8), fontsize=11,
                 color=ACCENT, ha='center', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, 51)
    ax1.grid(True, alpha=0.15)
    ax1.spines[['top', 'right']].set_visible(False)

    # Accuracy
    ax2.plot(epochs, train_acc, color=PRIMARY, linewidth=2.5, label='Train accuracy')
    ax2.plot(epochs, val_acc, color=ACCENT, linewidth=2.5, label='Val accuracy')
    ax2.axvline(x=28, color='gray', linewidth=1, linestyle='--', alpha=0.6)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.set_xlim(0, 51)
    ax2.grid(True, alpha=0.15)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "training_curves.png")
    plt.close()


def fig_dl_timeline():
    """Timeline of key deep learning milestones."""
    events = [
        (2012, "AlexNet", "CNNs win\nImageNet"),
        (2014, "GANs", "Generative\nmodels"),
        (2015, "ResNet", "152 layers\nskip connections"),
        (2017, "Transformer", "Attention Is\nAll You Need"),
        (2018, "BERT / GPT", "Pretrained\nLMs"),
        (2020, "GPT-3", "175B params\nfew-shot"),
        (2022, "ChatGPT", "DL goes\nmainstream"),
        (2024, "GPT-4o\nLlama 3", "Multimodal\nopen-source"),
    ]

    fig, ax = plt.subplots(figsize=(16, 4))

    for i, (year, name, desc) in enumerate(events):
        color = COLORS[i % len(COLORS)]
        ax.plot(year, 0, 'o', color=color, markersize=14, markeredgecolor='white',
                markeredgewidth=2, zorder=5)
        yoff = 0.6 if i % 2 == 0 else -0.8
        ax.annotate(f"{name}\n{desc}", xy=(year, 0), xytext=(year, yoff),
                    fontsize=10, ha='center', va='center' if yoff > 0 else 'center',
                    color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color=color, lw=1.5))
        ax.text(year, -1.5 if i % 2 == 0 else 1.4, str(year),
                ha='center', fontsize=10, color='gray')

    ax.axhline(y=0, color='#e2e8f0', linewidth=3, zorder=1)
    ax.set_xlim(2010, 2026)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    ax.set_title('The Deep Learning Revolution', fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(OUT / "dl_timeline.png")
    plt.close()


def fig_mlp_architecture():
    """Draw a clean MLP architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))

    layer_sizes = [4, 6, 6, 3]
    layer_names = ['Input\n$\\mathbf{x}$', 'Hidden 1\n$\\mathbf{h}_1$',
                   'Hidden 2\n$\\mathbf{h}_2$', 'Output\n$\\hat{\\mathbf{y}}$']
    layer_colors = ['#e2e8f0', PRIMARY_LIGHT, PRIMARY, ACCENT]

    x_positions = np.linspace(0.15, 0.85, len(layer_sizes))

    for l, (n, x_pos, name, color) in enumerate(zip(layer_sizes, x_positions, layer_names, layer_colors)):
        y_positions = np.linspace(0.15, 0.85, n)
        for i, y in enumerate(y_positions):
            circle = plt.Circle((x_pos, y), 0.025, color=color, ec='white',
                                linewidth=2, zorder=3)
            ax.add_patch(circle)

            # Draw connections to next layer
            if l < len(layer_sizes) - 1:
                next_n = layer_sizes[l + 1]
                next_x = x_positions[l + 1]
                next_ys = np.linspace(0.15, 0.85, next_n)
                for ny in next_ys:
                    ax.plot([x_pos + 0.025, next_x - 0.025], [y, ny],
                            color='#cbd5e0', linewidth=0.5, alpha=0.4, zorder=1)

        ax.text(x_pos, 0.02, name, ha='center', va='center', fontsize=11,
                color=color if color != '#e2e8f0' else 'gray', fontweight='bold')

    # Add activation labels
    for i in range(len(layer_sizes) - 1):
        mid_x = (x_positions[i] + x_positions[i+1]) / 2
        label = 'ReLU' if i < len(layer_sizes) - 2 else 'Softmax'
        ax.text(mid_x, 0.93, label, ha='center', fontsize=10,
                color=SUCCESS, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=SUCCESS, alpha=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Multi-Layer Perceptron (MLP)', fontsize=15, pad=15)

    plt.tight_layout()
    plt.savefig(OUT / "mlp_architecture.png")
    plt.close()


def fig_feature_hierarchy():
    """Visualize hierarchical feature learning in deep networks."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))

    titles = ['Input\nImage', 'Layer 1\nEdges', 'Layer 2\nTextures',
              'Layer 3\nParts', 'Layer 4\nObjects']
    colors_bg = ['#f7fafc', '#dbeafe', '#c7d2fe', '#e9d5ff', '#fce7f3']

    for ax, title, bg in zip(axes, titles, colors_bg):
        ax.set_facecolor(bg)
        ax.set_title(title, fontsize=12, fontweight='bold', color=PRIMARY)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#e2e8f0')

    # Draw simple representations in each
    # Layer 0: grid pattern (raw pixels)
    ax = axes[0]
    for i in range(8):
        for j in range(8):
            c = np.random.random() * 0.5 + 0.3
            ax.add_patch(plt.Rectangle((i, j), 1, 1, facecolor=str(c), edgecolor='white', linewidth=0.5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)

    # Layer 1: edges (lines at various angles)
    ax = axes[1]
    for angle in [0, 45, 90, 135]:
        r = np.radians(angle)
        cx, cy = np.random.uniform(2, 6), np.random.uniform(2, 6)
        dx, dy = 1.5 * np.cos(r), 1.5 * np.sin(r)
        ax.plot([cx-dx, cx+dx], [cy-dy, cy+dy], color=PRIMARY, linewidth=3)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)

    # Layer 2: textures (hatching patterns)
    ax = axes[2]
    for i in range(3):
        for j in range(3):
            pattern = np.random.choice(['////', '....', '||||', 'xxxx'])
            rect = plt.Rectangle((i*2.5+0.5, j*2.5+0.5), 2, 2,
                                  facecolor='white', edgecolor=PRIMARY_LIGHT,
                                  linewidth=1.5, hatch=pattern[:2])
            ax.add_patch(rect)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)

    # Layer 3: parts (simple shapes)
    ax = axes[3]
    ax.add_patch(plt.Circle((2.5, 5.5), 1.2, facecolor='#e9d5ff', edgecolor=PURPLE, linewidth=2))
    ax.add_patch(plt.Rectangle((4.5, 1.5), 2.5, 1.5, facecolor='#e9d5ff',
                                edgecolor=PURPLE, linewidth=2, angle=10))
    ax.plot([1, 3, 2, 1], [1.5, 1.5, 3.5, 1.5], color=PURPLE, linewidth=2)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)

    # Layer 4: objects (smiley face as proxy)
    ax = axes[4]
    face = plt.Circle((4, 4), 2.5, facecolor='#fce7f3', edgecolor=ACCENT, linewidth=2.5)
    ax.add_patch(face)
    ax.plot(3, 5, 'o', color=PRIMARY, markersize=8)
    ax.plot(5, 5, 'o', color=PRIMARY, markersize=8)
    # Smile
    theta = np.linspace(np.pi + 0.3, 2*np.pi - 0.3, 30)
    ax.plot(4 + 1.2*np.cos(theta), 3.2 + 0.8*np.sin(theta), color=PRIMARY, linewidth=2.5)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)

    # Arrows between layers
    for i in range(4):
        fig.patches.append(matplotlib.patches.FancyArrowPatch(
            (0.19 + i*0.195, 0.48), (0.21 + i*0.195, 0.48),
            transform=fig.transFigure, arrowstyle='->', mutation_scale=20,
            color=SUCCESS, linewidth=2, zorder=10))

    plt.tight_layout()
    plt.savefig(OUT / "feature_hierarchy.png")
    plt.close()


def fig_weight_init():
    """Show effect of different weight initializations on activation distributions."""
    np.random.seed(42)
    n_in, n_hidden, n_layers = 784, 256, 6
    n_samples = 1000

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    inits = [
        ("Too Small ($\\sigma=0.01$)", 0.01, ACCENT),
        ("He / Kaiming ($\\sigma=\\sqrt{2/n_{in}}$)", np.sqrt(2/n_in), SUCCESS),
        ("Too Large ($\\sigma=1.0$)", 1.0, WARNING),
    ]

    for ax, (name, std, color) in zip(axes, inits):
        x = np.random.randn(n_samples, n_in)
        activations = []
        for l in range(n_layers):
            fan_in = n_in if l == 0 else n_hidden
            W = np.random.randn(fan_in, n_hidden) * std
            x = x @ W
            x = np.maximum(0, x)  # ReLU
            activations.append(x.flatten())

        for i, act in enumerate(activations):
            act_clip = act[np.abs(act) < 5]  # clip for visualization
            if len(act_clip) > 0:
                ax.hist(act_clip, bins=50, alpha=0.5, density=True,
                        label=f'Layer {i+1}', color=plt.cm.viridis(i/n_layers))

        ax.set_title(name, fontsize=12)
        ax.set_xlabel('Activation value')
        ax.set_xlim(-2, 5)
        ax.legend(fontsize=8, loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, alpha=0.15)

    axes[0].set_ylabel('Density')
    plt.suptitle('Effect of Weight Initialization on Activations (6-layer ReLU MLP, illustrative)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "weight_init_activations.png")
    plt.close()


def fig_computational_graph():
    """Draw a computational graph for a 2-layer MLP forward + backward."""
    fig, ax = plt.subplots(figsize=(14, 5))

    nodes = {
        'x':     (0.05, 0.5, 'Input\n$\\mathbf{x}$', '#e2e8f0', 'black'),
        'W1':    (0.15, 0.85, '$\\mathbf{W}_1$', PRIMARY_LIGHT, 'white'),
        'mm1':   (0.25, 0.5, '$\\times$', '#dbeafe', PRIMARY),
        'b1':    (0.25, 0.85, '$\\mathbf{b}_1$', PRIMARY_LIGHT, 'white'),
        'add1':  (0.35, 0.5, '$+$', '#dbeafe', PRIMARY),
        'relu1': (0.45, 0.5, 'ReLU', '#d1fae5', SUCCESS),
        'W2':    (0.55, 0.85, '$\\mathbf{W}_2$', PRIMARY_LIGHT, 'white'),
        'mm2':   (0.55, 0.5, '$\\times$', '#dbeafe', PRIMARY),
        'b2':    (0.65, 0.85, '$\\mathbf{b}_2$', PRIMARY_LIGHT, 'white'),
        'add2':  (0.65, 0.5, '$+$', '#dbeafe', PRIMARY),
        'y':     (0.75, 0.5, '$\\hat{\\mathbf{y}}$', '#fce7f3', ACCENT),
        'ytrue': (0.85, 0.85, '$\\mathbf{y}$', '#e2e8f0', 'black'),
        'loss':  (0.85, 0.5, '$\\mathcal{L}$', '#fef3c7', WARNING),
    }

    edges = [
        ('x', 'mm1'), ('W1', 'mm1'), ('mm1', 'add1'), ('b1', 'add1'),
        ('add1', 'relu1'), ('relu1', 'mm2'), ('W2', 'mm2'),
        ('mm2', 'add2'), ('b2', 'add2'), ('add2', 'y'),
        ('y', 'loss'), ('ytrue', 'loss'),
    ]

    # Draw edges
    for src, dst in edges:
        sx, sy = nodes[src][0], nodes[src][1]
        dx, dy = nodes[dst][0], nodes[dst][1]
        ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=1.5))

    # Draw nodes
    for name, (x, y, label, bg, tc) in nodes.items():
        circle = plt.Circle((x, y), 0.035, facecolor=bg, edgecolor='#94a3b8',
                             linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=9,
                color=tc, fontweight='bold', zorder=6)

    # Forward / backward arrows
    ax.annotate('Forward pass', xy=(0.45, 0.15), fontsize=12, color=PRIMARY,
                fontweight='bold', ha='center')
    ax.annotate('', xy=(0.65, 0.15), xytext=(0.25, 0.15),
                arrowprops=dict(arrowstyle='->', color=PRIMARY, lw=2))

    ax.annotate('Backward pass (gradients)', xy=(0.45, 0.05), fontsize=12,
                color=ACCENT, fontweight='bold', ha='center')
    ax.annotate('', xy=(0.25, 0.05), xytext=(0.65, 0.05),
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=2))

    ax.set_xlim(-0.02, 0.95)
    ax.set_ylim(-0.05, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Computational Graph: 2-Layer MLP', fontsize=15, pad=10)

    plt.tight_layout()
    plt.savefig(OUT / "computational_graph.png")
    plt.close()


def fig_train_val_test_split():
    """Visualize train/val/test split."""
    fig, ax = plt.subplots(figsize=(12, 2.5))

    # Draw the bar
    ax.barh(0, 80, left=0, height=0.6, color=PRIMARY, edgecolor='white', linewidth=2)
    ax.barh(0, 10, left=80, height=0.6, color=WARNING, edgecolor='white', linewidth=2)
    ax.barh(0, 10, left=90, height=0.6, color=ACCENT, edgecolor='white', linewidth=2)

    # Labels inside bars
    ax.text(40, 0, 'Training Set (80%)', ha='center', va='center',
            fontsize=14, color='white', fontweight='bold')
    ax.text(85, 0, 'Val\n(10%)', ha='center', va='center',
            fontsize=11, color='white', fontweight='bold')
    ax.text(95, 0, 'Test\n(10%)', ha='center', va='center',
            fontsize=11, color='white', fontweight='bold')

    # Descriptions below
    ax.text(40, -0.6, 'Learn weights\n(every epoch)', ha='center', fontsize=10, color='gray')
    ax.text(85, -0.6, 'Tune\nhyperparams', ha='center', fontsize=10, color='gray')
    ax.text(95, -0.6, 'Final eval\n(once)', ha='center', fontsize=10, color='gray')

    ax.set_xlim(-2, 102)
    ax.set_ylim(-1.2, 0.6)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUT / "train_val_test_split.png")
    plt.close()


if __name__ == "__main__":
    print("Generating Lecture 1 figures...")
    fig_activation_functions()
    print("  activation_functions.png")
    fig_activation_gradients()
    print("  activation_gradients.png")
    fig_loss_surface_3d()
    print("  loss_surface_3d.png")
    fig_loss_surface_contour()
    print("  loss_surface_contour.png")
    fig_vanishing_gradient()
    print("  vanishing_gradient.png")
    fig_training_curves()
    print("  training_curves.png")
    fig_dl_timeline()
    print("  dl_timeline.png")
    fig_mlp_architecture()
    print("  mlp_architecture.png")
    fig_feature_hierarchy()
    print("  feature_hierarchy.png")
    fig_weight_init()
    print("  weight_init_activations.png")
    fig_computational_graph()
    print("  computational_graph.png")
    fig_train_val_test_split()
    print("  train_val_test_split.png")
    print("Done: all Lecture 1 figures saved to", OUT)
