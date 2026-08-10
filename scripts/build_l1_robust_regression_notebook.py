#!/usr/bin/env python3
"""Build the compact, executed robust-linear-regression companion notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/L01/03_robust_linear_regression.ipynb"


def md(source: str):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str, purpose: str):
    return nbformat.v4.new_code_cell(
        textwrap.dedent(source).strip(), metadata={"purpose": purpose}
    )


def build_notebook():
    cells = [
        md(
            r"""
            # Robust linear regression: which observation model should we use?

            A straight line can be trained with several probability models. The model determines how a residual
            $r_i=y_i-\hat y_i$ is penalized:

            - **Gaussian noise** $\Rightarrow$ squared error (MSE),
            - **Laplace noise** $\Rightarrow$ absolute error (MAE),
            - **Student-t noise** $\Rightarrow$ a heavy-tailed robust loss.

            We will fit the **same data** with all three and watch two outliers change the answer.
            """
        ),
        code(
            """
            import math
            import torch
            import torch.distributions as D
            import matplotlib.pyplot as plt

            %config InlineBackend.figure_format = 'retina'
            torch.set_default_dtype(torch.float64)
            _ = torch.manual_seed(7)
            """,
            "Import PyTorch and configure reproducible retina figures",
        ),
        code(
            """
            INK = "#17343b"
            MUTED = "#72858a"
            ORANGE = "#ef7d00"
            BLUE = "#2f6fbb"
            TEAL = "#238b8e"
            GREEN = "#20a647"
            RED = "#d9485f"

            plt.rcParams.update({
                "figure.dpi": 150,
                "savefig.dpi": 240,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.edgecolor": INK,
                "axes.labelcolor": INK,
                "text.color": INK,
                "xtick.color": INK,
                "ytick.color": INK,
                "font.size": 11,
                "legend.frameon": False,
                "lines.linewidth": 2.3,
            })
            """,
            "Define the course palette and plotting defaults",
        ),
        md(
            r"""
            ## 1. Make one simple regression dataset

            The underlying relationship is $y=1+2x$. Most observations have small Gaussian noise. We then replace
            two measurements by obvious recording errors.
            """
        ),
        code(
            """
            x = torch.linspace(-2.0, 2.0, 25)
            true_intercept = torch.tensor(1.0)
            true_slope = torch.tensor(2.0)
            noise_scale = torch.tensor(0.35)

            y_mean = true_intercept + true_slope * x
            y_clean = y_mean + D.Normal(0.0, noise_scale).sample((x.numel(),))
            """,
            "Generate clean observations from the known straight line",
        ),
        code(
            """
            outlier_index = torch.tensor([4, 20])
            y = y_clean.clone()
            y[outlier_index] += torch.tensor([5.0, -5.0])

            print("true parameters: intercept = 1, slope = 2")
            print("outlier indices:", outlier_index.tolist())
            """,
            "Insert two visible outliers",
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(8.2, 4.2))
            ax.scatter(x, y, s=34, color=INK, label="observed data", zorder=3)
            ax.scatter(x[outlier_index], y[outlier_index], s=90, facecolor="none",
                       edgecolor=RED, linewidth=2, label="two outliers", zorder=4)
            ax.plot(x, y_mean, color=MUTED, linestyle="--", label=r"true line $1+2x$")
            ax.set(xlabel="input x", ylabel="target y", title="The same line, plus two unusual measurements")
            ax.grid(alpha=0.25)
            ax.legend(ncols=3, loc="upper center")
            plt.tight_layout()
            plt.show()
            """,
            "Plot the data and mark the outliers",
        ),
        md(
            r"""
            ## 2. Ask what residuals each model considers plausible

            A residual is $r=y-\hat y$. Every model puts its highest density near $r=0$, but their tails differ.
            A heavy-tailed model leaves appreciable density far from zero, so an unusual measurement is possible
            without forcing the fitted line to chase it.
            """
        ),
        code(
            """
            residual = torch.linspace(-6.0, 6.0, 1201)
            zero = torch.tensor(0.0)

            residual_models = {
                "Gaussian → MSE": (D.Normal(zero, 1.0), BLUE),
                "Laplace → MAE": (D.Laplace(zero, 1.0), ORANGE),
                "Student-t (df=3)": (D.StudentT(3.0, zero, 1.0), TEAL),
            }
            """,
            "Create the three zero-centred residual distributions",
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(8.2, 4.2))
            for name, (distribution, color) in residual_models.items():
                ax.plot(residual, distribution.log_prob(residual).exp(), color=color, label=name)

            ax.set(xlabel=r"residual $r=y-\hat y$", ylabel=r"density $p(r)$",
                   title="Heavy tails retain more density far from zero")
            ax.axvspan(-1, 1, color=MUTED, alpha=0.08, label="small residuals")
            ax.grid(alpha=0.25)
            ax.legend()
            plt.tight_layout()
            plt.show()
            """,
            "Plot the probability density of the same residual under each model",
        ),
        code(
            """
            far_residual = torch.tensor(4.0)
            for name, (distribution, _) in residual_models.items():
                density = distribution.log_prob(far_residual).exp()
                print(f"{name:18s}: p(r=4) = {density:.6f}")
            """,
            "Compare how much tail density the three models assign to one large residual",
        ),
        md(
            r"""
            At $r=4$, Laplace and Student-t assign roughly **68 times** the density assigned by a standard Gaussian.
            This does not make the point *good*; it makes the point less astonishing under the assumed noise process.

            ## 3. Convert density into a training loss

            Maximum likelihood minimizes $-\log p(r)$. Therefore **more tail density means a smaller penalty** for a
            large residual. Fix the scale at $s=1$ below and subtract the loss at $r=0$, so only the shape remains.
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(8.2, 4.2))
            for name, (distribution, color) in residual_models.items():
                loss = -distribution.log_prob(residual)
                loss -= -distribution.log_prob(zero)
                ax.plot(residual, loss, color=color, label=name)

            ax.set(xlabel=r"residual $r=y-\hat y$", ylabel="extra negative log-likelihood",
                   title="Large residuals receive very different penalties", ylim=(-0.1, 10))
            ax.grid(alpha=0.25)
            ax.legend()
            plt.tight_layout()
            plt.show()
            """,
            "Plot Gaussian, Laplace, and Student-t residual losses",
        ),
        md(
            r"""
            For fixed scale, constants do not affect the best-fitting line:

            $$
            -\log \mathcal N(y\mid\mu,s^2)=\frac{(y-\mu)^2}{2s^2}+C
            \quad\Longrightarrow\quad \text{MSE},
            $$

            $$
            -\log \operatorname{Laplace}(y\mid\mu,b)=\frac{|y-\mu|}{b}+C
            \quad\Longrightarrow\quad \text{MAE}.
            $$

            Student-t grows only logarithmically for very large residuals, so one extreme point cannot dominate the
            whole fit as easily.

            The optimizer responds to the *slope* of these losses:

            - Gaussian: doubling a large residual roughly quadruples its extra loss, and its pull keeps growing.
            - Laplace: doubling a residual doubles its extra loss, so its pull is capped at a constant magnitude.
            - Student-t: the pull eventually decreases as a residual becomes extreme.

            That is the practical meaning of robustness here: an outlier is retained in the dataset, but it cannot
            dominate the update merely because it is far from the current line.
            """
        ),
        code(
            """
            r = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
            normal_extra = -D.Normal(0.0, 1.0).log_prob(r) + D.Normal(0.0, 1.0).log_prob(torch.tensor(0.0))
            laplace_extra = -D.Laplace(0.0, 1.0).log_prob(r) + D.Laplace(0.0, 1.0).log_prob(torch.tensor(0.0))

            print("Gaussian extra NLL:", normal_extra.tolist())
            print("0.5 × residual²:   ", (0.5 * r.square()).tolist())
            print("Laplace extra NLL:", laplace_extra.tolist())
            print("absolute residual:", r.abs().tolist())
            """,
            "Verify the MSE and MAE identities numerically",
        ),
        md(
            """
            ## 4. Fit the same line three times

            The prediction rule is always $\mu_i=b+wx_i$. Only the observation distribution—and therefore the
            negative log-likelihood—changes.
            """
        ),
        code(
            """
            fit_scale = torch.tensor(0.35)

            def observation_model(name, mean):
                if name == "Gaussian":
                    return D.Normal(mean, fit_scale)
                if name == "Laplace":
                    return D.Laplace(mean, fit_scale)
                if name == "Student-t":
                    return D.StudentT(3.0, mean, fit_scale)
                raise ValueError(name)
            """,
            "Define the three observation models with one shared scale",
        ),
        code(
            """
            def fit_line(model_name, target, steps=1200):
                theta = torch.zeros(2, requires_grad=True)  # [intercept, slope]
                optimizer = torch.optim.Adam([theta], lr=0.035)

                for _ in range(steps):
                    mean = theta[0] + theta[1] * x
                    loss = -observation_model(model_name, mean).log_prob(target).sum()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                return theta.detach(), loss.detach()
            """,
            "Define a short maximum-likelihood optimization loop",
        ),
        code(
            """
            fits = {}
            for model_name in ["Gaussian", "Laplace", "Student-t"]:
                theta_hat, final_nll = fit_line(model_name, y)
                fits[model_name] = theta_hat
                print(f"{model_name:9s}: intercept={theta_hat[0]: .3f}, "
                      f"slope={theta_hat[1]: .3f}, summed NLL={final_nll: .2f}")
            """,
            "Fit all three models and print their parameter estimates",
        ),
        code(
            """
            colors = {"Gaussian": BLUE, "Laplace": ORANGE, "Student-t": TEAL}
            x_line = torch.linspace(-2.15, 2.15, 300)

            fig, ax = plt.subplots(figsize=(8.4, 4.5))
            ax.scatter(x, y, s=30, color=INK, alpha=0.75, label="observed data", zorder=3)
            ax.scatter(x[outlier_index], y[outlier_index], s=90, facecolor="none",
                       edgecolor=RED, linewidth=2, label="outliers", zorder=4)
            ax.plot(x_line, true_intercept + true_slope * x_line, color=MUTED,
                    linestyle="--", label="true line")
            """,
            "Draw the observations and true line for the fit comparison",
        ),
        code(
            """
            for model_name, theta_hat in fits.items():
                prediction = theta_hat[0] + theta_hat[1] * x_line
                ax.plot(x_line, prediction, color=colors[model_name], label=model_name)

            ax.set(xlabel="input x", ylabel="target y",
                   title="Gaussian bends toward the outliers; heavy-tailed fits resist them")
            ax.grid(alpha=0.25)
            ax.legend(ncols=3, loc="upper center")
            plt.tight_layout()
            plt.show()
            """,
            "Add the three fitted lines and display the comparison",
        ),
        md(
            """
            ## 5. Which observations control each fit?

            For every fitted model, compute each point's penalty relative to a perfect residual of zero. A tall bar
            means that observation has a large effect on the summed training objective.
            """
        ),
        code(
            """
            point_penalties = {}
            for model_name, theta_hat in fits.items():
                mean = theta_hat[0] + theta_hat[1] * x
                model = observation_model(model_name, mean)
                perfect = observation_model(model_name, y)
                point_penalties[model_name] = -model.log_prob(y) + perfect.log_prob(y)
            """,
            "Compute each observation's contribution above a perfect fit",
        ),
        code(
            """
            fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True)
            for ax, model_name in zip(axes, ["Gaussian", "Laplace", "Student-t"]):
                bars = ax.bar(torch.arange(x.numel()), point_penalties[model_name],
                              color=colors[model_name], alpha=0.85)
                for index in outlier_index.tolist():
                    bars[index].set_color(RED)
                ax.set(title=model_name, xlabel="observation index")
                ax.grid(axis="y", alpha=0.25)

            axes[0].set_ylabel("per-point loss above a perfect fit")
            fig.suptitle("Red bars are the same two outliers in every model", fontweight="bold")
            fig.tight_layout()
            plt.show()
            """,
            "Plot per-observation loss contributions",
        ),
        md(
            """
            ## 6. A practical decision rule

            | Observation model | Equivalent loss | When it is a useful starting point |
            |---|---|---|
            | Gaussian | MSE | residuals are light-tailed; large errors really should be punished strongly |
            | Laplace | MAE | occasional large errors are expected; a constant-magnitude correction is desirable |
            | Student-t | heavy-tailed NLL | gross outliers may occur and should be discounted smoothly rather than discarded |

            **Takeaway.** A loss function is not merely a numerical preference. It states what kinds of measurement
            errors we believe the data-generating process can produce. Inspect residuals, compare fits, and validate on
            held-out data; robustness is an assumption to test, not an automatic guarantee.
            """
        ),
    ]

    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "title": "Robust linear regression: Gaussian, Laplace, and Student-t losses",
        }
    )
    return notebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="run all cells before writing")
    args = parser.parse_args()

    notebook = build_notebook()
    if args.execute:
        NotebookClient(
            notebook,
            timeout=240,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, OUTPUT)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
