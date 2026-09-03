#!/usr/bin/env python3
"""Deterministic float64 attention/learning evidence for the L11M bank example.

Run with an existing CPU PyTorch environment:
    python bank_training.py --check --json bank-training-numbers.json

All parameters are hand-chosen initial values, not a trained language model.
Attention sends a message h, W_O maps it to delta_e, and the residual addition
e_new = e_last + delta_e preserves the token's embedding before prediction.
X stacks the sequence's embedding rows; it is not the lookup table E. Without
positions its rows are e_i = E[t_i]. With positions, its rows are
e_i^(0) = e_i + r_i, and X_new stacks e_i^(1) after the contextual update.
Calculations never round intermediate tensors. JSON preserves float64 values;
round only when displaying them on slides. The head is one affine layer (the
smallest prediction head), so every logit and derivative is inspectable.

JSON schema compatibility: legacy evidence keys "x_new" and "delta" denote
the single-token e' and Delta e; "X_new" and "Delta" denote the stacked X'
and Delta X. "X" always means the sequence matrix; it does not rename a
single token's embedding.
These exported key spellings and legacy metadata formulas are retained so
existing slide/data consumers do not break; the arithmetic is unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


DTYPE = torch.float64
VOCAB = ["She", "deposited", "money", "in", "the", "bank", ".", "and"]
PREFIX = list(range(6))
GOLD = 6
ETA = 0.1
SCALE = math.sqrt(3.0)


def initial_parameters(*, positions: bool = False) -> dict[str, torch.Tensor]:
    """Independent trainable tensors, so each experiment starts identically."""
    params = {
        "E": torch.tensor(
            [[0, 0, 1], [1, 0, 1], [2, 0, 0], [0, 1, 0],
             [0, 0, 0], [1, 1, 0], [0, 0, 0.5], [0.5, 1, 1]],
            dtype=DTYPE,
        ),
        "W_Q": torch.diag(torch.tensor([1, 0, 0], dtype=DTYPE)),
        "W_K": torch.eye(3, dtype=DTYPE),
        "W_V": torch.diag(torch.tensor([1, 2, 1], dtype=DTYPE)),
        "W_O": torch.eye(3, dtype=DTYPE),
        "U": torch.zeros(3, len(VOCAB), dtype=DTYPE),
        "b": torch.zeros(len(VOCAB), dtype=DTYPE),
    }
    params["U"][:, GOLD] = torch.tensor([1, 0, 0], dtype=DTYPE)
    params["U"][:, 7] = torch.tensor([0, 1, 0], dtype=DTYPE)
    if positions:
        # Seven positions permit the six-input training example and one append.
        params["R"] = torch.zeros(7, 3, dtype=DTYPE)
    return {name: value.requires_grad_() for name, value in params.items()}


def states(params: dict[str, torch.Tensor], token_ids: list[int]) -> torch.Tensor:
    """Stack e_i rows, or e_i^(0) = e_i + r_i when positions are present."""
    X = params["E"][token_ids]
    if "R" in params:
        X = X + params["R"][:len(token_ids)]
    return X


def retain_tensors(values: dict) -> dict:
    for value in values.values():
        if isinstance(value, torch.Tensor) and value.requires_grad:
            value.retain_grad()
    return values


def single_query(
    params: dict[str, torch.Tensor], token_ids: list[int] = PREFIX,
    *, scale: float = SCALE, gold: int | None = GOLD,
) -> dict:
    """Last input position predicts the next token; every input is visible."""
    X = states(params, token_ids)
    Q, K, V = X @ params["W_Q"], X @ params["W_K"], X @ params["W_V"]
    q = Q[-1]
    dot_products = q @ K.T
    scores = dot_products / scale
    alpha = scores.softmax(dim=-1)
    h = alpha @ V
    delta_e = h @ params["W_O"]
    e_new = X[-1] + delta_e
    logits = e_new @ params["U"] + params["b"]
    probabilities = logits.softmax(dim=-1)
    out = {
        "token_ids": token_ids, "tokens": [VOCAB[j] for j in token_ids],
        "prediction_position": len(token_ids), "scale": scale,
        "X": X, "Q": Q, "q": q, "K": K, "V": V,
        "dot_products": dot_products, "scores": scores, "alpha": alpha,
        "h": h, "delta": delta_e, "x_new": e_new,
        "logits": logits, "probabilities": probabilities,
    }
    if gold is not None:
        out["gold_id"] = gold
        out["gold_token"] = VOCAB[gold]
        out["loss"] = F.cross_entropy(logits.unsqueeze(0), torch.tensor([gold]))
    return retain_tensors(out)


def full_causal(params: dict[str, torch.Tensor]) -> dict:
    """Six parallel next-token predictions with shifted targets and mean CE."""
    target_ids = [1, 2, 3, 4, 5, GOLD]
    X = states(params, PREFIX)
    Q, K, V = X @ params["W_Q"], X @ params["W_K"], X @ params["W_V"]
    S = Q @ K.T / SCALE
    M_forbidden = torch.ones(6, 6, dtype=torch.bool).triu(diagonal=1)
    S_masked = S.masked_fill(M_forbidden, -torch.inf)
    A = S_masked.softmax(dim=-1)
    H = A @ V
    Delta_X = H @ params["W_O"]
    X_new = X + Delta_X
    Z = X_new @ params["U"] + params["b"]
    P = Z.softmax(dim=-1)
    losses = F.cross_entropy(Z, torch.tensor(target_ids), reduction="none")
    return retain_tensors({
        "token_ids": PREFIX, "tokens": [VOCAB[j] for j in PREFIX],
        "target_ids": target_ids, "targets": [VOCAB[j] for j in target_ids],
        "X": X, "Q": Q, "K": K, "V": V, "scale": SCALE,
        "raw_scores": S, "forbidden_mask": M_forbidden,
        "masked_scores": S_masked, "alpha": A, "H": H,
        "Delta": Delta_X, "X_new": X_new,
        "logits": Z, "probabilities": P,
        "loss_per_position": losses, "mean_loss": losses.mean(),
    })


def manual_single_backward(params: dict[str, torch.Tensor], forward: dict) -> dict:
    """Explicit reverse-mode chain rule, independent of torch.autograd."""
    detached = {name: value.detach() for name, value in params.items()}
    X, q, K, V, alpha, h, e_new, p = [forward[name].detach()
        for name in ("X", "q", "K", "V", "alpha", "h", "x_new", "probabilities")]
    g_logits = p.clone()
    g_logits[forward["gold_id"]] -= 1
    g_e_new = g_logits @ detached["U"].T
    g_delta_e = g_e_new
    g_h = g_delta_e @ detached["W_O"].T

    # h = alpha @ V has TWO branches: supplied information and its weights.
    g_V = torch.outer(alpha, g_h)
    g_alpha = V @ g_h
    g_scores = alpha * (g_alpha - torch.dot(alpha, g_alpha))
    g_dot_products = g_scores / forward["scale"]
    g_q = g_dot_products @ K
    g_K = torch.outer(g_dot_products, q)
    g_Q = torch.zeros_like(X)
    g_Q[-1] = g_q

    g_X_values = g_V @ detached["W_V"].T
    g_X_keys = g_K @ detached["W_K"].T
    g_X_query = g_Q @ detached["W_Q"].T
    # e_new = X[-1] + delta_e also returns a direct residual gradient.
    g_X_residual = torch.zeros_like(X)
    g_X_residual[-1] = g_e_new
    g_X = g_X_values + g_X_keys + g_X_query + g_X_residual
    g_E = torch.zeros_like(detached["E"])
    # Repeated token IDs would accumulate into the same embedding-table row.
    g_E.index_add_(0, torch.tensor(forward["token_ids"]), g_X)
    parameter_gradients = {
        "E": g_E,
        "W_Q": X.T @ g_Q,
        "W_K": X.T @ g_K,
        "W_V": X.T @ g_V,
        "W_O": torch.outer(h, g_delta_e),
        "U": torch.outer(e_new, g_logits),
        "b": g_logits,
    }
    if "R" in params:
        g_R = torch.zeros_like(detached["R"])
        g_R[:len(forward["token_ids"])] = g_X
        parameter_gradients["R"] = g_R
    return {
        "activations": {
            "logits": g_logits, "x_new": g_e_new, "delta": g_delta_e,
            "h": g_h, "alpha": g_alpha,
            "scores": g_scores, "dot_products": g_dot_products,
            "q": g_q, "Q": g_Q, "K": g_K, "V": g_V, "X": g_X,
        },
        "paths_into_X": {
            "value_path": g_X_values,
            "matching_key_path": g_X_keys,
            "matching_query_path": g_X_query,
            "residual_path": g_X_residual,
            "sum": g_X,
        },
        "parameters": parameter_gradients,
    }


def gradients(tensors: dict) -> dict:
    return {name: tensor.grad.detach().clone()
            for name, tensor in tensors.items()
            if isinstance(tensor, torch.Tensor) and tensor.grad is not None}


def snapshot(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in params.items()}


def sgd_step(params: dict[str, torch.Tensor], eta: float = ETA) -> None:
    """Update every trainable parameter once; no optimizer state or momentum."""
    with torch.no_grad():
        for name, value in params.items():
            assert value.grad is not None, f"Missing gradient: {name}"
            value -= eta * value.grad


def finite_difference(parameter: str, index: tuple[int, ...]) -> dict:
    epsilon = 1e-6
    initial = initial_parameters()
    loss = single_query(initial)["loss"]
    auto = torch.autograd.grad(loss, initial[parameter])[0][index].item()
    losses = []
    for shift in (epsilon, -epsilon):
        params = initial_parameters()
        with torch.no_grad():
            params[parameter][index] += shift
        losses.append(single_query(params)["loss"].item())
    numeric = (losses[0] - losses[1]) / (2 * epsilon)
    assert math.isclose(auto, numeric, rel_tol=1e-7, abs_tol=1e-9)
    return {"parameter": parameter, "zero_based_index": index, "epsilon": epsilon,
            "loss_plus": losses[0], "loss_minus": losses[1],
            "autograd": auto, "central_difference": numeric,
            "absolute_error": abs(auto - numeric)}


def separation_checks() -> dict:
    """Matching chooses sources; values carry messages; neither edits E."""
    params = initial_parameters()
    untouched = snapshot(params)
    reference = single_query(params, gold=None)
    for name, value in params.items():
        torch.testing.assert_close(value.detach(), untouched[name], rtol=0, atol=0)

    # Change only what sources send, while keeping their matching fixed.
    alternate = initial_parameters()
    with torch.no_grad():
        alternate["W_V"][:, 0] = 0
    different_values = single_query(alternate, gold=None)
    for name in ("X", "Q", "K", "scores", "alpha"):
        torch.testing.assert_close(reference[name], different_values[name], rtol=0, atol=0)
    for name in ("V", "h", "delta", "x_new", "logits", "probabilities"):
        assert not torch.allclose(reference[name], different_values[name]), name
    assert different_values["h"][0].item() == 0
    torch.testing.assert_close(reference["h"][1:], different_values["h"][1:], rtol=0, atol=0)

    # Zero messages do not erase the original state: the skip path remains.
    silent = initial_parameters()
    with torch.no_grad():
        silent["W_V"].zero_()
    no_message = single_query(silent, gold=None)
    torch.testing.assert_close(no_message["delta"], torch.zeros(3, dtype=DTYPE), rtol=0, atol=0)
    torch.testing.assert_close(no_message["x_new"], no_message["X"][-1], rtol=0, atol=0)
    no_messages_causal = full_causal(silent)
    torch.testing.assert_close(no_messages_causal["X_new"], no_messages_causal["X"], rtol=0, atol=0)

    return {
        "reference": {name: reference[name] for name in ("alpha", "h", "delta", "x_new", "probabilities")},
        "changed_W_V": alternate["W_V"],
        "changed_values": {name: different_values[name] for name in ("alpha", "h", "delta", "x_new", "probabilities")},
        "zero_message_x_new": no_message["x_new"],
        "forward_does_not_mutate_parameters_or_embedding_table": True,
        "changing_values_preserves_matching_but_changes_output": True,
        "zero_message_retains_current_state": True,
    }


def serializable(value):
    if isinstance(value, torch.Tensor):
        return serializable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "-inf" if value < 0 else "inf" if value > 0 else "nan"
    return value


def build_evidence() -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    params = initial_parameters()
    before_params = snapshot(params)
    forward = single_query(params)
    manual = manual_single_backward(params, forward)
    forward["loss"].backward()
    autograd_params = gradients(params)
    autograd_activations = gradients(forward)
    errors = {}
    for name, expected in manual["parameters"].items():
        torch.testing.assert_close(autograd_params[name], expected, rtol=1e-12, atol=1e-12)
        errors[name] = (autograd_params[name] - expected).abs().max().item()
    for name, expected in manual["activations"].items():
        torch.testing.assert_close(autograd_activations[name], expected, rtol=1e-12, atol=1e-12)
    for branch in ("value_path", "matching_key_path", "matching_query_path", "residual_path"):
        assert torch.count_nonzero(manual["paths_into_X"][branch]) > 0
    assert all(torch.isfinite(gradient).all() for gradient in autograd_params.values())
    sgd_step(params)
    after_params = snapshot(params)
    after = single_query(params)
    assert after["loss"].item() < forward["loss"].item()
    assert after["probabilities"][GOLD].item() > forward["probabilities"][GOLD].item()
    for name in params:
        torch.testing.assert_close(after_params[name], before_params[name] - ETA * autograd_params[name])

    # Historical unscaled hand calculation, with explicitly rounded weights.
    unscaled = single_query(initial_parameters(), scale=1.0)
    torch.testing.assert_close(unscaled["dot_products"], torch.tensor([0, 1, 2, 0, 0, 1], dtype=DTYPE))
    rounded = torch.round(unscaled["alpha"] * 1000) / 1000
    expected_rounded = torch.tensor([.063, .172, .467, .063, .063, .172], dtype=DTYPE)
    torch.testing.assert_close(rounded, expected_rounded, rtol=0, atol=1e-12)
    rounded_h = rounded @ unscaled["V"]
    torch.testing.assert_close(rounded_h, torch.tensor([1.278, .470, .235], dtype=DTYPE), rtol=0, atol=1e-12)

    # Generation is a fresh BEFORE-UPDATE model. Select a possible non-argmax
    # outcome for illustration; do not misreport it as a seeded random sample.
    generation_params = initial_parameters()
    generation_initial = snapshot(generation_params)
    generation_before = single_query(generation_params, gold=None)
    chosen_id = VOCAB.index("and")
    assert generation_before["probabilities"][chosen_id] > 0
    assert generation_before["probabilities"].argmax().item() != chosen_id
    generation_after = single_query(generation_params, PREFIX + [chosen_id], gold=None)
    for name, value in generation_params.items():
        torch.testing.assert_close(value.detach(), generation_initial[name], rtol=0, atol=0)

    # A SEPARATE experiment: all six causal rows, shifted targets, learned R.
    causal_params = initial_parameters(positions=True)
    causal_initial = snapshot(causal_params)
    causal_before = full_causal(causal_params)
    for matrix_name, single_name in (("Q", "q"), ("alpha", "alpha"), ("H", "h"),
                                    ("Delta", "delta"), ("X_new", "x_new"),
                                    ("logits", "logits"), ("probabilities", "probabilities")):
        torch.testing.assert_close(causal_before[matrix_name][-1], forward[single_name])
    torch.testing.assert_close(causal_before["alpha"].sum(dim=-1), torch.ones(6, dtype=DTYPE))
    assert torch.count_nonzero(causal_before["alpha"].triu(diagonal=1)) == 0
    causal_before["mean_loss"].backward()
    causal_parameter_gradients = gradients(causal_params)
    causal_activation_gradients = gradients(causal_before)
    assert torch.count_nonzero(causal_parameter_gradients["R"][:6]) > 0
    torch.testing.assert_close(causal_parameter_gradients["R"][:6], causal_activation_gradients["X"])
    assert torch.count_nonzero(causal_parameter_gradients["R"][6]) == 0
    assert torch.count_nonzero(causal_activation_gradients["raw_scores"].triu(diagonal=1)) == 0
    sgd_step(causal_params)
    for name, value in causal_params.items():
        torch.testing.assert_close(value.detach(), causal_initial[name] - ETA * causal_parameter_gradients[name])
    causal_after = full_causal(causal_params)
    assert causal_after["mean_loss"].item() < causal_before["mean_loss"].item()

    evidence = {
        "metadata": {
            "dtype": "float64", "device": "cpu", "torch_version": torch.__version__,
            "vocabulary": VOCAB, "vocabulary_ids_are_zero_based": True,
            "positions_in_slides_are_one_based": True, "eta": ETA,
            "score_scaling": "q @ K.T / sqrt(3)",
            "residual": "delta = h @ W_O; x_new = X[-1] + delta; W_O initially identity",
            "head": "logits = x_new @ U + b (one affine prediction head)",
            "initialization": "hand-chosen toy parameters; no pretraining",
            "rounding": "No rounding in forward/backward graph; round only displayed numbers.",
            "experiments": "Single-query update and full-causal update each start from fresh parameters.",
        },
        "single_query": {
            "initial_parameters": before_params, "before": forward,
            "backward_manual": manual,
            "backward_autograd": {"activations": autograd_activations, "parameters": autograd_params},
            "manual_autograd_max_absolute_errors": errors,
            "updated_parameters": after_params, "after": after,
        },
        "finite_difference_check": finite_difference("W_Q", (0, 0)),
        "finite_difference_checks": [finite_difference(name, index) for name, index in
                                     (("W_Q", (0, 0)), ("W_K", (0, 0)),
                                      ("W_V", (0, 0)), ("W_O", (0, 0)),
                                      ("E", (5, 0)), ("U", (0, GOLD)), ("b", (GOLD,)))],
        "role_separation": separation_checks(),
        "unscaled_reproduction": {
            "dot_products": unscaled["dot_products"], "alpha": unscaled["alpha"],
            "h_unrounded": unscaled["h"], "rounded_alpha": rounded,
            "h_using_rounded_alpha": rounded_h,
        },
        "generation_before_update": {
            "prefix_forward": generation_before,
            "selection_method": "Explicitly selected illustrative possible draw; not RNG output and not argmax.",
            "selected_id": chosen_id, "selected_token": VOCAB[chosen_id],
            "selected_probability": generation_before["probabilities"][chosen_id],
            "argmax_token": VOCAB[generation_before["probabilities"].argmax().item()],
            "appended_embedding": generation_params["E"][chosen_id],
            "appended_forward": generation_after,
            "parameters_updated_during_generation": False,
        },
        "full_causal_training": {
            "initial_parameters": causal_initial, "before": causal_before,
            "backward_autograd": {"activations": causal_activation_gradients,
                                  "parameters": causal_parameter_gradients},
            "updated_parameters": snapshot(causal_params), "after": causal_after,
            "position_gradient_nonzero_entries": torch.count_nonzero(causal_parameter_gradients["R"]).item(),
        },
        "checks": {
            "manual_backward_matches_autograd": True,
            "finite_differences_cover_all_single_query_parameters": True,
            "residual_gradient_path_is_nonzero": True,
            "forward_does_not_mutate_embedding_table": True,
            "changing_W_V_preserves_alpha_and_changes_message_and_output": True,
            "zero_message_retains_current_state": True,
            "generation_parameters_remain_fixed": True,
            "full_causal_last_row_matches_single_query": True,
            "single_query_loss_decreases": True,
            "single_query_gold_probability_increases": True,
            "unscaled_example_reproduced": True,
            "future_attention_weights_are_zero": True,
            "full_causal_mean_loss_decreases": True,
            "learned_positions_receive_nonzero_gradients": True,
        },
    }
    return serializable(evidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run assertions (also run by default).")
    parser.add_argument("--json", type=Path, help="Write complete numerical evidence as JSON.")
    args = parser.parse_args()
    result = build_evidence()
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    single, causal, generation = result["single_query"], result["full_causal_training"], result["generation_before_update"]
    print("PASS: residual manual/autograd, finite differences, role separation, causal mask, fixed generation, SGD, and unscaled checks")
    print("scaled alpha =", [round(v, 6) for v in single["before"]["alpha"]])
    print("scaled h6 =", [round(v, 6) for v in single["before"]["h"]])
    print("contextual embedding e_new6 =", [round(v, 6) for v in single["before"]["x_new"]])
    print(f"single: p(.) {single['before']['probabilities'][GOLD]:.6f} -> {single['after']['probabilities'][GOLD]:.6f}; "
          f"loss {single['before']['loss']:.6f} -> {single['after']['loss']:.6f}")
    print("g_h6 =", [round(v, 6) for v in single["backward_manual"]["activations"]["h"]])
    print(f"full causal: mean loss {causal['before']['mean_loss']:.6f} -> {causal['after']['mean_loss']:.6f}; "
          f"p6(.) {causal['before']['probabilities'][-1][GOLD]:.6f} -> {causal['after']['probabilities'][-1][GOLD]:.6f}")
    print(f"illustrative append 'and' (p={generation['selected_probability']:.6f}), "
          f"new q7={generation['appended_forward']['q']}, h7="
          f"{[round(v, 6) for v in generation['appended_forward']['h']]}")
    if args.json:
        print(f"JSON: {args.json.resolve()}")


if __name__ == "__main__":
    main()
