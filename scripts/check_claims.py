"""Recompute every claim in claims.yaml from the saved results and compare it with the
value printed in the paper.

Status per value:
  MATCH          equal at the printed precision (or the bound / statement holds)
  CHANGED-minor  within 5% of the printed value
  CHANGED-major  anything else, including a qualitative statement that does not hold
  MISSING        the saved results needed for the check are not there

Run:  python scripts/check_claims.py      (exit status 1 only if something is MISSING)
"""
from __future__ import annotations

import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


class Results:
    """Lazy access to results/<name>.json ("results" and "config") and results/<name>.npz."""

    def __init__(self):
        self._cache = {}

    def _payload(self, name):
        if name not in self._cache:
            self._cache[name] = json.loads((RESULTS / f"{name}.json").read_text())
        return self._cache[name]

    def __getitem__(self, name):
        return self._payload(name)["results"]

    def config(self, name):
        return self._payload(name)["config"]

    def arrays(self, name):
        with np.load(RESULTS / f"{name}.npz") as f:
            return dict(f)


def _eig(G):
    w, V = np.linalg.eigh(np.asarray(G))
    order = np.argsort(-w)
    return w[order], V[:, order]


def _sampled_tight_count(r):
    """Eigenvectors of G along which the sampled posterior s.d. is below 1/sqrt(2),
    i.e. posterior precision above 2 (lambda > 1 at w = 1)."""
    a = r.arrays("mfsir_laplace_vs_posterior")
    _, V = _eig(a["G"])
    sd = [math.sqrt(V[:, k] @ a["cov_reference"] @ V[:, k]) for k in range(V.shape[1])]
    return sum(s < 1 / math.sqrt(2) for s in sd)


def _path(r):
    return r["mfsir_calibration_path"]["checkpoints"]


def _lap(r):
    return r["mfsir_laplace_vs_posterior"]


def _fd(r):
    return r["mfsir_ggn_vs_fd_hessian"]


def _cf(r):
    return r["mfsir_counterfactual"]


def _design(r, name):
    return _cf(r)["designs"][name]


def _check(r):
    return r["mfsir_sloppy_direction_check"]


def _window(r, T):
    return next(w for w in r["netsir_observation_window"]["windows"] if w["T"] == T)


def _estimator(r, T):
    return next(w for w in r["netsir_estimator_comparison"]["windows"] if w["T"] == T)


def _points(r):
    return r["netsir_gradient_robustness"]["operating_points"]


def _reading_holds(r):
    V = np.array(r["mfsir_spectrum_loadings"]["eigvecs"])
    stiff = all(np.linalg.norm(V[:3, k]) > 0.99 for k in range(3))
    return stiff and abs(V[3, 3]) > 0.95 and abs(V[4, 4]) > 0.95


def _early_holds(w):
    V = np.abs(np.array(w["eigvecs"]))
    return int(np.argmax(V[:, -1])) == 1 and int(np.argmax(V[2, :])) < V.shape[1] - 1


def _horizon_angle(p, h):
    return next(x["angle_to_full_deg"] for x in p["horizon_truncation"] if x["grad_horizon"] == h)


def _table1_mfsir(r):
    m = r.config("mfsir_laplace_vs_posterior")["model"]
    th = m["theta_star"]
    return {"N": m["population"], "T": m["horizon"], "beta": th["beta"], "gamma": th["gamma"],
            "I0": th["I0"], "t_lock": th["t_lock"], "f_lock": th["f_lock"],
            "sigma": m["sigma_obs"], "w": m["temperature"]}


def _table1_netsir(r):
    m = r.config("netsir_gradient_robustness")["model"]
    a, b = m["operating_points"]["A"], m["operating_points"]["B"]
    p = m["prior_sd"]
    return {"N": m["population"], "degree_A": a["mean_degree"], "degree_B": b["mean_degree"],
            "seed_A": a["graph_seed"], "seed_B": b["graph_seed"],
            "T": a["horizon"] if a["horizon"] == b["horizon"] else float("nan"),
            "D": m["random_features"], "M": m["seeds"], "tau": m["gumbel_tau"],
            "kappa": m["kappa"], "kappa_init": m["kappa_init"], "prior_beta": p["beta"],
            "prior_gamma": p["gamma"], "prior_I0": p["I0"], "prior_t_lock": p["t_lock"],
            "prior_f_lock": p["f_lock"]}


def _table1_netsir_reference(r):
    ops = r.config("netsir_gradient_robustness")["model"]["operating_points"]
    return {f"{tag}_{k}": ops[tag][k] for tag in ("A", "B")
            for k in ("beta", "gamma", "I0", "t_lock", "f_lock")}


# Each extractor returns {value name: code value}, in the units printed in the paper.
EXTRACT = {
    "path_d_data_first_iterate": lambda r: {"holds": _path(r)[0]["d_data"] == _path(r)[-1]["d_data"]},
    "path_angle_at_start": lambda r: {"angle_deg": _path(r)[0]["angle_to_final_deg"]["2"]},
    "path_within_1deg_loss_ratio": lambda r: {
        "angle_deg": _path(r)[1]["angle_to_final_deg"]["2"],
        "loss_over_final": _path(r)[1]["loss"] / r["mfsir_calibration_path"]["loss_final"]},
    "path_start_displacement": lambda r: {
        "z_init": r.config("mfsir_calibration_path")["start_offset"]},
    "path_caption_panel_c": lambda r: {"caption": False},
    "laplace_spectrum": lambda r: {f"lambda_{i + 1}": v for i, v in enumerate(_lap(r)["eigvals"])},
    "laplace_d_data": lambda r: {"d_data_G": _lap(r)["d_data"],
                                 "d_data_sampled": _sampled_tight_count(r)},
    "laplace_axis_angles": lambda r: dict(zip(("axis1_deg", "axis2_deg"), _lap(r)["axis_angles_deg"])),
    "laplace_profile_error": lambda r: {"max_rel_error_pct": 100 * max(
        float(np.max(np.abs(np.array(p["energy"]) - np.array(p["prediction"]))
                     / np.abs(np.array(p["prediction"])))) for p in _lap(r)["profiles"].values())},
    "laplace_differences_only_sloppy": lambda r: {"holds": max(
        _lap(r)["marginals"], key=lambda k: _lap(r)["marginals"][k]["sd_rel_error"])
        in ("t_lock", "f_lock")},
    "laplace_caption_panel_b": lambda r: {"caption": False},
    "is_settings": lambda r: {"n_draws": r.config("mfsir_laplace_vs_posterior")["is_draws"],
                              "inflation": r.config("mfsir_laplace_vs_posterior")["is_inflation"]},
    "is_ess": lambda r: {"ess": _lap(r)["is_ess"]},
    "laplace_reference_tight_directions": lambda r: {
        "holds": _sampled_tight_count(r) == _lap(r)["d_data"] == 3},
    "laplace_cov_frobenius": lambda r: {"rel_frobenius_pct": 100 * _lap(r)["cov_rel_frobenius_error"]},
    "laplace_tlock_marginal": lambda r: {
        "rel_error_pct": 100 * _lap(r)["marginals"]["t_lock"]["sd_rel_error"]},
    "hessian_leading_angle": lambda r: {"angle_deg": max(_fd(r)["leading_angles_deg"].values())},
    "hessian_max_eig_error": lambda r: {"max_rel_error_pct": 100 * max(_fd(r)["rel_eigval_error"])},
    "hessian_cost": lambda r: {"G_jvps": _fd(r)["G_jvps"], "fd_evals": _fd(r)["fd_loss_evaluations"]},
    "hessian_fd_step": lambda r: {"h": _fd(r)["fd_step"]},
    "hessian_disagreement_sloppiest": lambda r: {"holds": int(np.argmax(
        _fd(r)["rel_eigval_error"])) == len(_fd(r)["rel_eigval_error"]) - 1},
    "reading_stiff_sloppy_parameters": lambda r: {"holds": _reading_holds(r)},
    "counterfactual_range": lambda r: {"Q_lo": _design(r, "incidence")["Q_range"]["lo"],
                                       "Q_hi": _design(r, "incidence")["Q_range"]["hi"],
                                       "Q_fit": _cf(r)["Q_fit"]},
    "counterfactual_observed_unchanged": lambda r: {"holds": max(
        abs(p["dA"]) for p in _design(r, "incidence")["profile"]
        if p["dU"] <= _cf(r)["budget_nats"]) / _cf(r)["A_fit"] < 0.01},
    "counterfactual_budget": lambda r: {"budget_nats": _cf(r)["budget_nats"]},
    "reference_peak_day": lambda r: {"peak_day": _cf(r)["reference"]["peak_day"]},
    "reference_lockdown_day": lambda r: {"lockdown_day": _cf(r)["reference"]["lockdown_day"]},
    "reference_infected_by_lockdown": lambda r: {
        "infected_pct": 100 * _cf(r)["reference"]["infected_by_lockdown_fraction"]},
    "flock_cumulative_incidence": lambda r: {
        "range_pct": 100 * _cf(r)["reference"]["f_lock_cumulative_incidence_rel_range"]},
    "compliance_design_settings": lambda r: {
        "first_day": r.config("mfsir_counterfactual")["compliance"]["first_day"],
        "last_day": r.config("mfsir_counterfactual")["compliance"]["end_day"] - 1,
        "noise_sd": r.config("mfsir_counterfactual")["compliance"]["noise_sd"]},
    "compliance_smallest_eigenvalue": lambda r: {
        "before": _design(r, "incidence")["eigvals"][-1],
        "after": _design(r, "incidence+compliance")["eigvals"][-1]},
    "compliance_d_data": lambda r: {"before": _design(r, "incidence")["d_data"],
                                    "after": _design(r, "incidence+compliance")["d_data"]},
    "compliance_range_narrowing": lambda r: {
        "width_incidence": _design(r, "incidence")["Q_range"]["width"],
        "width_compliance": _design(r, "incidence+compliance")["Q_range"]["width"],
        "factor": _cf(r)["narrowing_compliance"]},
    "prevalence_design_settings": lambda r: {
        "noise_pct_of_peak": 100 * r.config("mfsir_counterfactual")["prevalence_noise_fraction"]},
    "prevalence_spectrum_barely_changes": lambda r: {"holds": all(
        max(a / b, b / a) < 2 for a, b in zip(_design(r, "incidence")["eigvals"],
                                              _design(r, "incidence+prevalence")["eigvals"]))},
    "prevalence_range_narrowing": lambda r: {"factor": _cf(r)["narrowing_prevalence"]},
    "sloppy_check_ratios": lambda r: {
        "moments": _check(r)["ratio_stiff_over_sloppy_at_max_alpha"]["moments"],
        "acf": _check(r)["ratio_stiff_over_sloppy_at_max_alpha"]["autocorrelation"],
        "tail_quantiles": _check(r)["ratio_stiff_over_sloppy_at_max_alpha"]["tail_quantiles"]},
    "sloppy_check_unchanged": lambda r: {"holds": all(
        max(_check(r)["mean"]["sloppiest"][k]) < 0.01 * max(_check(r)["mean"]["stiffest"][k])
        for k in ("moments", "autocorrelation", "tail_quantiles"))},
    "sloppy_check_settings": lambda r: {
        "replicates": r.config("mfsir_sloppy_direction_check")["replicates"],
        "sigma": r.config("mfsir_sloppy_direction_check")["model"]["sigma_obs"],
        "noise_draws": r.config("mfsir_sloppy_direction_check")["noise_draws"]},
    "window_early_reading": lambda r: {
        "final_size_pct": 100 * _window(r, 8)["fraction_of_final_size"],
        "holds": _early_holds(_window(r, 8))},
    "window_i0_loading_rise": lambda r: {
        "first_window": r["netsir_observation_window"]["windows"][0]["I0_loading_sloppiest"],
        "last_window": r["netsir_observation_window"]["windows"][-1]["I0_loading_sloppiest"]},
    "window_late_reading": lambda r: {
        "attack_pct": 100 * _window(r, 90)["attack_rate"],
        "holds": all(abs(_window(r, 90)["eigvecs"][i][-1]) > 0.5 for i in (1, 2))},
    "window_caption_fractions": lambda r: {
        "panel_b_final_size_pct": 100 * _window(r, 8)["fraction_of_final_size"],
        "panel_c_final_size_pct": 100 * _window(r, 90)["fraction_of_final_size"]},
    "window_settings": lambda r: {
        "n_windows": len(r.config("netsir_observation_window")["windows"]),
        "first_T": r.config("netsir_observation_window")["windows"][0],
        "last_T": r.config("netsir_observation_window")["windows"][-1]},
    "estimator_sloppiest_agreement": lambda r: {"max_angle_deg": max(
        _estimator(r, T)["sloppiest_angle_deg"] for T in (8, 90))},
    "estimator_plugin_bias_fraction": lambda r: {
        "min_pct": 100 * min(_estimator(r, T)["plugin_bias_fraction_of_lambda_min"] for T in (8, 90)),
        "max_pct": 100 * max(_estimator(r, T)["plugin_bias_fraction_of_lambda_min"] for T in (8, 90))},
    "robustness_surrogate_angles": lambda r: {
        "min_deg": min(v for p in _points(r).values() for v in p["surrogate_angle_deg_by_k"].values()),
        "max_deg": max(v for p in _points(r).values() for v in p["surrogate_angle_deg_by_k"].values())},
    "robustness_gumbel_fd": lambda r: {"holds": all(
        p["per_surrogate"]["gumbel"]["fd_rel_jacobian_error_median"] < 1e-3
        for p in _points(r).values())},
    "robustness_st_fd_error": lambda r: {
        f"point_{k}_pct": 100 * p["per_surrogate"]["straight_through"]["fd_rel_jacobian_error_median"]
        for k, p in _points(r).items()},
    "robustness_eigenvalue_scale": lambda r: {"holds": all(
        abs(p["per_surrogate"]["straight_through"]["eigvals"][0]
            / p["per_surrogate"]["gumbel"]["eigvals"][0] - 1) > 0.2
        and max(p["surrogate_angle_deg_by_k"].values()) < 15 for p in _points(r).values())},
    "robustness_truncation_rotation": lambda r: {
        f"point_{k}_deg": _horizon_angle(p, 20) for k, p in _points(r).items()},
    "table1_mfsir": _table1_mfsir,
    "table1_netsir": _table1_netsir,
    "table1_netsir_reference": _table1_netsir_reference,
}


# ------------------------------------------------------------------ comparison
def _round_as_printed(x: float, printed: str) -> float:
    """Round x to the precision of the printed value: printed decimals; for integers
    without their trailing zeros; for "a.be-n" to the printed mantissa."""
    d = Decimal(printed)
    if "e" in printed.lower():
        return float(f"{x:.{len(d.as_tuple().digits) - 1}e}")
    exp = d.as_tuple().exponent
    if exp < 0:
        return round(x, -exp)
    digits = d.as_tuple().digits
    trailing = len(digits) - len(str(int(d)).rstrip("0")) if int(d) != 0 else 0
    return float(round(x, -trailing)) if trailing else float(round(x))


def status(printed, code) -> str:
    if isinstance(printed, bool):
        return "MATCH" if bool(code) == printed else "CHANGED-major"
    if code is None or (isinstance(code, float) and math.isnan(code)):
        return "MISSING"
    if printed.startswith("<="):
        bound = float(printed[2:])
        ok = _round_as_printed(code, printed[2:]) <= bound
    elif printed.startswith("<"):
        bound = float(printed[1:])
        ok = code < bound
    else:
        bound = float(printed)
        ok = _round_as_printed(code, printed) == bound
    if ok:
        return "MATCH"
    return "CHANGED-minor" if abs(code - bound) <= 0.05 * abs(bound) else "CHANGED-major"


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.4g}"


def main() -> int:
    claims = yaml.safe_load((ROOT / "claims.yaml").read_text())
    res = Results()
    rows, counts = [], {}
    order = ("MISSING", "CHANGED-major", "CHANGED-minor", "MATCH")
    for c in claims:
        try:
            code = EXTRACT[c["id"]](res)
        except FileNotFoundError:
            code = {}
        sts = []
        for k, p in c["values"].items():
            st = status(p, code[k]) if k in code else "MISSING"
            sts.append(st)
            rows.append((c["id"], k, str(p).lower() if isinstance(p, bool) else str(p),
                         _fmt(code[k]) if k in code else "-", st))
        worst = next(s for s in order if s in sts)
        counts[worst] = counts.get(worst, 0) + 1

    head = ("claim", "value", "paper", "code", "status")
    w = [max(len(r[i]) for r in rows + [head]) for i in range(5)]
    line = lambda r: "  ".join(s.ljust(n) for s, n in zip(r, w)).rstrip()
    print(line(head))
    print(line(["-" * n for n in w]))
    for r in rows:
        print(line(r))
    print("\nclaims: " + ", ".join(f"{s} {counts[s]}" for s in order if s in counts))
    return 1 if "MISSING" in counts else 0


if __name__ == "__main__":
    sys.exit(main())
