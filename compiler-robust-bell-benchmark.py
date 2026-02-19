"""
bell_benchmark.py — Bell-state fidelity benchmark (Qiskit Aer).

Provides:
- prepare_bell_state: build one of four Bell states on 2 qubits
- add_pauli_measurement: measure ZZ / XX / YY correlators via basis changes
- make_noise_model: depolarizing + thermal relaxation (+ optional readout error)
- run_benchmark: sweep noise, run circuits, save CSV + plot
"""

import math
from typing import Dict, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)


# ── Bell-state stabilizer signs ───────────────────────────────────────────────
# |B><B| = 1/4 (II + sXX*XX + sYY*YY + sZZ*ZZ)
# F = (1 + sZZ<EZZ> + sXX<EXX> + sYY<EYY>) / 4

BELL_SIGNS: Dict[str, Dict[str, int]] = {
    "phi_plus":  {"XX": +1, "YY": -1, "ZZ": +1},
    "phi_minus": {"XX": -1, "YY": +1, "ZZ": +1},
    "psi_plus":  {"XX": +1, "YY": +1, "ZZ": -1},
    "psi_minus": {"XX": -1, "YY": -1, "ZZ": -1},
}


# ── Circuit construction ─────────────────────────────────────────────────────

def prepare_bell_state(state: str) -> QuantumCircuit:
    """Prepare one of the 4 Bell states on 2 qubits (no measurement)."""
    if state not in BELL_SIGNS:
        raise ValueError(f"Unknown Bell state: {state}")

    qc = QuantumCircuit(2, name=f"prep_{state}")
    qc.h(0)
    qc.cx(0, 1)

    if state == "phi_minus":
        qc.z(0)
    elif state == "psi_plus":
        qc.x(1)
    elif state == "psi_minus":
        qc.x(1)
        qc.z(0)

    return qc


def add_pauli_measurement(prep: QuantumCircuit, pauli: str) -> QuantumCircuit:
    """
    Measure correlator <ZZ>, <XX>, or <YY> using basis changes + Z measurement.

    Bit order note:
    We measure q0 -> c1 and q1 -> c0 so the printed bitstring reads "q0q1".
    """
    if pauli not in ("ZZ", "XX", "YY"):
        raise ValueError("pauli must be one of ZZ, XX, YY")

    qc = QuantumCircuit(2, 2, name=f"{prep.name}_{pauli}")
    qc.compose(prep, inplace=True)

    qc.barrier(0, 1)

    if pauli == "XX":
        qc.h(0)
        qc.h(1)
    elif pauli == "YY":
        qc.sdg(0)
        qc.h(0)
        qc.sdg(1)
        qc.h(1)

    qc.barrier(0, 1)
    qc.measure(0, 1)  # q0 -> c1
    qc.measure(1, 0)  # q1 -> c0
    return qc


# ── Noise model ──────────────────────────────────────────────────────────────

def make_noise_model(
    p_1q: float,
    p_2q: float,
    include_readout: bool = True,
    noisy_rz: bool = False,
) -> NoiseModel:
    """
    Synthetic device-style noise model.
    - depolarizing + thermal relaxation on 1Q and 2Q gates
    - optional readout error
    - RZ treated as virtual/noiseless unless noisy_rz=True
    """
    basis = ["id", "rz", "sx", "x", "cx"]
    noise = NoiseModel(basis_gates=basis)

    t1, t2 = 30e-6, 60e-6
    dur_1q = 35e-9
    dur_2q = 300e-9

    therm_1q = thermal_relaxation_error(t1, t2, dur_1q)
    therm_2q = thermal_relaxation_error(t1, t2, dur_2q).tensor(
        thermal_relaxation_error(t1, t2, dur_2q)
    )

    err_1q = depolarizing_error(p_1q, 1).compose(therm_1q)
    err_2q = depolarizing_error(p_2q, 2).compose(therm_2q)

    oneq_gates = ["sx", "x"] + (["rz"] if noisy_rz else [])
    noise.add_all_qubit_quantum_error(err_1q, oneq_gates)
    noise.add_all_qubit_quantum_error(err_2q, ["cx"])

    if include_readout:
        noise.add_readout_error(ReadoutError([[0.921, 0.079], [0.079, 0.921]]), [0])
        noise.add_readout_error(ReadoutError([[0.941, 0.059], [0.059, 0.941]]), [1])

    return noise


# ── Estimators ───────────────────────────────────────────────────────────────

def parity_expectation(counts: Dict[str, int], shots: int) -> Tuple[float, float]:
    """E = P(same) - P(diff). Also returns binomial stderr for ±1 outcomes."""
    n = shots
    p00 = counts.get("00", 0) / n
    p01 = counts.get("01", 0) / n
    p10 = counts.get("10", 0) / n
    p11 = counts.get("11", 0) / n
    E = (p00 + p11) - (p01 + p10)
    stderr = math.sqrt(max(0.0, 1.0 - E * E) / n)
    return E, stderr


def bell_fidelity(state: str, EZZ: float, EXX: float, EYY: float) -> float:
    """Bell fidelity from stabilizer correlators."""
    s = BELL_SIGNS[state]
    return 0.25 * (1.0 + s["ZZ"] * EZZ + s["XX"] * EXX + s["YY"] * EYY)


# ── Benchmark runner ─────────────────────────────────────────────────────────

def run_benchmark(
    shots_per_basis: int = 10_000,
    noise_levels: Optional[Iterable[float]] = None,
    bell_states=("phi_plus", "phi_minus", "psi_plus", "psi_minus"),
    optimization_levels=(1, 2, 3),
    routing_methods=("sabre", "lookahead"),
    include_readout: bool = True,
    noisy_rz: bool = False,
    seed_transpiler: int = 42,
    seed_simulator: int = 42,
    csv_path: str = "bell_fidelity_results.csv",
    show_plot: bool = True,
) -> pd.DataFrame:
    """
    Run a noise sweep. For each noise point:
    - execute ZZ/XX/YY stabilizer measurements
    - estimate Bell fidelity
    - record transpiler stats (depth, CX, SWAP)
    Saves a CSV and a plot, and returns a dataframe.
    """
    if noise_levels is None:
        noise_levels = np.round(np.arange(0.01, 0.11, 0.01), 2)

    coupling_map = [[0, 1], [1, 0]]

    template_noise = make_noise_model(0.0, 0.0, include_readout=include_readout, noisy_rz=noisy_rz)
    basis_gates = template_noise.basis_gates

    compiled = []
    for state in bell_states:
        prep = prepare_bell_state(state)
        raw = {P: add_pauli_measurement(prep, P) for P in ("ZZ", "XX", "YY")}

        for opt in optimization_levels:
            for routing in routing_methods:
                circs = {}
                depth_list, cx_list, swap_list = [], [], []

                for P in ("ZZ", "XX", "YY"):
                    try:
                        tqc = transpile(
                            raw[P],
                            basis_gates=basis_gates,
                            coupling_map=coupling_map,
                            optimization_level=opt,
                            routing_method=routing,
                            initial_layout=[0, 1],
                            seed_transpiler=seed_transpiler,
                        )
                    except Exception:
                        tqc = transpile(
                            raw[P],
                            basis_gates=basis_gates,
                            coupling_map=coupling_map,
                            optimization_level=opt,
                            initial_layout=[0, 1],
                            seed_transpiler=seed_transpiler,
                        )

                    if tqc.count_ops().get("cx", 0) == 0:
                        raise RuntimeError(
                            f"CX disappeared after transpile for {state} {P} opt={opt} routing={routing}"
                        )

                    circs[P] = tqc
                    ops = tqc.count_ops()
                    depth_list.append(tqc.depth())
                    cx_list.append(int(ops.get("cx", 0)))
                    swap_list.append(int(ops.get("swap", 0)))

                compiled.append({
                    "bell_state": state,
                    "optimization": opt,
                    "routing": routing,
                    "circuits": circs,
                    "depth_mean": float(np.mean(depth_list)),
                    "cx_mean": float(np.mean(cx_list)),
                    "swap_mean": float(np.mean(swap_list)),
                })

    sim = AerSimulator(coupling_map=coupling_map, basis_gates=basis_gates)

    rows = []
    for p in noise_levels:
        p_1q = float(p)
        p_2q = float(1.5 * p_1q)

        noise = make_noise_model(p_1q, p_2q, include_readout=include_readout, noisy_rz=noisy_rz)
        sim.set_options(noise_model=noise, seed_simulator=seed_simulator)

        batch, meta = [], []
        for i, cfg in enumerate(compiled):
            for P in ("ZZ", "XX", "YY"):
                batch.append(cfg["circuits"][P])
                meta.append((i, P))

        res = sim.run(batch, shots=shots_per_basis).result()

        E = {i: {} for i in range(len(compiled))}
        SE = {i: {} for i in range(len(compiled))}
        for j, (cfg_i, P) in enumerate(meta):
            counts = res.get_counts(j)
            e, se = parity_expectation(counts, shots_per_basis)
            E[cfg_i][P] = e
            SE[cfg_i][P] = se

        for i, cfg in enumerate(compiled):
            state = cfg["bell_state"]
            EZZ, EXX, EYY = E[i]["ZZ"], E[i]["XX"], E[i]["YY"]
            sZZ, sXX, sYY = SE[i]["ZZ"], SE[i]["XX"], SE[i]["YY"]

            F = bell_fidelity(state, EZZ, EXX, EYY)
            F_stderr = 0.25 * math.sqrt(sZZ * sZZ + sXX * sXX + sYY * sYY)

            rows.append({
                "bell_state": state,
                "optimization": cfg["optimization"],
                "routing": cfg["routing"],
                "p_1q": p_1q,
                "p_2q": p_2q,
                "shots_per_basis": shots_per_basis,
                "E_ZZ": EZZ,
                "E_XX": EXX,
                "E_YY": EYY,
                "bell_fidelity": F,
                "bell_fidelity_stderr": F_stderr,
                "depth_mean": cfg["depth_mean"],
                "cx_mean": cfg["cx_mean"],
                "swap_mean": cfg["swap_mean"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    best_df = df.loc[df.groupby(["bell_state", "p_1q"])["bell_fidelity"].idxmax()].copy()

    plt.figure(figsize=(8, 5))
    for state in bell_states:
        s = best_df[best_df["bell_state"] == state].sort_values("p_1q")
        plt.plot(s["p_1q"], s["bell_fidelity"], marker="o", label=state)

    plt.xlabel("1Q depolarizing rate p_1q")
    plt.ylabel("Best Bell fidelity (stabilizer estimate)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("bell_fidelity_vs_noise.png", dpi=300, bbox_inches="tight")
    print("Saved plot: bell_fidelity_vs_noise.png")

    if show_plot:
        plt.show()
    else:
        plt.close()

    best = df.loc[df["bell_fidelity"].idxmax()]
    print("Best observed configuration:")
    print(best[["bell_state", "optimization", "routing", "p_1q", "bell_fidelity", "bell_fidelity_stderr"]])

    return df


# ── Example run (Colab) ──────────────────────────────────────────────────────

df = run_benchmark()
df.head()
