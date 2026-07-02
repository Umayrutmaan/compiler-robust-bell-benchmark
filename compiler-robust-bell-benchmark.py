import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # needed for saving without display
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)


# stabilizer signs -- this part was annoying to get right
# each bell state has different signs for XX YY ZZ correlators
# F = (1 + sZZ*EZZ + sXX*EXX + sYY*EYY) / 4
# double checked with nielsen and chuang chapter 4 and also some online notes
BELL_SIGNS = {
    "phi_plus":  {"XX": +1, "YY": -1, "ZZ": +1},
    "phi_minus": {"XX": -1, "YY": +1, "ZZ": +1},
    "psi_plus":  {"XX": +1, "YY": +1, "ZZ": -1},
    "psi_minus": {"XX": -1, "YY": -1, "ZZ": -1},
}

# global params, easier to change here
SHOTS = 10_000
N_NOISE  = 10
COUPLING_MAP = [[0, 1], [1, 0]]   # simple 2 qubit line topology


# make the bell state circuit (no measurement yet)
# measurement added separately depending on which basis we need
def make_bell_circuit(state):
    if state not in BELL_SIGNS:
        raise ValueError("dont know state: " + state)

    qc = QuantumCircuit(2, name="prep_" + state)
    qc.h(0)
    qc.cx(0, 1)   # this gives phi_plus

    # corrections for other states
    if state == "phi_minus":
        qc.z(0)
    elif state == "psi_plus":
        qc.x(1)
    elif state == "psi_minus":
        qc.x(1)
        qc.z(0)    # had to add both

    return qc


# add measurement in ZZ XX or YY basis
# bit order is q0->c1 and q1->c0 so printed bitstring is "q0 q1"
# took me time to figre out why results looked wrong, was the bit order
def add_measurement(prep_circ, pauli):
    if pauli not in ["ZZ", "XX", "YY"]:
        raise ValueError("pauli must be ZZ XX or YY")

    qc = QuantumCircuit(2, 2, name=prep_circ.name + "_" + pauli)
    qc.compose(prep_circ, inplace=True)

    qc.barrier(0, 1)

    if pauli == "XX":
        # rotate to X basis with hadamard
        qc.h(0)
        qc.h(1)
    elif pauli == "YY":
        # sdg then H rotates Y to Z, had to look this up twice
        qc.sdg(0)
        qc.h(0)
        qc.sdg(1)
        qc.h(1)
    # ZZ doesnt need any rotation

    qc.barrier(0, 1)
    qc.measure(0, 1)   # q0 -> bit 1
    qc.measure(1, 0)   # q1 -> bit 0
    return qc


# noise model
# chose T1=30us T2=60us cuz ibm uses smth similar for superconducting qubits
# gate times from ibm docs roughly
# p_2q = 1.5 * p_1q not sure if best ratio but 2q gates are always worse
def make_noise(p1q, p2q, add_readout=True):
    basis_g = ["id", "rz", "sx", "x", "cx"]
    nm = NoiseModel(basis_gates=basis_g)

    T1 = 30e-6
    T2 = 60e-6    # T2 must be <= 2*T1, this is fine
    gate_t_1q = 35e-9
    gate_t_2q = 300e-9   # cx is much slower than 1q gates

    # thermal relaxation errors
    therm1 = thermal_relaxation_error(T1, T2, gate_t_1q)
    therm2 = thermal_relaxation_error(T1, T2, gate_t_2q).tensor(
                thermal_relaxation_error(T1, T2, gate_t_2q))

    # depolarizing composed with thermal -- order matters here
    err1 = depolarizing_error(p1q, 1).compose(therm1)
    err2 = depolarizing_error(p2q, 2).compose(therm2)

    nm.add_all_qubit_quantum_error(err1, ["sx", "x"])
    nm.add_all_qubit_quantum_error(err2, ["cx"])
    # rz is virtual gate on real ibm devices so no noise on it

    if add_readout:
        # readout errors, q0 slightly worse than q1, not sure why but copied from ibm specs
        ro0 = [[0.921, 0.079],[0.079, 0.921]]
        ro1 = [[0.941, 0.059],[0.059, 0.941]]
        nm.add_readout_error(ReadoutError(ro0), [0])
        nm.add_readout_error(ReadoutError(ro1), [1])

    return nm


# compute parity expectation E = P(same) - P(different)
# also returns stderr, standard binomial formula
def get_parity(counts, shots):
    n = shots
    p00 = counts.get("00", 0) / n
    p01 = counts.get("01", 0) / n
    p10 = counts.get("10", 0) / n
    p11 = counts.get("11", 0) / n
    E = (p00 + p11) - (p01 + p10)
    # variance formula for +-1 random variable
    std_err = math.sqrt(max(0.0, 1.0 - E*E) / n)
    return E, std_err


# compute bell fidelity from stabilizer measurements
def get_fidelity(state, EZZ, EXX, EYY):
    s = BELL_SIGNS[state]
    F = 0.25 * (1.0 + s["ZZ"]*EZZ + s["XX"]*EXX + s["YY"]*EYY)
    return F


# =========================================================
# main script starts here
# =========================================================

all_states   = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]
opt_levels   = [1, 2, 3]
routing_opts = ["sabre", "lookahead"]

noise_levels = np.round(np.arange(0.01, 0.11, 0.01), 2)   # 0.01 to 0.10

# need basis gates -- get from dummy zero noise model
dummy_nm = make_noise(0.0, 0.0, add_readout=False)
bg = dummy_nm.basis_gates

# ==== STEP 1: transpile all circuits first ====
# no need to redo transpilation for every noise level since topology doesnt change
# this was one of the optimizations i added later to make it faster

print("="*50)
print("transpiling circuits...")
print("="*50)

all_configs = []

for st in all_states:
    bell_prep = make_bell_circuit(st)
    # make ZZ XX YY measurement circuits for this state
    meas_circs = {}
    for P in ["ZZ", "XX", "YY"]:
        meas_circs[P] = add_measurement(bell_prep, P)

    for opt in opt_levels:
        for routing in routing_opts:
            transpiled = {}
            d_list = []
            cx_list = []
            sw_list = []

            for P in ["ZZ", "XX", "YY"]:
                try:
                    tc = transpile(
                        meas_circs[P],
                        basis_gates=bg,
                        coupling_map=COUPLING_MAP,
                        optimization_level=opt,
                        routing_method=routing,
                        initial_layout=[0, 1],
                        seed_transpiler=42,
                    )
                except Exception as ex:
                    # lookahead sometimes fails with certain configs
                    # fallback: transpile without routing_method arg
                    print(f"  warning: routing={routing} failed for {st} {P} opt={opt}, using default")
                    tc = transpile(
                        meas_circs[P],
                        basis_gates=bg,
                        coupling_map=COUPLING_MAP,
                        optimization_level=opt,
                        initial_layout=[0, 1],
                        seed_transpiler=42,
                    )

                # cx should not disappear after transpile, if it does smth is wrong
                n_cx = tc.count_ops().get("cx", 0)
                if n_cx == 0:
                    raise RuntimeError(f"cx vanished after transpile: {st} {P} opt={opt} routing={routing}")

                transpiled[P] = tc
                op_counts = tc.count_ops()
                d_list.append(tc.depth())
                cx_list.append(int(op_counts.get("cx", 0)))
                sw_list.append(int(op_counts.get("swap", 0)))

            all_configs.append({
                "state":   st,
                "opt":     opt,
                "routing": routing,
                "circs":   transpiled,
                "depth":   np.mean(d_list),
                "cx_avg":  np.mean(cx_list),
                "sw_avg":  np.mean(sw_list),
            })

print(f"done. got {len(all_configs)} total configs")
print()

# ==== STEP 2: run noise sweep ====
print("running noise sweep...")

sim = AerSimulator(coupling_map=COUPLING_MAP, basis_gates=bg)
all_rows = []

for p_val in noise_levels:
    p1 = float(p_val)
    p2 = round(1.5 * p1, 4)    # p_2q always 1.5x the 1q rate

    current_nm = make_noise(p1, p2)
    sim.set_options(noise_model=current_nm, seed_simulator=42)

    # put all circuits into one big batch and run at once
    # way faster than calling sim.run for each one separately
    batch_circs = []
    batch_meta  = []
    for ci, cfg in enumerate(all_configs):
        for P in ["ZZ", "XX", "YY"]:
            batch_circs.append(cfg["circs"][P])
            batch_meta.append((ci, P))

    result_obj = sim.run(batch_circs, shots=SHOTS).result()

    # unpack: store expectation values per config
    E_vals  = {i: {} for i in range(len(all_configs))}
    SE_vals = {i: {} for i in range(len(all_configs))}

    for j, (ci, P) in enumerate(batch_meta):
        cnts = result_obj.get_counts(j)
        ev, se = get_parity(cnts, SHOTS)
        E_vals[ci][P]  = ev
        SE_vals[ci][P] = se

    for ci, cfg in enumerate(all_configs):
        st = cfg["state"]
        EZZ = E_vals[ci]["ZZ"];   EXX = E_vals[ci]["XX"];   EYY = E_vals[ci]["YY"]
        sZZ = SE_vals[ci]["ZZ"];  sXX = SE_vals[ci]["XX"];  sYY = SE_vals[ci]["YY"]

        F  = get_fidelity(st, EZZ, EXX, EYY)
        Fs = 0.25 * math.sqrt(sZZ**2 + sXX**2 + sYY**2)   # error propagation

        all_rows.append({
            "bell_state":           st,
            "opt_level":            cfg["opt"],
            "routing":              cfg["routing"],
            "p_1q":                 p1,
            "p_2q":                 p2,
            "shots":                SHOTS,
            "E_ZZ":                 round(EZZ, 6),
            "E_XX":                 round(EXX, 6),
            "E_YY":                 round(EYY, 6),
            "bell_fidelity":        round(F,  6),
            "bell_fidelity_stderr": round(Fs, 6),
            "depth":                cfg["depth"],
            "cx_avg":               cfg["cx_avg"],
            "swap_avg":             cfg["sw_avg"],
        })

    print(f"  noise p_1q={p1:.2f}  p_2q={p2:.3f}  done")

# ==== STEP 3: save results ====
df = pd.DataFrame(all_rows)
csv_name = "bell_fidelity_results.csv"
df.to_csv(csv_name, index=False)
print(f"\nsaved: {csv_name}  ({len(df)} rows)")

# ==== STEP 4: plot ====
# just showing best fidelity per state vs noise, simple line plot
# not sure if best way to visualize but its clear enuf

best_df = df.loc[df.groupby(["bell_state","p_1q"])["bell_fidelity"].idxmax()].copy()

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#2980b9","#c0392b","#27ae60","#e67e22"]

for i, st in enumerate(all_states):
    sub = best_df[best_df["bell_state"]==st].sort_values("p_1q")
    ax.plot(sub["p_1q"], sub["bell_fidelity"],
            marker="o", color=colors[i], label=st)

ax.set_xlabel("1Q depolarizing rate (p_1q)")
ax.set_ylabel("bell fidelity (stabilizer tomography)")
ax.set_title("Bell Fidelity vs Noise\n(best compiler setting per state)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("bell_fidelity_vs_noise.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved: bell_fidelity_vs_noise.png")

# print best overall result
top_row = df.loc[df["bell_fidelity"].idxmax()]
print("\nbest result found:")
print(f"  state    = {top_row['bell_state']}")
print(f"  opt      = {top_row['opt_level']}")
print(f"  routing  = {top_row['routing']}")
print(f"  p_1q     = {top_row['p_1q']:.2f}")
print(f"  fidelity = {top_row['bell_fidelity']:.4f} +/- {top_row['bell_fidelity_stderr']:.4f}")
print()
print("first few rows of data:")
print(df[["bell_state","opt_level","routing","p_1q","bell_fidelity"]].head(8).to_string(index=False))
print("\ndone.")
