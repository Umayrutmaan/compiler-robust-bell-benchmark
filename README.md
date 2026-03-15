# compiler-robust-bell-benchmark

A Qiskit Aer benchmark for studying how transpiler optimization and routing choices affect Bell-state fidelity under a synthetic device-style noise model.

This project simulates the preparation and measurement of the four standard Bell states and evaluates how compilation choices change fidelity, circuit depth, CX count, and SWAP count as noise increases. :contentReference[oaicite:1]{index=1}

---

## Motivation

In noisy quantum hardware and realistic simulator settings, compiler decisions matter. Different transpiler settings can change the final circuit structure even when the target state is the same. In practice, this can affect entanglement fidelity through changes in:

- circuit depth
- number of CX gates
- number of inserted SWAP gates
- routing method

This benchmark provides a controlled two-qubit test case for comparing compilation strategies under depolarizing noise, thermal relaxation, and optional readout error. :contentReference[oaicite:2]{index=2}

---

## What the project does

The benchmark works with the following Bell states:

- `phi_plus`
- `phi_minus`
- `psi_plus`
- `psi_minus`

For each Bell state, the code:

1. Prepares the Bell state on 2 qubits.
2. Builds measurement circuits for the stabilizer correlators:
   - $\langle ZZ \rangle$
   - $\langle XX \rangle$
   - $\langle YY \rangle$
3. Estimates Bell-state fidelity from these correlators using the Bell-state sign convention defined in the code.
4. Transpiles each circuit using:
   - optimization levels `1`, `2`, and `3`
   - routing methods `sabre` and `lookahead`
5. Sweeps the 1-qubit depolarizing error rate over:
   - `p_1q = 0.01, 0.02, ..., 0.10`
   - with `p_2q = 1.5 * p_1q`
6. Simulates execution with Qiskit Aer using:
   - depolarizing noise
   - thermal relaxation noise
   - readout error by default
7. Records:
   - $\langle ZZ \rangle$, $\langle XX \rangle$, $\langle YY \rangle$
   - Bell fidelity
   - Bell-fidelity statistical error
   - mean circuit depth
   - mean CX count
   - mean SWAP count

The script saves the full benchmark data to `bell_fidelity_results.csv` and generates a plot `bell_fidelity_vs_noise.png`. For the plot, it keeps the best-fidelity compiled configuration for each Bell state at each noise level. :contentReference[oaicite:3]{index=3}

---

## Fidelity estimator

Bell fidelity is estimated from the measured stabilizer correlators as

$$
F = \frac{1 + s_{ZZ}\langle ZZ \rangle + s_{XX}\langle XX \rangle + s_{YY}\langle YY \rangle}{4},
$$

where the signs $s_{ZZ}$, $s_{XX}$, and $s_{YY}$ depend on the selected Bell state. The implementation uses the standard Bell-state stabilizer-sign table defined in the script. :contentReference[oaicite:4]{index=4}

---

## Noise model

The simulator uses a synthetic device-style noise model built from:

- 1-qubit depolarizing noise
- 2-qubit depolarizing noise
- thermal relaxation noise on 1-qubit and 2-qubit gates
- optional readout error
- optional noise on `rz` gates, which are noiseless by default

The transpilation basis gates are:

- `id`
- `rz`
- `sx`
- `x`
- `cx`

and the benchmark uses a bidirectional 2-qubit coupling map:
- `[0, 1]`
- `[1, 0]` :contentReference[oaicite:5]{index=5}

---

## Outputs

Running the benchmark produces:

- a CSV file with all raw benchmark results: `bell_fidelity_results.csv`
- a summary plot of best Bell fidelity versus noise: `bell_fidelity_vs_noise.png`
- a printed summary of the best observed configuration

By default, the benchmark uses `10_000` shots per measurement basis and fixed seeds for both transpilation and simulation. :contentReference[oaicite:6]{index=6}

---

## Use cases

This project can be used to:

- compare transpiler optimization levels for Bell-state preparation
- study how routing choices affect fidelity under noise
- analyze the relationship between circuit cost and entanglement quality
- generate a lightweight, reproducible benchmark for noisy two-qubit circuits
- explore how compilation decisions interact with synthetic hardware-style noise :contentReference[oaicite:7]{index=7}

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# or
.venv\Scripts\activate           # Windows
