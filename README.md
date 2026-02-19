# compiler-robust-bell-benchmark

A Bell-state fidelity benchmark for studying the impact of compiler optimization and routing strategies under synthetic device-style noise using Qiskit Aer.

This project evaluates how transpiler optimization levels and routing strategies affect entanglement fidelity in the presence of realistic quantum noise.

---

## Motivation

In the NISQ (Noisy Intermediate-Scale Quantum) era, compilation choices can significantly affect circuit performance under noise. Even small differences in:

- circuit depth
- number of CX gates
- number of inserted SWAP operations
- routing strategy

can influence the final fidelity of entangled states.

This benchmark provides a controlled environment to analyze how compiler-level decisions impact Bell-state fidelity under depolarizing, thermal relaxation, and optional readout noise.

---

## Methodology

For each of the four Bell states:

- `phi_plus`
- `phi_minus`
- `psi_plus`
- `psi_minus`

the benchmark performs the following steps:

1. Prepare the Bell state on 2 qubits.

2. Construct measurement circuits for stabilizer correlators:
   - $\langle ZZ \rangle$
   - $\langle XX \rangle$
   - $\langle YY \rangle$

3. Estimate Bell fidelity using the stabilizer-based estimator:

$$
F = \frac{1 + s_{ZZ}\langle ZZ \rangle + s_{XX}\langle XX \rangle + s_{YY}\langle YY \rangle}{4}
$$

where the signs $s_{ZZ}$, $s_{XX}$, and $s_{YY}$ depend on the chosen Bell state.

4. Transpile circuits using:
   - multiple optimization levels
   - different routing strategies

5. Sweep depolarizing noise levels by varying $p_{1q}$ and setting:

$$
p_{2q} = 1.5 \, p_{1q}
$$

6. Execute simulations using Qiskit Aer with:
   - depolarizing noise
   - thermal relaxation noise
   - optional readout error

7. Record:
   - Bell fidelity $F$
   - statistical error
   - mean circuit depth (averaged across $\langle ZZ \rangle$, $\langle XX \rangle$, and $\langle YY \rangle$ circuits)
   - mean CX count
   - mean SWAP count

The benchmark selects the best configuration per noise level and generates a comparative plot.

---

## Application

This benchmark can be used to:

- study compiler robustness under realistic noise
- compare transpiler optimization strategies
- analyze routing-induced fidelity degradation
- explore correlations between circuit complexity and entanglement loss
- serve as a lightweight experimental framework for NISQ benchmarking

---

## Installation

Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# or
.venv\Scripts\activate           # Windows
