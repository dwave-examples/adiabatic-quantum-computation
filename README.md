[![Open in GitHub Codespaces](
  https://img.shields.io/badge/Open%20in%20GitHub%20Codespaces-333?logo=github)](
  https://codespaces.new/dwave-examples/adiabatic-quantum-computation-notebook?quickstart=1)

### Adiabatic Quantum Computation (with Landau-Zener like scaling)

A demonstration of using the
[fast-anneal protocol](https://docs.dwavequantum.com/en/latest/quantum_research/annealing.html)
on a quantum computer to determine ground states for frustrated 16-variable Ising models
compatible with [Adiabatic Quantum Computation](https://en.wikipedia.org/wiki/Adiabatic_quantum_computation).

This example employs quantum annealing in the coherent regime of a D-Wave
quantum processing unit (QPU), producing ground states at a rate aligned with
theoretical predictions: in particular gap-dependent scaling based upon the
[Landau-Zener formula](https://en.wikipedia.org/wiki/Landau%E2%80%93Zener_formula).

This demo complements the Kibble-Zurek demo, which demonstrates statistical properties in
the regime with high rates of excitation away from the ground state.

![Demo Example](static/demo.png "Image of demo interface")

* [Installation](#Installation)
* [Usage](#Usage)
* [Adiabatic Quantum Computation](#Adiabatic Quantum Computation)
* [Model Overview](#Model-Overview)
* [Code](#Code)
* [License](#License)

## <a name="Installation"></a> Installation

You can run this example without installation in cloud-based IDEs that support the
[Development Containers Specification](https://containers.dev/supporting) (aka "devcontainers")
such as GitHub Codespaces.

For development environments that do not support `devcontainers`, install requirements:

```bash
pip install -r requirements.txt
```

If you are cloning the repo to your local system, working in a
[virtual environment](https://docs.python.org/3/library/venv.html) is recommended.

## <a name="Usage"></a> Usage

Your development environment should be configured to access the
[Leap&trade; quantum cloud service](https://docs.dwavequantum.com/en/latest/ocean/sapi_access_basic.html).
You can see information about supported IDEs and authorizing access to your Leap account
[here](https://docs.dwavequantum.com/en/latest/ocean/leap_authorization.html).

Run the following terminal command to start the Dash application:

```bash
python app.py
```

Access the user interface with your browser at http://127.0.0.1:8050/.

The demo program opens an interface where you can configure problems and submit these problems to
a solver.

Configuration options can be found in the [demo_configs.py](demo_configs.py) file.

> [!NOTE]\
> If you plan on editing any files while the application is running, please run the application
with the `--debug` command-line argument for live reloads and easier debugging:
`python app.py --debug`

## Adiabatic Quantum Computation
Adiabatic quantum computation is a canonical theoretical model of quantum computation.
In simplest formulations, a state is prepared as a ground state of a driver, the driver
energy term is slowly scaled to zero whilst a problem Hamiltonian is scaled up.
Adiabatic evolution guarantees that with sufficient isolation from the environment, and
Hamiltonian evolution at a sufficiently small rate, the system evolves to the ground
state of the problem Hamiltonian.

In D-Wave Quantum annealers a transverse-field driver is combined with a diagonal problem
Hamiltonian. The problem Hamiltonian can be programmed to encode in its ground state
solutions to an optimization problem.

For many optimization problem the spectrum of the Hamiltonian, as it evolves from driver
dominated to problem dominated, demonstrates a well defined closing gap (or anti-crossing).
At closing gaps transitions out of the ground state are possible via diabatic transitions.
Where transitions from the lowest energy state to the 1st excited state at a single
anti-crossing the limit the success probability (attainment of a ground state) a two state
(Landau-Zener) process can describe the the exponentially small rates of excitation as
annealing time increases. 

## Model Overview

As annealing time increases we anticipate decreased rates of excitation.
Asymptotically this is expected to follow an exponential decay, with the
decay rate determined by spectral gap properties.

### Experiment

The problem Hamiltonians studied are 16 qubit, 52 coupler, frustrated Ising models with
uniform (AFM and FM) absolute coupling strength matched to those presented in [1] Figure 2c.
These are stored in static/model as json files and selected by the user to study problems with variable gap size.

The problem Hamiltonian can be scaled up or down to modify the gap properties, using a
slider control. The evolution time is controlled by an annealing time slider.

The problem is small enough that its programming can be parallelized across several
disjoint processor locations. Data is collected from a single programming of 1000 reads.
Standard error bars are calculated treating each embedding as an independent estimators
(errors reflect a combination of control error and sampling error). 

### Theoretical models

The evolution of the driver term and problem Hamiltonians is modeled using device-specific
annealing schedules. The spectrum is calculated via exact diagonalization on the basis of
the programmed Hamlitonian and schedule. As demo parameters are adjusted it is recalculated.

A theory curve f(t_a) plotted along side the data is an exact match to [1] Figure 2c.
A Landau-Zener transition rate tau_n can be calculated based on the gap size and the rate at
which it closes (with a locally linear approximation). Assuming a Landau Zener process to be a good description
of dynamics we anticipate f(t_a tau_0/tau_n) to describe all experimental conditions.
tau_0 characterizes the gap properties of the QPU on which experiments [1] were originally performed:
problem (large gap, medium gap); problem Hamiltonian rescaling; QPU (schedule).
tau_n characterizes the gap properties of the Hamiltonian on newer generally available devices, the ratio
tau_n/tau_0 compensates for changes in energy scales.

Uncertainty in the theoretical approximations arise from uncertainty in the schedule, and
small corrections to the Landau-Zener approximation. Nevertheless, very good agreement is seen across
models and QPUs, and under rescaling of the problem Hamiltonian.

## Code Overview

To do: Use KZM as a model.

## References

## References

<a name="1">[1]</a> King, A.D., Suzuki, S., Raymond, J. et al.
Coherent quantum annealing in a programmable 2,000 qubit Ising chain.
Nat. Phys. 18, 1324–1328 (2022). https://doi.org/10.1038/s41567-022-01741-6

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE) file.
