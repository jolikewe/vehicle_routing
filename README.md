# Vehicle Routing and Personnel Scheduling with CP-SAT

Constraint-programming models (Google OR-Tools CP-SAT) for scheduling personnel and vehicles through a two-stage transport system.

**[▶ Live demo](https://jolikewe.github.io/vehicle_routing/)**: change the problem (people, vehicles, seats, travel times, objective) and watch a planner in your browser build the schedule live, then compare it with the CP-SAT optimum of 531.

![Python](https://img.shields.io/badge/python-3-blue) ![OR-Tools](https://img.shields.io/badge/OR--Tools-CP--SAT-green)

## Overview

Personnel travel from START points to DESTINATION points via HOLDING points. Beta vehicles carry them from START to HOLDING, and Alpha vehicles from HOLDING to DESTINATION. Vehicles have limited capacity and can make several trips. The model decides which vehicle and trip each person takes, and the resulting arrival times, so that the total arrival time is minimised. The full formulation (variables, constraints, objectives, planned local-search ideas) is in [docs/problem_formulation.md](docs/problem_formulation.md).

<img src="docs/images/overview.jpg" alt="System overview" width="50%">

## Results

Values below are taken from the stored notebook outputs and the notes in `notebooks/multi_multi.ipynb`.

| Notebook / model | Setup | Objective (sum of arrival times) |
| --- | --- | --- |
| `single_vehicle_cpsat.ipynb` | 5 personnel, 1 alpha + 1 beta vehicle | 68 |
| `multi_vehicle.ipynb` | 5 personnel, 1 alpha + 2 beta vehicles | 34 |
| `multi_multi.ipynb` (stored run) | 18 personnel, max 3 trips, capacity 3 | 531 (optimal) |
| `models/model_2.pkl` | 14 personnel, 4 trips, capacity 3 | 471 (9.3 min, per notebook note) |
| `models/model_3.pkl` | 18 personnel, 4 trips, capacity 3 | 531 (33 min, per notebook note) |

## Repository structure

```
.
├── notebooks/
│   ├── single_vehicle_cpsat.ipynb  # prototype: one alpha and one beta vehicle, single trip
│   ├── multi_vehicle.ipynb         # prototype: several vehicles, one trip each
│   ├── multi_multi.ipynb           # full model: several vehicles, several trips
│   └── planck_optimiser.py         # Model class (CP-SAT) used by multi_multi.ipynb
├── models/                         # pickled solved Model objects from multi_multi runs
├── docs/
│   ├── problem_formulation.md      # formulation, constraints, design notes
│   └── images/                     # diagrams
└── requirements.txt
```

## Getting started

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Run `multi_multi.ipynb` for the full model. It imports `planck_optimiser.py`, so start Jupyter from the `notebooks/` directory. The larger instances took minutes to tens of minutes in the original runs (see the notes in the notebook).

The pickles in `models/` were written with a specific OR-Tools version and may not load with a newer one; they are commented out in the notebook and are only kept as saved results.

## Model extensions

`notebooks/planck_optimiser.py` has extra methods (the original ones are unchanged and `multi_multi.ipynb` still gives 531):

- `set_objective('makespan' | 'wait' | 'mix')` adds the formulation's other objectives: last arrival, average wait at the holding point, and `last arrival + weight * average wait`. Verified on the 18-person instance: 43, 0 and 43.33, all proven optimal.
- Optional `generate_launch_constraint`, `tighten_horizon`, `add_symmetry_breaking` and a `max_wait` cap. The last two speed-up options did not make the 18-person sum solve faster in tests, so they are off by default.

What changed, why, and the measured results are in [docs/problem_formulation.md](docs/problem_formulation.md#amendments).

## Limitations

- The maximum number of trips per vehicle must be fixed in advance; large values increase solve time sharply.
- The notebooks still minimise the sum of arrival times; the other objectives are available through `set_objective` but are not used in the stored notebook runs.
- Average waiting time alone is trivial to drive to zero (hold the buses back), so it is best used in the weighted mix.
- With the holding point left free, the 18-person sum was not solved to optimality by CP-SAT within 400 s.
