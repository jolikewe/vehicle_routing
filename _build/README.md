# Page build scripts

Everything here regenerates `index.html` and the numbers it shows. Run from the root of this `gh-pages` checkout, with the main branch cloned next to it as `../vehicle_routing-main`:

```bash
git clone https://github.com/jolikewe/vehicle_routing.git ../vehicle_routing-main
python3 _build/vrp_extract.py        # CP-SAT optimal schedule -> schedule.json (needs ortools)
python3 _build/vrp_cpsat_reference.py ../vehicle_routing-main/notebooks   # CP-SAT optimum per objective -> _build/cpsat_reference.json
python3 _build/vrp_cpsat_reference.py ../vehicle_routing-main/notebooks free   # optional, slow: free-hub runs -> cpsat_free_hub.json
node _build/vrp_test.js 20 5000      # checks: simulator scores the CP-SAT schedule as 531, routes match numpy seed 8, solver gap
node _build/vrp_verify_browser.js 20 3000   # browser solver vs CP-SAT -> _build/reference.json
python3 _build/vrp_build.py          # inlines solver, data and reference into index.html
```

- `vrp_solver.js`: schedule simulator and simulated-annealing solver (browser worker and node).
- `vrp_engine.js`: slices the search and reports progress.
- `vrp_template.html`: page source.
