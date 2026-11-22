"""Solves the repo instance (18 people, 3 trips, capacity 3, hubs fixed to n mod 3) with the repo's CP-SAT model
for each objective and writes cpsat_reference.json. Usage: python vrp_cpsat_reference.py [path/to/repo/notebooks]"""
import sys, json, time
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else '../vehicle_routing-main/notebooks')
from ortools.sat.python import cp_model
from planck_optimiser import Model

def build(kind, weight=0.5, n=18, fixed_hub=True):
    params = {'num_personnel': n, 'max_trips': 3, 'num_points_start': 2, 'num_points_holding': 3,
              'num_points_destination': 2, 'num_vehicles_alpha': 2, 'num_vehicles_beta': 2,
              'vehicle_capacity': 3, 'loading_time': 0}
    m = Model(params); m.generate_route()
    m.generate_personnel_route(); m.generate_vehicle_route(); m.generate_node_visitation_vehicle()
    m.generate_node_visitation_personnel(); m.generate_arrival_vehicle(); m.generate_arrival_personnel(); m.generate_waiting()
    for f in ['vehicle_node', 'vehicle_next_trip', 'vehicle_trip_limit', 'personnel_node', 'board_vehicle', 'personnel_start',
              'personnel_trip_limit', 'vehicle_capacity', 'vehicle_arrival', 'holding_arrival', 'holding_arrival_0',
              'personnel_arrival', 'personnel_holding']:
        getattr(m, 'generate_' + f + '_constraint')()
    for i in range(n):
        if fixed_hub:
            m.Add(m.node_p[(i, m.points_holding[i % 3])] == 1)
        k = 0 if i < n // 2 else 1
        m.Add(m.node_p[(i, m.points_start[k])] == 1); m.Add(m.node_p[(i, m.points_destination[k])] == 1)
    return m, m.set_objective(kind, weight)

FREE = len(sys.argv) > 2 and sys.argv[2] == 'free'   # slower: hubs chosen by the solver, written to cpsat_free_hub.json
if FREE:
    out = {}
    for kind, tl in [('sum', 400), ('makespan', 200)]:
        m, conv = build(kind, fixed_hub=False)
        s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tl; s.parameters.num_workers = 8
        t = time.time(); st = s.Solve(m)
        out[kind] = dict(value=conv(s.ObjectiveValue()), bound=conv(s.BestObjectiveBound()), status=s.StatusName(st), secs=round(time.time() - t, 1))
        print('free hubs', kind, out[kind])
    json.dump(out, open('cpsat_free_hub.json', 'w'), indent=1)
    sys.exit()
out = {}
for kind in ['sum', 'makespan', 'wait', 'mix']:
    m, conv = build(kind)
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = 600; s.parameters.num_workers = 8
    t = time.time(); st = s.Solve(m)
    out[kind] = dict(value=conv(s.ObjectiveValue()), bound=conv(s.BestObjectiveBound()), status=s.StatusName(st), secs=round(time.time() - t, 1))
    print(kind, out[kind])
assert out['sum']['value'] == 531
json.dump(out, open('cpsat_reference.json', 'w'), indent=1)
