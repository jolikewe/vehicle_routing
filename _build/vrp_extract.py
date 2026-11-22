"""Re-runs the repo's own model (notebooks/planck_optimiser.Model) with the
parameters stored in notebooks/multi_multi.ipynb (18 personnel, 3 trips, cap 3)
and exports the optimal schedule to schedule.json."""
import sys, json
sys.path.insert(0, '../vehicle_routing-main/notebooks')
from ortools.sat.python import cp_model
from planck_optimiser import Model
params = {'num_personnel': 18,'max_trips': 3,'num_points_start': 2,'num_points_holding': 3,
          'num_points_destination': 2,'num_vehicles_alpha': 2,'num_vehicles_beta': 2,
          'vehicle_capacity': 3,'loading_time': 0}
m = Model(params)
m.generate_route()
m.generate_personnel_route(); m.generate_vehicle_route(); m.generate_node_visitation_vehicle()
m.generate_node_visitation_personnel(); m.generate_arrival_vehicle(); m.generate_arrival_personnel(); m.generate_waiting()
for f in ['vehicle_node','vehicle_next_trip','vehicle_trip_limit','personnel_node','board_vehicle','personnel_start',
          'personnel_trip_limit','vehicle_capacity','vehicle_arrival','holding_arrival','holding_arrival_0',
          'personnel_arrival','personnel_holding']:
    getattr(m, 'generate_'+f+'_constraint')()

# Initial state (notebook cell 12) and objective (cell 14)
for n in range(m.num_personnel):
    m.Add(m.node_p[(n, m.points_holding[n % 3])] == 1)
    k = 0 if n < m.num_personnel // 2 else 1
    m.Add(m.node_p[(n, m.points_start[k])] == 1)
    m.Add(m.node_p[(n, m.points_destination[k])] == 1)
m.Minimize(sum(m.arrival_p[(n, i)] for i in m.points_destination for n in range(m.num_personnel)))
solver = cp_model.CpSolver()
status = solver.Solve(m)
assert status == cp_model.OPTIMAL, status
obj = int(solver.ObjectiveValue())
print('objective', obj, 'wall', solver.WallTime())
assert obj == 531
V = solver.Value
d = lambda a, b: int(m.routes.loc[a, b])

people = []
for n in range(m.num_personnel):
    S = next(i for i in m.points_start if V(m.node_p[(n, i)]))
    H = next(i for i in m.points_holding if V(m.node_p[(n, i)]))
    D = next(i for i in m.points_destination if V(m.node_p[(n, i)]))
    vb = next(k for k, v in m.var_y.items() if k[0] == n and V(v))
    va = next(k for k, v in m.var_x.items() if k[0] == n and V(v))
    tH, tD = V(m.arrival_p[(n, H)]), V(m.arrival_p[(n, D)])
    people.append(dict(id=n, S=S, H=H, D=D, beta=vb[1], beta_trip=vb[2], alpha=va[1], alpha_trip=va[2],
                       depart_S=tH - d(S, H), arrive_H=tH, depart_H=tD - d(H, D), arrive_D=tD))
assert sum(p['arrive_D'] for p in people) == obj

legs = []
for key, kind in ((m.var_q, 'beta'), (m.var_p, 'alpha')):
    for (v, r, i, j), var in key.items():
        if not V(var): continue
        loaded = (kind == 'beta' and i[0] == 'S') or (kind == 'alpha' and i[0] == 'H')
        arr = V(m.arrival_v[(v, r, j)])
        riders = [p['id'] for p in people if p[kind] == v and p[kind + '_trip'] == r] if loaded else []
        legs.append(dict(vehicle=v, kind=kind, trip=r, frm=i, to=j, depart=arr - d(i, j), arrive=arr,
                         loaded=loaded, riders=riders))
legs.sort(key=lambda l: (l['vehicle'], l['depart'], l['arrive']))
nodes = dict(S=m.points_start, H=m.points_holding, D=m.points_destination)
out = dict(objective=obj, params=params, solver_status='OPTIMAL', routes={f'{a}-{b}': d(a, b) for a in m.points for b in m.points
           if a[0] < b[0] and d(a, b) < 999 and (a[0], b[0]) in (('S','H'),('H','S'))} | {f'{a}-{b}': d(a, b) for a in m.points_holding for b in m.points_destination},
           nodes=nodes, people=people, legs=legs, makespan=max(p['arrive_D'] for p in people))
json.dump(out, open('./schedule.json', 'w'), separators=(',', ':'))
for l in legs: print(l)
