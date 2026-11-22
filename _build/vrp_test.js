/* Node checks: (1) simulator scores the CP-SAT schedule as 531, (2) numpy-compatible routes,
 * (3) solver gap over repeated runs on the repo instance. Usage: node vrp_test.js [runs] [ms] */
const V = require('./vrp_solver.js'), fs = require('fs'), path = require('path');
const sch = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'schedule.json'), 'utf8'));
function instFromSchedule(sch) {
  const p = sch.params, r = sch.routes, nH = p.num_points_holding;
  const dSH = [[], []], dHD = [];
  for (let s = 0; s < 2; s++) for (let h = 0; h < nH; h++) dSH[s].push(r[`H${h + 1}-S${s + 1}`]);
  for (let h = 0; h < nH; h++) dHD.push([r[`H${h + 1}-D1`], r[`H${h + 1}-D2`]]);
  return { counts: [p.num_personnel / 2, p.num_personnel / 2], nH, nB: p.num_vehicles_beta, nA: p.num_vehicles_alpha, cap: p.vehicle_capacity, R: p.max_trips, load: p.loading_time, dSH, dHD, fixedHub: true };
}
const P = instFromSchedule(sch), I = V.makeInst(P);
// 1. simulator on the CP-SAT schedule
const st = V.stateFromSchedule(I, sch);
const ok = V.simulate(I, st.bSlot, st.aSlot, st.hub, false);
const rebuilt = V.buildSchedule(I, st, false);
let mism = 0;
rebuilt.people.forEach((p, i) => { const q = sch.people[i]; if (p.arrive_D !== q.arrive_D || p.arrive_H !== q.arrive_H) mism++; });
console.log('simulator valid:', ok, ' sum =', rebuilt.sum, ' makespan =', rebuilt.makespan, ' per-person arrival mismatches:', mism);
if (!ok || rebuilt.sum !== 531 || mism) { console.log('FAIL: simulator does not reproduce CP-SAT'); process.exit(1); }
// 2. routes from seed 8 match the repo's generate_route(seed=8)
const rr = V.randomRoutes(3, 8);
console.log('seed 8 routes equal repo:', JSON.stringify(rr.dSH) === JSON.stringify(P.dSH) && JSON.stringify(rr.dHD) === JSON.stringify(P.dHD), JSON.stringify(rr));
// 3. solver
const runs = +process.argv[2] || 20, ms = +process.argv[3] || 5000;
const res = [], tHit = [];
for (let k = 0; k < runs; k++) {
  const S = new V.Solver(V.makeInst(P), { kind: 'sum', lambda: 0.5 }, 1000 + k, ms);
  while (!S.step(50)) {}
  const c = S.curve; let hit = null; c.forEach(q => { if (hit === null && q.v <= 531) hit = q.t; });
  res.push(S.best.val); tHit.push(hit);
  // validate the schedule against the simulator again
  const sc = S.bestSchedule(); if (sc.sum !== S.best.val) { console.log('mismatch', sc.sum, S.best.val); process.exit(1); }
}
const hits = res.filter(x => x === 531).length;
console.log(`runs=${runs} budget=${ms}ms  best=${Math.min(...res)} worst=${Math.max(...res)} mean=${(res.reduce((a, b) => a + b) / runs).toFixed(2)}  reached 531: ${hits}/${runs}`);
const th = tHit.filter(x => x !== null).sort((a, b) => a - b); if (th.length) console.log('time to 531 (ms): median', th[th.length >> 1], 'max', th[th.length - 1]);
console.log('mean gap to 531:', ((res.reduce((a, b) => a + b) / runs / 531 - 1) * 100).toFixed(2) + '%');
