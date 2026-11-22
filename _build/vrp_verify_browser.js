/* Runs the browser solver 20 times per objective on the repo instance and writes the numbers shown on the page.
 * Usage: node vrp_verify_browser.js [runs] [ms]   (reads cpsat_reference.json, writes reference.json) */
const V = require('./vrp_solver.js'), fs = require('fs'), path = require('path');
const sch = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'schedule.json'), 'utf8'));
const cp = JSON.parse(fs.readFileSync(path.join(__dirname, 'cpsat_reference.json'), 'utf8'));
const p = sch.params, r = sch.routes, nH = p.num_points_holding, dSH = [[], []], dHD = [];
for (let s = 0; s < 2; s++) for (let h = 0; h < nH; h++) dSH[s].push(r[`H${h + 1}-S${s + 1}`]);
for (let h = 0; h < nH; h++) dHD.push([r[`H${h + 1}-D1`], r[`H${h + 1}-D2`]]);
const P = { counts: [9, 9], nH, nB: 2, nA: 2, cap: 3, R: 3, load: 0, dSH, dHD, fixedHub: true };
const runs = +process.argv[2] || 20, ms = +process.argv[3] || 3000;
const out = { runs, ms, browser: { runs }, cpsat: cp };
for (const [name, obj] of [['sum', { kind: 'sum' }], ['makespan', { kind: 'makespan' }], ['wait', { kind: 'wait' }], ['mix', { kind: 'mix', lambda: 0.5 }]]) {
  const vals = [];
  for (let k = 0; k < runs; k++) { const S = new V.Solver(V.makeInst(P), obj, 500 + k, ms); while (!S.step(50)) {} vals.push(S.best.val); }
  const ref = cp[name].value, best = Math.min(...vals);
  out.browser[name] = { best, worst: Math.max(...vals), mean: vals.reduce((a, b) => a + b) / runs, hits: vals.filter(v => Math.abs(v - ref) < 1e-6).length };
  console.log(name, JSON.stringify(out.browser[name]), 'CP-SAT', ref);
}
fs.writeFileSync(path.join(__dirname, 'reference.json'), JSON.stringify(out, null, 1));
/* free-hub variant (the planner chooses hubs): browser results next to the CP-SAT values in cpsat_free_hub.json */
const fr = JSON.parse(fs.readFileSync(path.join(__dirname, 'cpsat_free_hub.json'), 'utf8')); out.free = { cpsat: fr, browser: {} };
for (const [name, obj] of [['sum', { kind: 'sum' }], ['makespan', { kind: 'makespan' }]]) {
  const vals = []; for (let k = 0; k < runs; k++) { const S = new V.Solver(V.makeInst(Object.assign({}, P, { fixedHub: false })), obj, 900 + k, ms); while (!S.step(50)) {} vals.push(S.best.val); }
  out.free.browser[name] = { best: Math.min(...vals), worst: Math.max(...vals), mean: vals.reduce((a, b) => a + b) / runs };
  console.log('free hubs', name, JSON.stringify(out.free.browser[name]), 'CP-SAT', fr[name].value, 'bound', fr[name].bound);
}
fs.writeFileSync(path.join(__dirname, 'reference.json'), JSON.stringify(out, null, 1));
