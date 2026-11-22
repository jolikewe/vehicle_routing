/* Browser solver for the two-leg personnel transport problem of planck_optimiser.py.
 * Exact schedule simulator + simulated annealing over trip assignments.
 * Works in a Web Worker, on the main thread, and under node (module.exports). */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.VRP = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  var INF = Infinity;

  /* ---------- numpy-compatible random routes (legacy MT19937 + randint(5,12)) ---------- */
  function mt19937(seed) {
    var mt = new Uint32Array(624), idx = 624;
    mt[0] = seed >>> 0;
    for (var i = 1; i < 624; i++) {
      var s = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = (((((s & 0xffff0000) >>> 16) * 1812433253) << 16) + (s & 0x0000ffff) * 1812433253 + i) >>> 0;
    }
    return function () {
      if (idx >= 624) {
        for (var k = 0; k < 624; k++) {
          var y = (mt[k] & 0x80000000) | (mt[(k + 1) % 624] & 0x7fffffff);
          mt[k] = mt[(k + 397) % 624] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
        }
        idx = 0;
      }
      var x = mt[idx++];
      x ^= x >>> 11; x ^= (x << 7) & 0x9d2c5680; x ^= (x << 15) & 0xefc60000; x ^= x >>> 18;
      return x >>> 0;
    };
  }
  /* Same draw order as Model.generate_route: pickup x hub, then hub x destination, randint(5, 12). */
  function randomRoutes(nH, seed) {
    var g = mt19937(seed);
    function ri() { var v; do { v = g() & 7; } while (v > 6 - 1); return 5 + v; }
    var dSH = [], dHD = [], s, h;
    for (s = 0; s < 2; s++) { dSH.push([]); for (h = 0; h < nH; h++) dSH[s].push(ri()); }
    for (h = 0; h < nH; h++) { dHD.push([]); for (var d = 0; d < 2; d++) dHD[h].push(ri()); }
    return { dSH: dSH, dHD: dHD };
  }
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* ---------- instance ---------- */
  /* p: {counts:[n0,n1], nH, nB, nA, cap, R, load, dSH[2][nH], dHD[nH][2], fixedHub} */
  function makeInst(p) {
    var I = {
      counts: p.counts.slice(), nH: p.nH, nB: p.nB, nA: p.nA, cap: p.cap, R: p.R, load: p.load | 0,
      dSH: p.dSH, dHD: p.dHD, fixedHub: !!p.fixedHub, nD: 2
    };
    var N = I.counts[0] + I.counts[1];
    I.N = N;
    I.ps = new Int8Array(N);
    for (var n = 0; n < N; n++) I.ps[n] = n < I.counts[0] ? 0 : 1;   /* pickup k goes to destination k */
    var nbs = I.nB * I.R, nas = I.nA * I.R;
    I.bc = new Int32Array(nbs); I.bs = new Int32Array(nbs); I.ac = new Int32Array(nas); I.as_ = new Int32Array(nas);
    I.hubArr = new Int32Array(nbs); I.LH = new Int32Array(nbs);
    I.aReady = new Int32Array(nas); I.aDep = new Int32Array(nas); I.aArr = new Int32Array(nas); I.aTH = new Int32Array(nas);
    I.aD0 = new Int8Array(I.nA);
    I.out = { sum: 0, mk: 0, waitSum: 0 };
    return I;
  }
  function fixedHubOf(I, n) { return n % I.nH; }   /* repo notebook: person n is pre-assigned hub n mod #hubs */

  /* ---------- exact simulator ----------
   * State: bSlot[n] = beta vehicle*R + trip, aSlot[n] likewise, hub[n].
   * Mirrors constraints i-xiii of planck_optimiser.py with every time at its earliest value:
   *  beta trip r : origin -> hub (loaded) -> next origin (empty); alpha trip r : prev dest -> hub (empty) -> dest (loaded)
   *  T_hub(0) = l + d ; T_hub(r+1) = T_hub(r) + l + d(hub_r, origin_{r+1}) + l + d(origin_{r+1}, hub_{r+1})
   *  alpha loaded arrival T_D = max(T_hubA + l, latest rider arrival at hub) + d(hub, D)
   * jit=true additionally right-shifts bus trips as late as possible (only hub waiting changes).
   * Returns false when the state breaks a rule (capacity, mixed hub/origin/destination in one trip). */
  function simulate(I, bSlot, aSlot, hub, jit) {
    var N = I.N, R = I.R, nB = I.nB, nA = I.nA, nH = I.nH, cap = I.cap, l = I.load, dSH = I.dSH, dHD = I.dHD, ps = I.ps;
    var bc = I.bc, bs = I.bs, ac = I.ac, as_ = I.as_, hubArr = I.hubArr, aReady = I.aReady, aDep = I.aDep, aArr = I.aArr, aTH = I.aTH;
    bc.fill(0); bs.fill(-1); ac.fill(0); as_.fill(-1); aReady.fill(0);
    var n, s, sg;
    for (n = 0; n < N; n++) {
      s = bSlot[n]; if (++bc[s] > cap) return false;
      sg = ps[n] * nH + hub[n]; if (bs[s] < 0) bs[s] = sg; else if (bs[s] !== sg) return false;
      s = aSlot[n]; if (++ac[s] > cap) return false;
      sg = hub[n] * 2 + ps[n]; if (as_[s] < 0) as_[s] = sg; else if (as_[s] !== sg) return false;
    }
    var v, r, o, h, d, TH, prevH, prevT;
    for (v = 0; v < nB; v++) {
      prevH = -1; prevT = 0;
      for (r = 0; r < R; r++) {
        s = v * R + r; if (!bc[s]) continue;
        sg = bs[s]; o = (sg / nH) | 0; h = sg % nH;
        TH = prevH < 0 ? l + dSH[o][h] : prevT + 2 * l + dSH[o][prevH] + dSH[o][h];
        hubArr[s] = TH; prevH = h; prevT = TH;
      }
    }
    for (n = 0; n < N; n++) { s = aSlot[n]; var t = hubArr[bSlot[n]]; if (t > aReady[s]) aReady[s] = t; }
    for (v = 0; v < nA; v++) {
      var prevD = -1; prevT = 0;
      for (r = 0; r < R; r++) {
        s = v * R + r; if (!ac[s]) continue;
        sg = as_[s]; h = (sg / 2) | 0; d = sg - 2 * h;
        if (prevD < 0) { I.aD0[v] = dHD[h][0] <= dHD[h][1] ? 0 : 1; TH = l + dHD[h][I.aD0[v]]; }
        else TH = prevT + l + dHD[h][prevD];
        var TD = Math.max(TH + l, aReady[s]) + dHD[h][d];
        aTH[s] = TH; aArr[s] = TD; aDep[s] = TD - dHD[h][d]; prevD = d; prevT = TD;
      }
    }
    var sum = 0, mk = 0, waitSum = 0;
    for (n = 0; n < N; n++) { var a = aArr[aSlot[n]]; sum += a; if (a > mk) mk = a; }
    if (jit) {
      var LH = I.LH; LH.fill(1 << 28);
      for (n = 0; n < N; n++) { s = bSlot[n]; var dp = aDep[aSlot[n]]; if (dp < LH[s]) LH[s] = dp; }
      for (v = 0; v < nB; v++) {
        var nxt = -1;
        for (r = R - 1; r >= 0; r--) {
          s = v * R + r; if (!bc[s]) continue;
          if (nxt >= 0) {
            sg = bs[nxt]; var no = (sg / nH) | 0, nh = sg % nH; var h0 = bs[s] % nH;
            var lim = LH[nxt] - 2 * l - dSH[no][h0] - dSH[no][nh];
            if (lim < LH[s]) LH[s] = lim;
          }
          nxt = s;
        }
      }
      for (n = 0; n < N; n++) waitSum += aDep[aSlot[n]] - LH[bSlot[n]];
    } else {
      for (n = 0; n < N; n++) waitSum += aDep[aSlot[n]] - hubArr[bSlot[n]];
    }
    I.out.sum = sum; I.out.mk = mk; I.out.waitSum = waitSum;
    return true;
  }

  function objValue(I, obj, m) {
    var N = I.N;
    switch (obj.kind) {
      case 'makespan': return m.mk;
      case 'wait': return m.waitSum / N;
      case 'mix': return m.mk + obj.lambda * m.waitSum / N;
      default: return m.sum;
    }
  }
  function searchCost(I, obj, m) {   /* tiny tie-breakers keep the search moving on plateaus */
    var N = I.N;
    switch (obj.kind) {
      case 'makespan': return m.mk + 0.01 * m.sum / N;
      case 'wait': return m.waitSum / N + 0.01 * m.sum / N;
      case 'mix': return m.mk + obj.lambda * m.waitSum / N + 0.005 * m.sum / N;
      default: return m.sum;
    }
  }
  function needsJit(obj) { return obj.kind === 'wait' || obj.kind === 'mix'; }

  /* ---------- full schedule (same shape as schedule.json) ---------- */
  function buildSchedule(I, st, jit) {
    if (!simulate(I, st.bSlot, st.aSlot, st.hub, jit)) return null;
    var N = I.N, R = I.R, nH = I.nH, l = I.load, dSH = I.dSH, dHD = I.dHD;
    var people = [], legs = [], n, v, r, s;
    var TB = function (sl) { return jit ? I.LH[sl] : I.hubArr[sl]; };
    var ridersB = {}, ridersA = {};
    for (n = 0; n < N; n++) { (ridersB[st.bSlot[n]] = ridersB[st.bSlot[n]] || []).push(n); (ridersA[st.aSlot[n]] = ridersA[st.aSlot[n]] || []).push(n); }
    for (v = 0; v < I.nB; v++) {
      var trips = []; for (r = 0; r < R; r++) if (I.bc[v * R + r]) trips.push(v * R + r);
      trips.forEach(function (sl, k) {
        var sg = I.bs[sl], o = (sg / nH) | 0, h = sg % nH, T = TB(sl);
        legs.push({ vehicle: 'Vb' + (v + 1), kind: 'beta', trip: k, frm: 'S' + (o + 1), to: 'H' + (h + 1), depart: T - dSH[o][h], arrive: T, loaded: true, riders: ridersB[sl].slice() });
        var no = k + 1 < trips.length ? (I.bs[trips[k + 1]] / nH) | 0 : o;
        legs.push({ vehicle: 'Vb' + (v + 1), kind: 'beta', trip: k, frm: 'H' + (h + 1), to: 'S' + (no + 1), depart: T + l, arrive: T + l + dSH[no][h], loaded: false, riders: [] });
      });
    }
    for (v = 0; v < I.nA; v++) {
      var tr = []; for (r = 0; r < R; r++) if (I.ac[v * R + r]) tr.push(v * R + r);
      var prevD = -1;
      tr.forEach(function (sl, k) {
        var sg = I.as_[sl], h = (sg / 2) | 0, d = sg - 2 * h, from = k === 0 ? I.aD0[v] : prevD;
        legs.push({ vehicle: 'Va' + (v + 1), kind: 'alpha', trip: k, frm: 'D' + (from + 1), to: 'H' + (h + 1), depart: I.aTH[sl] - dHD[h][from], arrive: I.aTH[sl], loaded: false, riders: [] });
        legs.push({ vehicle: 'Va' + (v + 1), kind: 'alpha', trip: k, frm: 'H' + (h + 1), to: 'D' + (d + 1), depart: I.aDep[sl], arrive: I.aArr[sl], loaded: true, riders: ridersA[sl].slice() });
        prevD = d;
      });
    }
    var bIdx = {}, aIdx = {};
    for (v = 0; v < I.nB; v++) { var k1 = 0; for (r = 0; r < R; r++) if (I.bc[v * R + r]) bIdx[v * R + r] = k1++; }
    for (v = 0; v < I.nA; v++) { var k2 = 0; for (r = 0; r < R; r++) if (I.ac[v * R + r]) aIdx[v * R + r] = k2++; }
    for (n = 0; n < N; n++) {
      var bs_ = st.bSlot[n], as2 = st.aSlot[n];
      people.push({
        id: n, S: 'S' + (I.ps[n] + 1), H: 'H' + (st.hub[n] + 1), D: 'D' + (I.ps[n] + 1),
        beta: 'Vb' + (((bs_ / R) | 0) + 1), beta_trip: bIdx[bs_], alpha: 'Va' + (((as2 / R) | 0) + 1), alpha_trip: aIdx[as2],
        depart_S: TB(bs_) - dSH[I.ps[n]][st.hub[n]], arrive_H: TB(bs_), depart_H: I.aDep[as2], arrive_D: I.aArr[as2]
      });
    }
    legs.sort(function (a, b) { return a.vehicle < b.vehicle ? -1 : a.vehicle > b.vehicle ? 1 : a.depart - b.depart || a.arrive - b.arrive; });
    var m = I.out, mk = m.mk;
    return { people: people, legs: legs, sum: m.sum, makespan: mk, avgWait: m.waitSum / N, waitSum: m.waitSum, jit: !!jit };
  }

  /* encode an existing schedule (people[].beta, beta_trip, ...) into a state */
  function stateFromSchedule(I, sched) {
    var N = I.N, R = I.R, bSlot = new Int16Array(N), aSlot = new Int16Array(N), hub = new Int16Array(N);
    sched.people.forEach(function (p) {
      bSlot[p.id] = (+p.beta.slice(2) - 1) * R + p.beta_trip;
      aSlot[p.id] = (+p.alpha.slice(2) - 1) * R + p.alpha_trip;
      hub[p.id] = +p.H.slice(1) - 1;
    });
    return { bSlot: bSlot, aSlot: aSlot, hub: hub };
  }

  /* ---------- feasibility ---------- */
  function tripsNeeded(I) {
    var need = 0, k, h, cnt;
    for (k = 0; k < 2; k++) {
      if (I.fixedHub) {
        for (h = 0; h < I.nH; h++) {
          cnt = 0; for (var n = 0; n < I.N; n++) if (I.ps[n] === k && fixedHubOf(I, n) === h) cnt++;
          need += Math.ceil(cnt / I.cap);
        }
      } else need += Math.ceil(I.counts[k] / I.cap);
    }
    return need;
  }
  function feasibility(I) {
    if (I.N === 0) return { ok: false, msg: 'Add at least one person.' };
    var need = tripsNeeded(I);
    if (need > I.nB * I.R) return { ok: false, msg: 'Not enough bus trips: ' + need + ' trips are needed but ' + I.nB + ' bus(es) x ' + I.R + ' trip(s) give ' + (I.nB * I.R) + '. Add buses, trips or seats.' };
    if (need > I.nA * I.R) return { ok: false, msg: 'Not enough boat trips: ' + need + ' trips are needed but ' + I.nA + ' boat(s) x ' + I.R + ' trip(s) give ' + (I.nA * I.R) + '. Add boats, trips or seats.' };
    return { ok: true, need: need };
  }

  /* ---------- constructive greedy (randomised variants for restarts) ---------- */
  function construct(I, rnd, randomize) {
    var N = I.N, R = I.R, nH = I.nH, l = I.load, dSH = I.dSH, dHD = I.dHD, k, h, n;
    var hub = new Int16Array(N), groups = [];   /* groups: {o, h, ids} = one bus trip */
    for (k = 0; k < 2; k++) {
      var ids = []; for (n = 0; n < N; n++) if (I.ps[n] === k) ids.push(n);
      if (I.fixedHub) {
        for (h = 0; h < nH; h++) {
          var sub = ids.filter(function (x) { return fixedHubOf(I, x) === h; });
          for (var i = 0; i < sub.length; i += I.cap) groups.push({ o: k, h: h, ids: sub.slice(i, i + I.cap) });
        }
      } else {
        var best = 0, bc2 = INF;
        for (h = 0; h < nH; h++) { var c = dSH[k][h] + dHD[h][k]; if (c < bc2) { bc2 = c; best = h; } }
        for (var j = 0; j < ids.length; j += I.cap) {
          var hh = randomize && rnd() < 0.5 ? (rnd() * nH) | 0 : best;
          groups.push({ o: k, h: hh, ids: ids.slice(j, j + I.cap) });
        }
      }
    }
    groups.forEach(function (g) { g.ids.forEach(function (x) { hub[x] = g.h; }); g.key = dSH[g.o][g.h] + (randomize ? rnd() * 8 : 0); });
    groups.sort(function (a, b) { return a.key - b.key; });
    var bSlot = new Int16Array(N), aSlot = new Int16Array(N), cnt = new Int32Array(I.nB), prevH = new Int32Array(I.nB).fill(-1), prevT = new Int32Array(I.nB);
    var arr = new Int32Array(N), v;
    for (var gi = 0; gi < groups.length; gi++) {
      var g = groups[gi], bv = -1, bt = INF, fin;
      for (v = 0; v < I.nB; v++) {
        if (cnt[v] >= R) continue;
        fin = prevH[v] < 0 ? l + dSH[g.o][g.h] : prevT[v] + 2 * l + dSH[g.o][prevH[v]] + dSH[g.o][g.h];
        if (randomize) fin += rnd() * 6;
        if (fin < bt) { bt = fin; bv = v; }
      }
      if (bv < 0) return null;
      fin = prevH[bv] < 0 ? l + dSH[g.o][g.h] : prevT[bv] + 2 * l + dSH[g.o][prevH[bv]] + dSH[g.o][g.h];
      g.ids.forEach(function (x) { bSlot[x] = bv * R + cnt[bv]; arr[x] = fin; });
      cnt[bv]++; prevH[bv] = g.h; prevT[bv] = fin;
    }
    /* alpha trips: per (hub, destination) chunk people by arrival time */
    var ag = [];
    for (h = 0; h < nH; h++) for (k = 0; k < 2; k++) {
      var m = []; for (n = 0; n < N; n++) if (hub[n] === h && I.ps[n] === k) m.push(n);
      m.sort(function (a, b) { return arr[a] - arr[b]; });
      for (var q = 0; q < m.length; q += I.cap) {
        var ch = m.slice(q, q + I.cap), rdy = 0; ch.forEach(function (x) { if (arr[x] > rdy) rdy = arr[x]; });
        ag.push({ h: h, d: k, ids: ch, ready: rdy + (randomize ? rnd() * 6 : 0) });
      }
    }
    ag.sort(function (a, b) { return a.ready - b.ready; });
    var acnt = new Int32Array(I.nA), aprevD = new Int32Array(I.nA).fill(-1), aprevT = new Int32Array(I.nA);
    for (var ai = 0; ai < ag.length; ai++) {
      var t = ag[ai], av = -1, at = INF, TH, TD;
      for (v = 0; v < I.nA; v++) {
        if (acnt[v] >= R) continue;
        if (aprevD[v] < 0) TH = l + Math.min(dHD[t.h][0], dHD[t.h][1]); else TH = aprevT[v] + l + dHD[t.h][aprevD[v]];
        TD = Math.max(TH + l, t.ready) + dHD[t.h][t.d];
        if (TD < at) { at = TD; av = v; }
      }
      if (av < 0) return null;
      t.ids.forEach(function (x) { aSlot[x] = av * R + acnt[av]; });
      acnt[av]++; aprevD[av] = t.d; aprevT[av] = at;
    }
    return { bSlot: bSlot, aSlot: aSlot, hub: hub };
  }

  /* ---------- simulated annealing ---------- */
  function Solver(I, obj, seed, budgetMs) {
    this.I = I; this.obj = obj; this.rnd = mulberry32(seed | 0); this.budget = budgetMs || 10000;
    this.jit = needsJit(obj); this.elapsed = 0; this.iters = 0; this.cycle = 0; this.curve = []; this.done = false;
    this.error = null; this.accepted = 0; this.best = null;
    var f = feasibility(I);
    if (!f.ok) { this.error = f.msg; this.done = true; return; }
    var st = construct(I, this.rnd, false);
    if (!st || !simulate(I, st.bSlot, st.aSlot, st.hub, this.jit)) { this.error = 'Could not build a starting schedule for these settings.'; this.done = true; return; }
    this.cur = st; this.curCost = searchCost(I, obj, I.out);
    this.cand = { bSlot: st.bSlot.slice(), aSlot: st.aSlot.slice(), hub: st.hub.slice() };
    this.setBest(st, 0);
    this.initCost = this.curCost;
    this.scale = Math.max(0.5, 0.05 * this.curCost);
    this.cycIters = 25000 + 3000 * Math.min(I.nB * I.R + I.nA * I.R, 40);
    this.startCycle(true);
  }
  Solver.prototype.setBest = function (st, tms) {
    var I = this.I; simulate(I, st.bSlot, st.aSlot, st.hub, this.jit);
    var m = { sum: I.out.sum, mk: I.out.mk, waitSum: I.out.waitSum };
    this.best = { state: { bSlot: st.bSlot.slice(), aSlot: st.aSlot.slice(), hub: st.hub.slice() }, cost: searchCost(I, this.obj, m), val: objValue(I, this.obj, m), m: m };
    this.curve.push({ t: tms, v: this.best.val, it: this.iters });
    this.dirty = true;
  };
  Solver.prototype.startCycle = function (first) {
    var I = this.I, src;
    var fresh = !first && this.cycle % 5 === 4;
    if (fresh) src = construct(I, this.rnd, true);
    if (!src) src = this.best.state;
    this.cur.bSlot.set(src.bSlot); this.cur.aSlot.set(src.aSlot); this.cur.hub.set(src.hub);
    simulate(I, this.cur.bSlot, this.cur.aSlot, this.cur.hub, this.jit);
    this.curCost = searchCost(I, this.obj, I.out);
    this.T0 = this.scale * (first ? 1 : fresh ? 0.8 : 0.45);
    this.T1 = Math.max(this.T0 / 80, 1e-3);
    this.cycleLeft = this.cycIters; this.cycTotal = this.cycIters; this.cycle++;
  };
  Solver.prototype.propose = function () {
    var I = this.I, c = this.cand, rnd = this.rnd, N = I.N, R = I.R, i, n, m, s, s2, v, x = rnd();
    var nbs = I.nB * I.R, nas = I.nA * I.R, free = !I.fixedHub;
    var pb = free ? [0.16, 0.32, 0.42, 0.52, 0.64, 0.72, 0.76, 0.80, 0.88, 0.93] : [0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.90, 0.90, 0.90, 1];
    if (x < pb[0]) {                       /* swap two bus trips (any vehicles) */
      if (nbs < 2) return false; s = (rnd() * nbs) | 0; s2 = (rnd() * nbs) | 0; if (s === s2) return false;
      for (i = 0; i < N; i++) { if (c.bSlot[i] === s) c.bSlot[i] = s2; else if (c.bSlot[i] === s2) c.bSlot[i] = s; }
    } else if (x < pb[1]) {                /* swap two boat trips */
      if (nas < 2) return false; s = (rnd() * nas) | 0; s2 = (rnd() * nas) | 0; if (s === s2) return false;
      for (i = 0; i < N; i++) { if (c.aSlot[i] === s) c.aSlot[i] = s2; else if (c.aSlot[i] === s2) c.aSlot[i] = s; }
    } else if (x < pb[2] || x < pb[3]) {   /* re-sequence one vehicle: pull a trip to another position */
      var beta = x < pb[2], nv = beta ? I.nB : I.nA; if (R < 2) return false;
      v = (rnd() * nv) | 0; var r1 = (rnd() * R) | 0, r2 = (rnd() * R) | 0; if (r1 === r2) return false;
      var arr = beta ? c.bSlot : c.aSlot;
      for (i = 0; i < N; i++) {
        s = arr[i]; if (((s / R) | 0) !== v) continue; var r = s - v * R, nr = r;
        if (r === r1) nr = r2; else if (r1 < r2 && r > r1 && r <= r2) nr = r - 1; else if (r1 > r2 && r >= r2 && r < r1) nr = r + 1;
        arr[i] = v * R + nr;
      }
    } else if (x < pb[4]) {                /* swap two people between bus trips or between boat trips */
      n = (rnd() * N) | 0; m = (rnd() * N) | 0; if (n === m) return false;
      var useB = rnd() < 0.5;
      if (I.ps[n] !== I.ps[m] || c.hub[n] !== c.hub[m]) return false;
      if (useB) { s = c.bSlot[n]; c.bSlot[n] = c.bSlot[m]; c.bSlot[m] = s; } else { s = c.aSlot[n]; c.aSlot[n] = c.aSlot[m]; c.aSlot[m] = s; }
    } else if (x < pb[5]) {                /* move one person into another trip (bus or boat) */
      n = (rnd() * N) | 0;
      if (rnd() < 0.5) c.bSlot[n] = (rnd() * nbs) | 0; else c.aSlot[n] = (rnd() * nas) | 0;
    } else if (x < pb[6]) {                /* move one person to another hub (free-hub mode) */
      n = (rnd() * N) | 0; c.hub[n] = (rnd() * I.nH) | 0; c.bSlot[n] = (rnd() * nbs) | 0; c.aSlot[n] = (rnd() * nas) | 0;
    } else if (x < pb[7]) {                /* swap two people from the same pickup who use different hubs */
      n = (rnd() * N) | 0; m = (rnd() * N) | 0; if (n === m || I.ps[n] !== I.ps[m] || c.hub[n] === c.hub[m]) return false;
      s = c.hub[n]; c.hub[n] = c.hub[m]; c.hub[m] = s; s = c.bSlot[n]; c.bSlot[n] = c.bSlot[m]; c.bSlot[m] = s; s = c.aSlot[n]; c.aSlot[n] = c.aSlot[m]; c.aSlot[m] = s;
    } else if (x < pb[8]) {                /* retarget a whole trip pair to another hub */
      n = (rnd() * N) | 0; var h2 = (rnd() * I.nH) | 0, bsn = c.bSlot[n], asn = c.aSlot[n];
      for (i = 0; i < N; i++) if (c.bSlot[i] === bsn || c.aSlot[i] === asn) c.hub[i] = h2;
    } else {
      /* swap whole trip contents AND the alpha order: pair move keeps beta/alpha order consistent */
      if (nbs < 2) return false; s = (rnd() * nbs) | 0; s2 = (rnd() * nbs) | 0; if (s === s2) return false;
      for (i = 0; i < N; i++) { if (c.bSlot[i] === s) c.bSlot[i] = s2; else if (c.bSlot[i] === s2) c.bSlot[i] = s; }
      if (nas >= 2) { s = (rnd() * nas) | 0; s2 = (rnd() * nas) | 0; if (s !== s2) for (i = 0; i < N; i++) { if (c.aSlot[i] === s) c.aSlot[i] = s2; else if (c.aSlot[i] === s2) c.aSlot[i] = s; } }
    }
    return true;
  };
  /* run for about ms milliseconds; returns true when the search has finished */
  Solver.prototype.step = function (ms) {
    if (this.done) return true;
    var I = this.I, c = this.cand, cur = this.cur, t0 = Date.now(), tn = 0;
    var logR = Math.log(this.T1 / this.T0);
    for (;;) {
      for (var k = 0; k < 200; k++) {
        c.bSlot.set(cur.bSlot); c.aSlot.set(cur.aSlot); c.hub.set(cur.hub);
        this.iters++; this.cycleLeft--;
        if (this.propose() && simulate(I, c.bSlot, c.aSlot, c.hub, this.jit)) {
          var cost = searchCost(I, this.obj, I.out), T = this.T0 * Math.exp(logR * (1 - this.cycleLeft / this.cycTotal));
          var dlt = cost - this.curCost;
          if (dlt <= 0 || this.rnd() < Math.exp(-dlt / T)) {
            cur.bSlot.set(c.bSlot); cur.aSlot.set(c.aSlot); cur.hub.set(c.hub); this.curCost = cost; this.accepted++;
            if (cost < this.best.cost - 1e-9) this.setBest(c, this.elapsed + (Date.now() - t0));
          }
        }
        if (this.cycleLeft <= 0) this.startCycle(false);
      }
      tn = Date.now() - t0;
      if (tn >= ms) break;
    }
    this.elapsed += tn;
    if (this.elapsed >= this.budget) this.done = true;
    return this.done;
  };
  Solver.prototype.bestSchedule = function () { return buildSchedule(this.I, this.best.state, this.jit); };
  Solver.prototype.curNow = function () { return this.curCost; };

  return {
    makeInst: makeInst, simulate: simulate, buildSchedule: buildSchedule, stateFromSchedule: stateFromSchedule,
    feasibility: feasibility, Solver: Solver, randomRoutes: randomRoutes, objValue: objValue, needsJit: needsJit,
    construct: construct, mulberry32: mulberry32
  };
});
