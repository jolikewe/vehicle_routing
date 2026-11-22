/* Runs a Solver in slices and reports progress. Used inside a Web Worker, or on the main thread as a fallback. */
function makeEngine(post) {
  var S = null, timer = null, lastPost = 0, sent = 0, watch = false;
  function send(final) {
    var now = Date.now();
    if (!final && now - lastPost < 90) return;
    lastPost = now;
    var msg = { type: 'prog', iters: S.iters, elapsed: S.elapsed, val: S.best.val, m: S.best.m, done: !!S.done, newPts: S.curve.slice(sent), cyc: S.cycle };
    sent = S.curve.length;
    if (S.dirty) { msg.sched = S.bestSchedule(); S.dirty = false; }
    post(msg);
  }
  function loop() {
    if (!S) return;
    var slow = watch && S.elapsed < 1500;
    S.step(slow ? 3 : 30);
    send(S.done);
    if (!S.done) timer = setTimeout(loop, slow ? 40 : 0);
  }
  return {
    handle: function (m) {
      clearTimeout(timer);
      if (m.cmd === 'start') {
        S = new VRP.Solver(VRP.makeInst(m.inst), m.obj, m.seed, m.budget);
        sent = 0; watch = !!m.watch;
        if (S.error) { S = null; post({ type: 'error', msg: 'Cannot plan this setup.' }); return; }
        lastPost = 0; send(false); timer = setTimeout(loop, 0);
      } else if (m.cmd === 'stop') {
        if (S) { S.done = true; send(true); }
      }
    }
  };
}
