"""Assembles ../index.html from the template, the solver sources and the data files.
Run from the page folder:  python3 _build/vrp_build.py"""
import json, re
rd = lambda p: open(p, encoding='utf-8').read()
t = rd('_build/vrp_template.html')
ref = json.loads(rd('_build/reference.json'))
ref['note'] = ref.get('note', '')
t = t.replace('/*SOLVER_JS*/', rd('_build/vrp_solver.js')).replace('/*ENGINE_JS*/', rd('_build/vrp_engine.js'))
t = t.replace('/*DATA*/null', rd('schedule.json').strip()).replace('/*REF*/null', json.dumps(ref, separators=(',', ':')))
open('index.html', 'w', encoding='utf-8').write(t)
print('index.html', len(t), 'bytes')
