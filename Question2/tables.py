import json, os, numpy as np, pa2

rd, td = os.path.join(pa2.root, 'results'), os.path.join(pa2.root, '..', 'report', 'tables')
os.makedirs(td, exist_ok=True)
res = json.load(open(os.path.join(rd, 'metrics.json')))
t1 = json.load(open(os.path.join(rd, 'table1.json')))
f1 = json.load(open(os.path.join(rd, 'fig1.json')))

def fmt(x, sci=True):
    if x is None: return '--'
    if x == 0: return '0'
    if sci and abs(x) < 1e-3:
        m, e = f'{x:.2e}'.split('e')
        return f'${m}\\times10^{{{int(e)}}}$'
    if abs(x) >= 1000: return f'{int(round(x)):,}'.replace(',', '{,}')
    return f'{x:#.3g}'.rstrip('.')

def cell(key, col, h=None, sci=True):
    r = res[key]
    vs = [(r[s]['mseH'][h] if h is not None else r[s][col]) for s in sorted(r)]
    if None in vs: return '--'
    if len(vs) == 1 or col == 'params': return fmt(vs[0], sci)
    return f'\\makecell{{{fmt(float(np.mean(vs)), sci)}\\\\{{\\tiny({fmt(vs[0], sci)}, {fmt(vs[1], sci)})}}}}'

names = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'Blank': 'Blank', 'CopyLast': 'Copy-last', 'SymB': '$\\bar f$(B)', 'SymD': '$\\bar f$(D)'}
macros = []
for k, r in res.items():
    for col in r[sorted(r)[0]]:
        if col == 'mseH':
            for h in range(5):
                vs = [r[s]['mseH'][h] for s in sorted(r)]
                macros.append((f'{k}-MSE{h + 1}', float(np.mean(vs))))
            continue
        vs = [r[s][col] for s in sorted(r)]
        if None in vs: continue
        macros.append((f'{k}-{col}', float(np.mean(vs))))
        for s, w in zip(sorted(r), vs): macros.append((f'{k}-{col}-{s}', w))
macros += [('flow-n', f1['nKept']), ('flow-med', f1['medianSpeed']), ('flow-min', f1['dirMin']), ('flow-max', f1['dirMax'])]
open(os.path.join(td, 'nums.tex'), 'w').write('\n'.join(f'\\expandafter\\def\\csname v-{k}\\endcsname{{{fmt(w) if isinstance(w, float) else f"{w:,}".replace(",", "{,}")}}}' for k, w in macros) + '\n')
rows = [('(a)', '$\\max|M(M(x))-x|$', '0', f"{fmt(t1['a']['maxAbs'])}, \\texttt{{torch.equal}}={t1['a']['equal']}"), ('(b)', '$\\max|R^4(x)-x|$', '0', f"{fmt(t1['b']['maxAbs'])}, \\texttt{{torch.equal}}={t1['b']['equal']}"), ('(c)', '$\\max|MRM(x)-R^{-1}(x)|$', '0', f"{fmt(t1['c']['maxAbs'])}, \\texttt{{torch.equal}}={t1['c']['equal']}"), ('(d)', 'bright pixel of $M(p)$ (frame, row, column)', '(7, 10, 43)', str(tuple(t1['d']))), ('(e)', 'bright pixel of $R(p)$ (frame, row, column)', '(7, 43, 10)', str(tuple(t1['e'])))]
rows += [(f'({c})', f'{k}$\\times$ input / target frames', e, f"{', '.join(map(str, t1[c]['inp']))} / {', '.join(map(str, t1[c]['tgt']))}") for c, k, e in (('f', 1, '1--5 / 6--10'), ('g', 2, '1, 3, 5, 7, 9 / 11, 13, 15, 17, 19'), ('h', 3, '1, 4, 7, 10, 13 / 16, 19'))]
open(os.path.join(td, 't1.tex'), 'w').write('\\begin{tabular}{@{}l l l l@{}}\n\\toprule\n& Check & Expected & Yours \\\\\n\\midrule\n' + '\n'.join(f'{a} & {b} & {c} & {d} \\\\' for a, b, c, d in rows) + '\n\\bottomrule\n\\end{tabular}\n')
hdr = lambda cols: '\\toprule\nModel & ' + ' & '.join(cols) + ' \\\\\n\\midrule\n'
t2 = '\\begin{tabular}{@{}l ' + 'c' * 7 + '@{}}\n' + hdr([f'MSE$_{h}$' for h in range(1, 6)] + ['$E^{(5)}$', 'params'])
t2 += '\n'.join(f'{names[k]} & ' + ' & '.join([cell(k, None, h) for h in range(5)] + [cell(k, 'E5'), cell(k, 'params')]) + ' \\\\' for k in ('A', 'Blank', 'CopyLast')) + '\n\\bottomrule\n\\end{tabular}\n'
open(os.path.join(td, 't2.tex'), 'w').write(t2)
cols3 = ['E5', 'CM', 'EM', 'CR', 'ER']
t3 = '\\begin{tabular}{@{}l ' + 'c' * 5 + '@{}}\n' + hdr(['$E^{(5)}$', '$C_M$', '$E_M$', '$C_R$', '$E_R$'])
t3 += '\n'.join(f'{names[k]} & ' + ' & '.join(cell(k, c) for c in cols3) + ' \\\\' for k in ('A', 'Blank', 'CopyLast')) + '\n\\bottomrule\n\\end{tabular}\n'
open(os.path.join(td, 't3.tex'), 'w').write(t3)
cols4 = ['E2_1x', 'E2_2x', 'E2_3x', 'E5', 'E5_2x']
t4 = '\\begin{tabular}{@{}l ' + 'c' * 5 + '@{}}\n' + hdr(['$E^{(2)}$ 1$\\times$', '$E^{(2)}$ 2$\\times$', '$E^{(2)}$ 3$\\times$', '$E^{(5)}$ 1$\\times$', '$E^{(5)}$ 2$\\times$'])
t4 += '\n'.join(f'{names[k]} & ' + ' & '.join(cell(k, c) for c in cols4) + ' \\\\' for k in ('A', 'CopyLast', 'Blank')) + '\n\\bottomrule\n\\end{tabular}\n'
open(os.path.join(td, 't4.tex'), 'w').write(t4)
cols5 = ['E5', 'CM', 'EM', 'CR', 'ER', 'E2_1x', 'E2_2x', 'E2_3x', 'params', 'trainSec', 'infSec']
hdr5 = ['$E^{(5)}$ 1$\\times$', '$C_M$', '$E_M$', '$C_R$', '$E_R$', '$E^{(2)}$ 1$\\times$', '$E^{(2)}$ 2$\\times$', '$E^{(2)}$ 3$\\times$', 'params', '\\makecell{train\\\\s/epoch}', '\\makecell{infer.\\\\s}']
for tn, keys in (('t5', ['A', 'B', 'C', 'D']), ('t6', ['SymB', 'SymD'])):
    t = '\\begin{tabular}{@{}l ' + 'c' * len(cols5) + '@{}}\n' + hdr(hdr5)
    t += '\n'.join(f'{names[k]} & ' + ' & '.join(cell(k, c) for c in cols5) + ' \\\\' for k in keys) + '\n\\bottomrule\n\\end{tabular}\n'
    open(os.path.join(td, f'{tn}.tex'), 'w').write(t)
