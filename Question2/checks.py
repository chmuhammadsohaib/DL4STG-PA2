import torch, json, os, pa2

torch.manual_seed(0)
x = torch.rand(20, 1, 64, 64)
p = torch.zeros(20, 1, 64, 64)
p[7, 0, 10, 20] = 1
num = torch.arange(1, 21).float()[None, :, None, None, None].expand(1, 20, 1, 64, 64)
r4 = pa2.rot(pa2.rot(pa2.rot(pa2.rot(x))))
pairs = {'a': (pa2.mirror(pa2.mirror(x)), x), 'b': (r4, x), 'c': (pa2.mirror(pa2.rot(pa2.mirror(x))), pa2.rot(x, -1))}
rows = {k: dict(maxAbs=(u - w).abs().max().item(), equal=torch.equal(u, w)) for k, (u, w) in pairs.items()}
rows['d'] = tuple(pa2.mirror(p).nonzero()[0, [0, 2, 3]].tolist())
rows['e'] = tuple(pa2.rot(p).nonzero()[0, [0, 2, 3]].tolist())
for c, k in zip('fgh', (1, 2, 3)):
    xi, yi, mk = pa2.window(num, 1, k)
    rows[c] = dict(inp=[int(f) for f in xi[0, :, 0, 0, 0]], tgt=[int(f) for f in yi[0][mk[0]][:, 0, 0, 0]])
os.makedirs(os.path.join(pa2.root, 'results'), exist_ok=True)
json.dump(rows, open(os.path.join(pa2.root, 'results', 'table1.json'), 'w'), indent=1)
print(json.dumps(rows))
