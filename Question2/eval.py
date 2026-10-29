import torch, json, os, time, numpy as np, pa2

dev = pa2.dev
sync = torch.mps.synchronize if dev.type == 'mps' else torch.cuda.synchronize if dev.type == 'cuda' else (lambda: None)
te = pa2.loadVideos()[9000:]
X1, Y1, _ = pa2.window(te, 1, 1)
X2, Y2, _ = pa2.window(te, 1, 2)
X3, Y3, _ = pa2.window(te, 1, 3)
Y1, Y2, Y3 = Y1.float() / 255, Y2.float() / 255, Y3[:, :2].float() / 255
models = [('Blank', 'det', pa2.Blank(), None), ('CopyLast', 'det', pa2.CopyLast(), None)]
for name in ['A', 'B', 'C', 'D']:
    for seed in (0, 1):
        if os.path.exists(os.path.join(pa2.root, 'ckpt', f'{name}_s{seed}.pt')):
            m, ck = pa2.loadModel(name, seed)
            models.append((name, seed, m, ck))
            if name in ('B', 'D'): models.append(('Sym' + name, seed, pa2.Sym(m), ck))
res = json.load(open(os.path.join(pa2.root, 'results', 'metrics.json'))) if os.path.exists(os.path.join(pa2.root, 'results', 'metrics.json')) else {}
os.makedirs(os.path.join(pa2.root, 'results', 'pervideo'), exist_ok=True)
for key, seed, m, ck in models:
    m = m.to(dev).eval()
    p1 = pa2.predict(m, X1)
    pM, pR = pa2.predict(m, pa2.mirror(X1)), pa2.predict(m, pa2.rot(X1))
    e1 = pa2.perVideoMse(p1, Y1)
    cM, eM = pa2.perVideoMse(pM, pa2.mirror(p1)).mean(1), pa2.perVideoMse(pM, pa2.mirror(Y1)).mean(1)
    cR, eR = pa2.perVideoMse(pR, pa2.rot(p1)).mean(1), pa2.perVideoMse(pR, pa2.rot(Y1)).mean(1)
    e2 = pa2.perVideoMse(pa2.predict(m, X2), Y2)
    e3 = pa2.perVideoMse(pa2.predict(m, X3)[:, :2], Y3)
    with torch.no_grad():
        m(X1[:32].to(dev).float() / 255)
        tInf = 0
        for i in range(0, 1000, 32):
            xb = X1[i:i + 32].to(dev).float() / 255
            sync()
            t0 = time.perf_counter()
            m(xb)
            sync()
            tInf += time.perf_counter() - t0
    r = dict(mseH=e1.mean(0).tolist(), E5=e1.mean().item(), E2_1x=e1[:, :2].mean().item(), CM=cM.mean().item(), EM=eM.mean().item(), CR=cR.mean().item(), ER=eR.mean().item(), E5_2x=e2.mean().item(), E2_2x=e2[:, :2].mean().item(), E2_3x=e3.mean().item(), params=pa2.nParams(m), trainSec=float(np.mean([l['sec'] for l in ck['log']])) if ck else None, infSec=tInf)
    res.setdefault(key, {})[str(seed)] = r
    np.savez(os.path.join(pa2.root, 'results', 'pervideo', f'{key}_s{seed}.npz'), e1=e1.numpy(), cM=cM.numpy(), eM=eM.numpy(), cR=cR.numpy(), eR=eR.numpy(), e2=e2.numpy(), e3=e3.numpy())
    print(key, seed, json.dumps({k: (round(w, 6) if isinstance(w, float) else w) for k, w in r.items() if k != 'mseH'}), flush=True)
json.dump(res, open(os.path.join(pa2.root, 'results', 'metrics.json'), 'w'), indent=1)
