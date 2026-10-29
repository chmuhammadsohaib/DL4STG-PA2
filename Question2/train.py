import sys, time, json, os, torch, pa2

name, seed = sys.argv[1], int(sys.argv[2])
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
arch, speedAug, symAug = pa2.runs[name]
dev = pa2.dev
torch.manual_seed(seed)
gen = torch.Generator().manual_seed(seed)
v = pa2.loadVideos()
tr, va = v[:8000].to(dev), v[8000:9000]
m = pa2.Net(**pa2.archs[arch]).to(dev)
opt = torch.optim.Adam(m.parameters(), 1e-3)
st = pa2.startsAug if speedAug else {1: pa2.starts[1]}
ks = torch.tensor(list(st))
maxT = torch.tensor([st[int(k)] for k in ks])
log = []
print(name, seed, arch, speedAug, symAug, pa2.nParams(m), dev, flush=True)
for ep in range(epochs):
    m.train()
    t0 = time.time()
    tot = torch.zeros((), device=dev)
    perm = torch.randperm(8000, generator=gen)
    for i in range(0, 8000, 32):
        b = perm[i:i + 32]
        n = len(b)
        kIdx = torch.randint(len(ks), (n,), generator=gen)
        k = ks[kIdx]
        t = (torch.rand(n, generator=gen) * maxT[kIdx]).long() + 1
        x, y, mask = pa2.window(tr[b.to(dev)], t, k)
        clip, mask = torch.cat([x, y], 1).float() / 255, mask.float()
        if symAug:
            g = torch.randint(8, (n,), generator=gen)
            clip = torch.stack([pa2.gApply(clip[j], int(g[j])) for j in range(n)])
        y = m(clip[:, :5])
        loss = ((((y - clip[:, 5:]) ** 2).mean((2, 3, 4)) * mask).sum(1) / mask.sum(1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.detach() * n
    trainLoss = tot.item() / 8000
    sec = time.time() - t0
    rec = dict(epoch=ep + 1, loss=trainLoss, sec=sec)
    if (ep + 1) % 5 == 0 or ep + 1 == epochs:
        m.eval()
        for k in (1, 2, 3):
            x, yt, mk = pa2.window(va, 1, k)
            nk = int(mk[0].sum())
            e = pa2.perVideoMse(pa2.predict(m, x)[:, :nk], yt[:, :nk].float() / 255)
            rec[f'valE2_{k}x'] = e[:, :2].mean().item()
            if k == 1: rec['valE5_1x'] = e.mean().item()
    log.append(rec)
    print(json.dumps(rec), flush=True)
os.makedirs(os.path.join(pa2.root, 'ckpt'), exist_ok=True)
torch.save(dict(arch=arch, name=name, seed=seed, speedAug=speedAug, symAug=symAug, state=m.state_dict(), log=log, dev=str(dev)), os.path.join(pa2.root, 'ckpt', f'{name}_s{seed}.pt'))
