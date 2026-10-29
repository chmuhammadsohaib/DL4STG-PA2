import cv2, numpy as np, json, os, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, pa2

v = (pa2.loadVideos()[:50, :, 0].float() / 255 * 255).round().to(torch.uint8).numpy()
dx, dy = [], []
for vid in v:
    for s in range(19):
        a, b = vid[s], vid[s + 1]
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        keep = (a > 51) & (b > 51) & (np.hypot(fl[..., 0], fl[..., 1]) > 0.5)
        dx.append(fl[..., 0][keep])
        dy.append(fl[..., 1][keep])
dx, dy = np.concatenate(dx), np.concatenate(dy)
th = np.degrees(np.arctan2(-dy, dx)) % 360
sp = np.hypot(dx, dy)
med = float(np.median(sp))
edges = np.arange(0, 361, 20)
frac = np.histogram(th, edges)[0] / len(th)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].bar(edges[:-1], frac, width=20, align='edge', edgecolor='k', color='tab:blue')
ax[0].axhline(1 / 18, color='r', ls='--', label='uniform (1/18)')
ax[0].set(xlim=(0, 360), xticks=range(0, 361, 60), xlabel=r'flow direction $\theta$ (deg; 0 = right, 90 = up)', ylabel='fraction of kept pixels', title='(a) direction')
ax[0].legend(loc='lower right')
sEdges = np.arange(0, np.ceil(sp.max()) + 0.25, 0.25)
ax[1].bar(sEdges[:-1], np.histogram(sp, sEdges)[0] / len(sp), width=0.25, align='edge', edgecolor='k', color='tab:green')
ax[1].axvline(med, color='r', ls='--', label=f'median = {med:#.3g} px/frame')
ax[1].set(xlim=(0, 7), xlabel='flow speed (px/frame)', ylabel='fraction of kept pixels', title='(b) speed')
ax[1].legend()
fig.tight_layout()
os.makedirs(os.path.join(pa2.root, 'figs'), exist_ok=True)
fig.savefig(os.path.join(pa2.root, 'figs', 'fig1.pdf'))
fig.savefig(os.path.join(pa2.root, 'figs', 'fig1.png'), dpi=150)
stats = dict(nKept=int(len(sp)), medianSpeed=med, dirFrac=frac.tolist(), dirMin=float(frac.min()), dirMax=float(frac.max()))
json.dump(stats, open(os.path.join(pa2.root, 'results', 'fig1.json'), 'w'), indent=1)
print(json.dumps(stats))
