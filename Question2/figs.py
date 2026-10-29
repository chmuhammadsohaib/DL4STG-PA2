import torch, numpy as np, json, os, sys, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, pa2
from PIL import Image

stage = sys.argv[1]
fd, gd, rd = (os.path.join(pa2.root, d) for d in ('figs', 'gifs', 'results'))
os.makedirs(gd, exist_ok=True)
te = pa2.loadVideos()[9000:]
X1, Y1, _ = pa2.window(te, 1, 1)
tf = {'M': pa2.mirror, 'R': pa2.rot}

def grid(rows, labels, path, cols=None, title=None):
    fig, ax = plt.subplots(len(rows), len(rows[0]), figsize=(len(rows[0]) * 0.95, len(rows) * 1.0 + (0.4 if title else 0)), squeeze=False)
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            a = ax[i, j]
            a.imshow(im, cmap='gray', vmin=0, vmax=1)
            a.set_xticks([])
            a.set_yticks([])
            if j == 0: a.set_ylabel(labels[i], fontsize=7)
            if cols and i == 0: a.set_title(cols[j], fontsize=7)
    if title: fig.suptitle(title, fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(path + '.pdf', bbox_inches='tight')
    fig.savefig(path + '.png', dpi=110, bbox_inches='tight')
    plt.close(fig)

def panel(f, i, T, path, title):
    x, y = X1[i:i + 1], Y1[i:i + 1]
    xf, xt = x.float() / 255, tf[T](x)
    p, pT = pa2.predict(f, x), pa2.predict(f, xt)
    rows = [torch.cat([xf, p], 1), torch.cat([xt.float() / 255, tf[T](y).float() / 255], 1), torch.cat([xt.float() / 255, pT], 1), torch.cat([xt.float() / 255, tf[T](p)], 1)]
    rows = [r[0, :, 0].numpy() for r in rows]
    labels = [f'(1) X | f(X)', f'(2) {T}(X) | {T}(Y)', f'(3) {T}(X) | f({T}(X))', f'(4) {T}(X) | {T}(f(X))']
    grid(rows, labels, os.path.join(fd, path), [f'in {k}' for k in range(1, 6)] + [f'h={h}' for h in range(1, 6)], title)
    frames = []
    for h in range(5):
        fig, ax = plt.subplots(4, 2, figsize=(2.8, 5.4))
        for r in range(4):
            for c, im in enumerate([rows[r][4], rows[r][5 + h]]):
                ax[r, c].imshow(im, cmap='gray', vmin=0, vmax=1)
                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])
            ax[r, 0].set_ylabel(labels[r], fontsize=6)
        ax[0, 0].set_title('last input', fontsize=7)
        ax[0, 1].set_title(f'horizon h={h + 1}', fontsize=7)
        fig.suptitle(title, fontsize=6)
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    frames[0].save(os.path.join(gd, path + '.gif'), save_all=True, append_images=frames[1:], duration=900, loop=0)

def speedFig(f, i, path, title):
    rows, labels = [], []
    for k in (1, 2, 3):
        x, y, _ = pa2.window(te[i:i + 1], 1, k)
        rows.append(torch.cat([x.float() / 255, pa2.predict(f, x)[:, :2], y[:, :2].float() / 255], 1)[0, :, 0].numpy())
        labels.append(f'{k}x')
    grid(rows, labels, os.path.join(fd, path), [f'in {k}' for k in range(1, 6)] + ['pred h=1', 'pred h=2', 'true h=1', 'true h=2'], title)

if stage == 'base':
    mA, _ = pa2.loadModel('A', 0)
    nComp = np.array([[cv2.connectedComponents(cv2.dilate((fr > 51).astype(np.uint8), np.ones((5, 5), np.uint8)))[0] - 1 for fr in vid[:, 0].numpy()] for vid in te])
    clean = np.where((nComp[:, :10] >= 2).all(1))[0]
    cross = np.where((nComp[:, :5] >= 2).all(1) & ((nComp[:, 5:10] < 2).sum(1) >= 3))[0]
    fig2 = [int(clean[0]), int(clean[1]), int(cross[0])]
    rows, labels = [], []
    p = pa2.predict(mA, X1[fig2])
    for j, i in enumerate(fig2):
        rows += [X1[i, :, 0].numpy() / 255, p[j, :, 0].numpy(), Y1[i, :, 0].numpy() / 255]
        labels += [f'#{9000 + i} input 1-5', 'prediction 6-10', 'truth 6-10']
    grid(rows, labels, os.path.join(fd, 'fig2'))
    pv = np.load(os.path.join(rd, 'pervideo', 'A_s0.npz'))
    fig3 = [(int(i), 'M') for i in np.argsort(-pv['cM'])[:2]] + [(int(i), 'R') for i in np.argsort(-pv['cR'])[:2]]
    for n, (i, T) in enumerate(fig3):
        panel(mA, i, T, f'fig3_{n + 1}', f'model A (seed 0), test video #{9000 + i}, T={T}, C_{T}={pv["c" + T][i]:#.3g}')
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, T in zip(ax, 'MR'):
        c = pv['c' + T]
        a.hist(c, bins=np.logspace(np.log10(c.min()), np.log10(c.max()), 40), color='tab:blue', edgecolor='k')
        a.set_xscale('log')
        for i, t in fig3:
            if t == T: a.axvline(c[i], color='r', ls='--', label=f'#{9000 + i} (Fig. 3)')
        a.axvline(c.mean(), color='k', ls=':', label=f'mean = {c.mean():#.3g}')
        a.set(xlabel=f'per-video $C_{T}$ (log scale)', ylabel='number of test videos', title=f'({"ab"[T == "R"]}) $C_{T}$, model A seed 0')
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fd, 'fig4.pdf'))
    fig.savefig(os.path.join(fd, 'fig4.png'), dpi=130)
    shown = [f - 1 for f in sorted({1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 16, 19})]
    fig5 = int(np.where((nComp[:, shown] >= 2).all(1))[0][0])
    speedFig(mA, fig5, 'fig5', f'model A (seed 0), test video #{9000 + fig5}')
    json.dump(dict(fig2=fig2, fig3=fig3, fig5=fig5), open(os.path.join(rd, 'selection.json'), 'w'))
    print(dict(fig2=fig2, fig3=fig3, fig5=fig5))
else:
    sel = json.load(open(os.path.join(rd, 'selection.json')))
    base = pa2.loadModel(stage[3:], 0)[0]
    fFin = pa2.Sym(base)
    pv = np.load(os.path.join(rd, 'pervideo', f'{stage}_s0.npz'))
    for n, (i, T) in enumerate(sel['fig3']):
        panel(fFin, i, T, f'fig6a_{n + 1}', f'final model {stage} (seed 0), test video #{9000 + i}, T={T}, C_{T}={pv["c" + T][i]:#.3g}')
    speedFig(fFin, sel['fig5'], 'fig6b', f'final model {stage} (seed 0), test video #{9000 + sel["fig5"]}')
