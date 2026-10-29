import torch, torch.nn as nn, torch.nn.functional as F, os

root = os.path.dirname(os.path.abspath(__file__))
dev = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
starts = {1: 11, 2: 8, 3: 2}
startsAug = {1: 14, 2: 8, 3: 2}
archs = {'A': dict(hid=64, k=3), 'D': dict(hid=44, k=5)}
runs = {'A': ('A', False, False), 'B': ('A', False, True), 'C': ('A', True, False), 'D': ('D', True, False)}

def loadVideos():
    import torchvision
    return torchvision.datasets.MovingMNIST(root=os.path.join(root, 'data'), split=None, download=True).data

def mirror(x):
    return x.flip(-1)

def rot(x, a=1):
    return torch.rot90(x, a, (-2, -1))

def gApply(x, g):
    y = rot(x, g % 4)
    return mirror(y) if g >= 4 else y

def window(v, t, k):
    idx = (torch.as_tensor(t).reshape(-1, 1) - 1 + torch.as_tensor(k).reshape(-1, 1) * torch.arange(10)).expand(len(v), 10).to(v.device)
    clip = v[torch.arange(len(v), device=v.device)[:, None], idx.clamp(max=19)]
    return clip[:, :5], clip[:, 5:], idx[:, 5:] < 20

class Cell(nn.Module):
    def __init__(self, cin, hid, k):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv2d(cin + hid, 4 * hid, k, padding=k // 2)

    def forward(self, u, h, c):
        i, f, o, g = self.conv(torch.cat([u, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c

class Net(nn.Module):
    def __init__(self, hid=64, k=3):
        super().__init__()
        self.l1 = Cell(16, hid, k)
        self.l2 = Cell(hid, hid, k)
        self.out = nn.Conv2d(hid, 16, 1)

    def forward(self, x):
        h1 = c1 = h2 = c2 = x.new_zeros(x.shape[0], self.l1.hid, 16, 16)
        ys = []
        for tau in range(9):
            u = F.pixel_unshuffle(x[:, tau] if tau < 5 else y, 4)
            h1, c1 = self.l1(u, h1, c1)
            h2, c2 = self.l2(h1, h2, c2)
            y = torch.sigmoid(F.pixel_shuffle(self.out(h2), 4))
            if tau >= 4: ys.append(y)
        return torch.stack(ys, 1)

class Sym(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return torch.stack([gApply(self.f(gApply(x, g)), g) if g >= 4 else rot(self.f(gApply(x, g)), -g) for g in range(8)]).mean(0)

class Blank(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)

class CopyLast(nn.Module):
    def forward(self, x):
        return x[:, -1:].expand(-1, 5, -1, -1, -1)

def nParams(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def loadModel(name, seed):
    ck = torch.load(os.path.join(root, 'ckpt', f'{name}_s{seed}.pt'), map_location=dev)
    m = Net(**archs[ck['arch']]).to(dev)
    m.load_state_dict(ck['state'])
    return m.eval(), ck

@torch.no_grad()
def predict(m, x, bs=32):
    return torch.cat([m(x[i:i + bs].to(dev).float() / 255).cpu() for i in range(0, len(x), bs)])

def perVideoMse(a, b):
    return ((a - b) ** 2).mean((2, 3, 4))
