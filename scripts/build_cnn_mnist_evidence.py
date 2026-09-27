"""Reproduce the ML course's 28-pixel LeNet and export an inspectable lesson.

Run: uv run --with torch python scripts/build_cnn_mnist_evidence.py
Raw MNIST downloads and the checkpoint stay under tmp/; only compact evidence
and selected public MNIST images are exported to the lecture.
"""
from pathlib import Path
import gzip
import hashlib
import json
import struct
import time
import urllib.request

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "tmp/cnn-mnist"
OUT = ROOT / "interactives/cnn/evidence"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}

def download(name):
    p = CACHE / name
    if not p.exists():
        print("Downloading", name, flush=True)
        urllib.request.urlretrieve("https://ossci-datasets.s3.amazonaws.com/mnist/" + name, p)
    assert hashlib.md5(p.read_bytes()).hexdigest() == FILES[name], name
    b = gzip.decompress(p.read_bytes())
    magic, count = struct.unpack(">II", b[:8])
    offset = 16 if magic == 2051 else 8
    tensor = torch.tensor(list(b[offset:]), dtype=torch.uint8)
    return tensor.reshape(count, 1, 28, 28).float() / 255 if magic == 2051 else tensor.long()

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def trace(self, x):
        z1 = self.conv1(x); a1 = F.relu(z1); p1 = F.max_pool2d(a1, 2)
        z2 = self.conv2(p1); a2 = F.relu(z2); p2 = F.max_pool2d(a2, 2)
        flat = p2.flatten(1); h1 = F.relu(self.fc1(flat)); h2 = F.relu(self.fc2(h1))
        logits = self.fc3(h2)
        return dict(conv1=z1, relu1=a1, pool1=p1, conv2=z2, relu2=a2,
                    pool2=p2, flat=flat, hidden1=h1, hidden2=h2, logits=logits)

    def forward(self, x):
        return self.trace(x)["logits"]

def evaluate(model, x, y):
    model.eval()
    loss = 0; logits = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            z = model(x[start:start+256]); logits.append(z)
            loss += F.cross_entropy(z, y[start:start+256], reduction="sum").item()
    z = torch.cat(logits)
    return dict(loss=loss/len(x), correct=int((z.argmax(1)==y).sum()), total=len(x)), z

def weights(model):
    return {k: v.detach().flatten().tolist() for k, v in model.state_dict().items()}

def main():
    begin = time.time()
    x = download("train-images-idx3-ubyte.gz"); y = download("train-labels-idx1-ubyte.gz")
    test_x = download("t10k-images-idx3-ubyte.gz"); test_y = download("t10k-labels-idx1-ubyte.gz")
    train_x, train_y = x[:5000], y[:5000]
    val_x, val_y = x[50000:51000], y[50000:51000]
    model = LeNet()
    assert sum(p.numel() for p in model.parameters()) == 44426
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history = []; checkpoints = {}
    for epoch in range(11):
        if epoch:
            model.train()
            for bx, by in loader:
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(bx), by).backward(); optimizer.step()
        tr, _ = evaluate(model, train_x, train_y); va, _ = evaluate(model, val_x, val_y)
        history.append(dict(epoch=epoch, train=tr, validation=va))
        if epoch in [0, 1, 3, 10]: checkpoints[str(epoch)] = weights(model)
        print(json.dumps(history[-1]), flush=True)
    test, logits = evaluate(model, test_x, test_y)
    pred = logits.argmax(1)
    chosen = [int(torch.where(test_y == i)[0][0]) for i in range(10)]
    chosen += [int(i) for i in torch.where(pred != test_y)[0][:4]]
    samples = []
    with torch.no_grad():
        for i in chosen:
            t = model.trace(test_x[i:i+1])
            samples.append(dict(index=i, label=int(test_y[i]), prediction=int(pred[i]),
                                pixels=(test_x[i]*255).round().byte().flatten().tolist(),
                                logits=t["logits"][0].tolist(),
                                checks={k:dict(shape=list(v.shape), sum=float(v.sum()),
                                              first=v.flatten()[:8].tolist()) for k,v in t.items()}))
        # Same 1,000 test examples for the two deterministic PCA views. No labels
        # enter PCA; labels colour the points only. These are explanatory views.
        raw = test_x[:1000].flatten(1)
        hidden = model.trace(test_x[:1000])["hidden2"]
        def project(z):
            z=z-z.mean(0,keepdim=True)
            _,_,v=torch.linalg.svd(z,full_matrices=False)
            return (z @ v[:2].T).tolist()
        pca=dict(raw=project(raw), learned=project(hidden), labels=test_y[:1000].tolist())
    confusion=torch.zeros(10,10,dtype=torch.long)
    for truth,guess in zip(test_y.tolist(),pred.tolist()):confusion[truth,guess]+=1
    evidence=dict(format=1, dataset="MNIST", seed=0, torch_version=torch.__version__,
                  parameters=44426, epochs=10, batch_size=64, optimizer="Adam", lr=.001,
                  splits={"train":"official train [0:5000]", "validation":"official train [50000:51000]",
                          "test":"official test [0:10000]"},
                  provenance="Reproduction of ml-teaching/notebooks/cnn.ipynb; separate seeded run, not its historical output.",
                  checkpoint_selection="Fixed 10 epochs; test evaluated only after training.",
                  history=history,test=test,confusion=confusion.tolist(),samples=samples,pca=pca,
                  checkpoints=checkpoints,download_md5=FILES)
    (OUT/"mnist.json").write_text(json.dumps(evidence,separators=(",",":")))
    (OUT/"mnist.js").write_text("window.CNN_MNIST="+json.dumps(evidence,separators=(",",":"))+";\n")
    torch.save(model.state_dict(),CACHE/"lenet.pt")
    print("TEST",json.dumps(test),"SECONDS",round(time.time()-begin,1),flush=True)

if __name__ == "__main__": main()
