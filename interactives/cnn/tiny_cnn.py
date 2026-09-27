"""Exact-initialization PyTorch companion to the browser CNN.

Run: python3 interactives/cnn/tiny_cnn.py --check
Requires PyTorch. No dataset downloads. CPU, float64, full-batch SGD.
"""
import argparse
import json
import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--steps', type=int, default=600)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    data = json.loads((HERE / 'initial-model.json').read_text())
    m = data['model']
    x = torch.tensor([s['x'] for s in data['samples']], dtype=torch.float64).unsqueeze(1)
    y = torch.tensor([s['y'] for s in data['samples']], dtype=torch.long)
    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(2, 2))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(m['k']).unsqueeze(1))
        model[0].bias.copy_(torch.tensor(m['b']))
        model[4].weight.copy_(torch.tensor(m['w']))
        model[4].bias.copy_(torch.tensor(m['a']))
    if args.check:
        js = """const M=require('./model.js');const m=M.init();
        console.log(JSON.stringify({objective:M.objective(m),
          logits:M.samples.map(s=>M.forward(m,s.x).logits)}));"""
        ref = json.loads(subprocess.check_output(['node', '-e', js], cwd=HERE))
        logits = model(x)
        torch.testing.assert_close(logits, torch.tensor(ref['logits']), atol=1e-12, rtol=1e-12)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        assert abs(loss.item() - ref['objective']['loss']) < 1e-12
        grad = ref['objective']['grad']
        for actual, expected in [(model[0].weight.grad[:, 0], grad['k']),
                                 (model[0].bias.grad, grad['b']),
                                 (model[4].weight.grad, grad['w']),
                                 (model[4].bias.grad, grad['a'])]:
            torch.testing.assert_close(actual, torch.tensor(expected), atol=1e-12, rtol=1e-12)
        print('PASS: browser / PyTorch initial logits, mean loss, all 26 gradients')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.4)
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(x)
        print(f'Steps: {args.steps}; training CE: {F.cross_entropy(logits,y):.6f}; '
              f'training correct: {(logits.argmax(1)==y).sum().item()}/6')
    print('These are training examples, not a generalization benchmark.')


if __name__ == '__main__':
    main()
