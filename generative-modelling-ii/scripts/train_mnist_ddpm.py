"""One-based DDPM: linear T=1000, epsilon prediction, posterior variance.
python scripts/train_mnist_ddpm.py --mode smoke
python scripts/train_mnist_ddpm.py --mode train --steps 2000 --device mps
python scripts/train_mnist_ddpm.py --mode sample --checkpoint models/model.pt
"""
from pathlib import Path
import argparse, gzip, hashlib, json, math, time, urllib.request
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]

def make_schedule(T=1000,device='cpu',dtype=torch.float32):
    if T!=1000: raise ValueError('This linear teaching schedule requires T=1000; define a new endpoint-checked schedule before changing T.')
    beta=torch.zeros(T+1,device=device,dtype=dtype)
    beta[1:]=torch.linspace(1e-4,2e-2,T,device=device,dtype=dtype)
    alpha=1-beta; ab=alpha.cumprod(0); var=torch.zeros_like(beta)
    var[1:]=beta[1:]*(1-ab[:-1])/(1-ab[1:])
    return beta,alpha,ab,var

def extract(values,t,x):
    assert t.dtype==torch.long and t.shape==(x.shape[0],)
    return values[t].reshape(x.shape[0],*([1]*(x.ndim-1)))

class TimeBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__();self.conv1=nn.Conv2d(cin,cout,3,padding=1);self.norm1=nn.GroupNorm(8,cout)
        self.time=nn.Linear(64,cout);self.conv2=nn.Conv2d(cout,cout,3,padding=1);self.norm2=nn.GroupNorm(8,cout)
        self.skip=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x,h):
        y=F.silu(self.norm1(self.conv1(x)))+self.time(h)[:,:,None,None]
        return F.silu(self.norm2(self.conv2(y))+self.skip(x))

class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__();self.time_mlp=nn.Sequential(nn.Linear(64,64),nn.SiLU(),nn.Linear(64,64))
        self.encoder1=TimeBlock(1,32);self.down1=nn.Conv2d(32,64,4,2,1);self.encoder2=TimeBlock(64,64)
        self.down2=nn.Conv2d(64,128,4,2,1);self.middle=TimeBlock(128,128)
        self.up1=nn.Conv2d(128,64,3,padding=1);self.decoder1=TimeBlock(128,64)
        self.up2=nn.Conv2d(64,32,3,padding=1);self.decoder2=TimeBlock(64,32);self.head=nn.Conv2d(32,1,1)
    def forward(self,x,t):
        freq=torch.exp(-math.log(10000)*torch.arange(32,device=x.device)/31)
        angle=t.float()[:,None]*freq;h=self.time_mlp(torch.cat([angle.sin(),angle.cos()],1))
        e1=self.encoder1(x,h);e2=self.encoder2(self.down1(e1),h);m=self.middle(self.down2(e2),h)
        u1=self.up1(F.interpolate(m,size=e2.shape[-2:],mode='nearest'))
        d1=self.decoder1(torch.cat([u1,e2],1),h)
        u2=self.up2(F.interpolate(d1,size=e1.shape[-2:],mode='nearest'))
        return self.head(self.decoder2(torch.cat([u2,e1],1),h))

def training_loss(model,x0,ab,T=1000):
    t=torch.randint(1,T+1,(len(x0),),device=x0.device);eps=torch.randn_like(x0);a=extract(ab,t,x0)
    xt=a.sqrt()*x0+(1-a).sqrt()*eps;pred=model(xt,t)
    assert pred.shape==x0.shape
    return F.mse_loss(pred,eps)

def reverse_step(x,eps,step,schedule,noise):
    b,a,ab,v=schedule
    mean=(x-b[step]*eps/(1-ab[step]).sqrt())/a[step].sqrt()
    return mean+v[step].sqrt()*noise if step>1 else mean

@torch.no_grad()
def sample_ddpm(model,shape,schedule,T=1000,device='cpu',seed=667,record=False):
    g=torch.Generator(device='cpu').manual_seed(seed)
    rand=lambda:torch.randn(shape,generator=g).to(device)
    x=rand();was=model.training;model.eval();frames={};estimates={}
    try:
        for step in range(T,0,-1):
            t=torch.full((shape[0],),step,device=device,dtype=torch.long);eps=model(x,t)
            if record and step in [1000,800,600,400,200,50,1]:
                frames[str(step)]=x.cpu().numpy();ab=schedule[2][step]
                estimates[str(step)]=((x-(1-ab).sqrt()*eps)/ab.sqrt()).cpu().numpy()
            x=reverse_step(x,eps,step,schedule,rand() if step>1 else torch.zeros_like(x))
        if record:frames['0']=x.cpu().numpy()
    finally:model.train(was)
    return (x,frames,estimates) if record else x

FILES={'train-images-idx3-ubyte.gz':'f68b3c2dcbeaaa9fbdd348bbdeb94873','t10k-images-idx3-ubyte.gz':'9fb629c4189551a2d022fa330f9573f3'}
def load_mnist(cache,train=True):
    name='train-images-idx3-ubyte.gz' if train else 't10k-images-idx3-ubyte.gz';p=Path(cache)/name;p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():urllib.request.urlretrieve('https://ossci-datasets.s3.amazonaws.com/mnist/'+name,p)
    assert hashlib.md5(p.read_bytes()).hexdigest()==FILES[name]
    return torch.from_numpy(np.frombuffer(gzip.decompress(p.read_bytes()),np.uint8,offset=16).copy().reshape(-1,1,28,28)).float()/127.5-1

def run(a):
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);device=a.device
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);s=make_schedule(device=device);m=TinyDenoiser().to(device)
    if a.checkpoint:
        saved=torch.load(a.checkpoint,map_location=device,weights_only=True);assert saved['contract']=='linear1000-eps-posterior-v1';m.load_state_dict(saved['model'])
    if a.mode=='sample':
        x,frames,est=sample_ddpm(m,(8,1,28,28),s,device=device,seed=a.seed,record=True)
        np.savez_compressed(out/'trajectory.npz',samples=x.cpu().numpy(),**{'state_'+k:v for k,v in frames.items()},**{'estimate_'+k:v for k,v in est.items()});print('Saved trajectory',flush=True);return
    data=load_mnist(a.cache);opt=torch.optim.Adam(m.parameters(),lr=2e-4);start=time.time();history=[]
    steps=1 if a.mode=='smoke' else a.steps
    for step in range(1,steps+1):
        ix=torch.randint(len(data),(a.batch_size,));x=data[ix].to(device);loss=training_loss(m,x,s[2]);opt.zero_grad(set_to_none=True);loss.backward()
        grad=float(sum(p.grad.detach().square().sum().cpu() for p in m.parameters() if p.grad is not None).sqrt());assert math.isfinite(loss.item()) and grad>0
        opt.step()
        if step==1 or step%100==0 or step==steps:
            history.append({'step':step,'batch_noise_mse':loss.item(),'gradient_norm':grad});print(json.dumps(history[-1]),flush=True)
        if a.mode=='train' and (step%500==0 or step==steps):
            torch.save({'model':m.state_dict(),'contract':'linear1000-eps-posterior-v1','steps':step,'seed':a.seed},out/'model.pt')
    meta=dict(mode=a.mode,steps=steps,batch_size=a.batch_size,seed=a.seed,normalization='uint8/127.5 - 1',T=1000,alpha_bar_T=s[2][-1].item(),variance='posterior',parameters=sum(p.numel() for p in m.parameters()),device=device,torch=torch.__version__,elapsed_seconds=time.time()-start,history=history,claim='Execution check only' if a.mode=='smoke' else 'Small teaching model; no image-quality metric computed')
    if (out/'model.pt').exists():meta['checkpoint_sha256']=hashlib.sha256((out/'model.pt').read_bytes()).hexdigest()
    (out/(a.mode+'-log.json')).write_text(json.dumps(meta,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','train','sample'],default='smoke');p.add_argument('--steps',type=int,default=2000);p.add_argument('--seed',type=int,default=667);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--device',default='cpu');p.add_argument('--cache',default=str(ROOT/'work/data'));p.add_argument('--output',default=str(ROOT/'models'));p.add_argument('--checkpoint');run(p.parse_args())
