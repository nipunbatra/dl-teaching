from pathlib import Path
import json,hashlib
import numpy as np
import torch
from PIL import Image,ImageDraw
from train_mnist_ddpm import TinyDenoiser,make_schedule,load_mnist
R=Path(__file__).resolve().parents[1];A=R/'assets';torch.set_num_threads(4)
ck=R/'models/model.pt';saved=torch.load(ck,map_location='cpu',weights_only=True);m=TinyDenoiser();m.load_state_dict(saved['model']);m.eval();s=make_schedule();digits=load_mnist(R/'work/data',False)[:10]
tuples={}
with torch.no_grad():
 for seed in [667,668]:
  torch.manual_seed(seed);eps=torch.randn_like(digits)
  for t in [7,350,920]:
   xt=s[2][t].sqrt()*digits+(1-s[2][t]).sqrt()*eps;pred=m(xt,torch.full((10,),t,dtype=torch.long))
   tuples[f'{seed}-{t}']=dict(eps=eps[:,0].numpy().round(6).tolist(),xt=xt[:,0].numpy().round(6).tolist(),pred=pred[:,0].numpy().round(6).tolist())
Image.fromarray(np.uint8(np.clip((pred[0,0].numpy()+1)*127.5,0,255))).resize((280,280),Image.Resampling.NEAREST).save(A/'new-model-prediction.png')
(A/'tuple-data.json').write_text(json.dumps(tuples,separators=(',',':')))
z=np.load(R/'models/trajectory.npz')
def tile(v,sz):return Image.fromarray(np.uint8(np.clip((v.squeeze()+1)*127.5,0,255))).resize((sz,sz),Image.Resampling.NEAREST).convert('RGB')
can=Image.new('RGB',(8*120,150),'#f7f8fa');d=ImageDraw.Draw(can)
for j,a in enumerate(z['samples']):can.paste(tile(a,112),(j*120,0));d.text((j*120+35,120),str(j+1),fill='black')
can.save(A/'new-model-gallery.png')
ts=[1000,800,600,400,200,50,1];can=Image.new('RGB',(7*132+100,320),'#f7f8fa');d=ImageDraw.Draw(can)
d.text((5,80),'State xt',fill='black');d.text((5,225),'Estimate x0',fill='black')
for j,t in enumerate(ts):
 d.text((105+j*132,8),f't = {t}',fill='black');can.paste(tile(z[f'state_{t}'][0],126),(100+j*132,28));can.paste(tile(z[f'estimate_{t}'][0],126),(100+j*132,179))
can.save(A/'new-model-trace.png')
# Four source samples with their original pixel content; preserve watermark strips.
im=Image.open(A/'ldm.png');ys=[(103,226),(228,347),(350,474),(477,602)]
can=Image.new('RGB',(1000,280),'#f7f8fa')
for i,(y0,y1) in enumerate(ys):
 cr=im.crop((0,y0,125,y1));cr.resize((240,240)).save(A/f'ldm-{i+1}.png');can.paste(cr.resize((240,240)),(i*250,0))
ImageDraw.Draw(can).text((10,252),'LDM Figure 17: A basket full of cherries. Four published samples; seeds not supplied.',fill='black');can.save(A/'ldm-samples.png')
# Exact mixture score used in a finite-step Gaussian reverse approximation.
ab=s[2].numpy().astype(float);b=s[0].numpy().astype(float);al=s[1].numpy().astype(float);var=s[3].numpy().astype(float);mu=np.stack([2*np.cos(np.arange(8)*np.pi/4),2*np.sin(np.arange(8)*np.pi/4)],1)
oracles={}
for seed in [667,668]:
 rng=np.random.default_rng(seed);clean=mu[rng.integers(0,8,256)]+rng.normal(0,.18,(256,2));eps=rng.normal(size=(256,2));x=rng.normal(size=(256,2));frames={'1000':x.round(5).tolist()}
 for t in range(1000,0,-1):
  means=np.sqrt(ab[t])*mu;v=ab[t]*.18**2+1-ab[t];delta=means[None]-x[:,None];log=-np.sum(delta**2,2)/(2*v);r=np.exp(log-log.max(1,keepdims=True));r/=r.sum(1,keepdims=True);score=(r[:,:,None]*delta).sum(1)/v;pred=-np.sqrt(1-ab[t])*score
  x=(x-b[t]*pred/np.sqrt(1-ab[t]))/np.sqrt(al[t]);
  if t>1:x+=np.sqrt(var[t])*rng.normal(size=x.shape)
  if (t-1)%50==0:frames[str(t-1)]=x.round(5).tolist()
 oracles[str(seed)]=dict(clean=clean.tolist(),noise=eps.tolist(),reverse=frames)
(A/'oracle-data.json').write_text(json.dumps(oracles,separators=(',',':')))
meta=json.loads((R/'models/train-log.json').read_text());meta.update(checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),sample_seed=667,sample_count=8,sampler='1000 ancestral DDPM steps; posterior variance; no estimate clipping',selected_digit='MNIST test 0 (7)',tuple_seeds=[667,668],tuple_timesteps=[7,350,920]);(A/'new-model-provenance.json').write_text(json.dumps(meta,indent=2))
print('Evidence exported',meta['parameters'],'parameters',meta['steps'],'steps')
