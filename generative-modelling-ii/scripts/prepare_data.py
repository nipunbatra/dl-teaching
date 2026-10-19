from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw
from train_mnist_ddpm import load_mnist
R=Path(__file__).resolve().parents[1];A=R/'assets';rng=np.random.default_rng(667)
x=load_mnist(R/'work/data',False).numpy();digit=x[0,0];eps=rng.normal(size=(28,28))
beta=np.r_[0,np.linspace(.0001,.02,1000)];alpha=1-beta;ab=np.cumprod(alpha)
def png(a,name):Image.fromarray(np.uint8(np.clip((a+1)*127.5,0,255))).resize((280,280),Image.Resampling.NEAREST).save(A/name)
png(digit,'digit.png');png(eps,'noise.png')
frames={};y=digit.copy()
for t in range(1001):
 if t:y=np.sqrt(alpha[t])*y+np.sqrt(beta[t])*rng.normal(size=y.shape)
 if t in [0,100,300,500,700,1000]:png(y,f'forward-{t}.png');frames[str(t)]=y.tolist()
# Reference image variants are deterministic computations, never model output.
for k,t in enumerate([7,350,920]):png(np.sqrt(ab[t])*x[k,0]+np.sqrt(1-ab[t])*rng.normal(size=(28,28)),f'batch-{t}.png')
# Crop sources only after inspecting the author-published complete figures.
crops={'ldm-samples':('ldm.png',(0.003,.118,.13,.983)), 'repaint-row':('repaint.png',(0,.0,1,.254)), 'repaint-input':('repaint.png',(0,.0,.105,.254)), 'repaint-outputs':('repaint.png',(.435,0,1,.254)), 'sdedit-input':('sdedit.jpg',(0,0,1,.36)), 'sdedit-output':('sdedit.jpg',(0,.36,1,1)), 'control-pair':('control.png',(.505,0,.997,.335)), 'dreambooth-input':('dreambooth.jpg',(0,0,.265,1)), 'dreambooth-output':('dreambooth.jpg',(.27,0,1,1)), 'sr3-top':('sr3.png',(0,0,1,.515))}
manifest=json.loads((R/'asset-manifest.json').read_text())
for key,(src,box) in crops.items():
 im=Image.open(A/src);w,h=im.size;im.crop(tuple(round(v*(w if i%2==0 else h)) for i,v in enumerate(box))).save(A/(key+'.png'))
 parent=next(v for v in manifest if v['local_path']=='assets/'+src);entry=parent.copy();entry.update(id=key,local_path='assets/'+key+'.png',transformations=f'Crop normalised box {box}; original retained as {src}. Caption supplied adjacent to crop.');manifest.append(entry)
for name in ['mnist_samples.png','mnist_cfg_comparison.png','mnist_denoising_grid.png','mnist_reverse_strip.png','autoencoder_vs_vae.svg','gan_game.svg']:
 p=R.parent/'figures'/name
 if p.exists():shutil.copy2(p,A/name)
for name in ['mnist_diffusion_metrics.json','toy_diffusion_metrics.json']:
 source=R.parent/'results'/name
 if source.exists():shutil.copy2(source,R/'assets'/name)
 elif not (R/'assets'/name).exists():raise FileNotFoundError('Retained older experiment metadata is missing: '+name)
(R/'assets/demo-data.json').write_text(json.dumps(dict(digits=x[:10,0].tolist(),digit_index=0,split='MNIST test',seed=667,beta=beta.tolist(),alpha_bar=ab.tolist(),forward=frames),separators=(',',':')))
manifest.append(dict(id='mnist-forward',local_path='assets/forward-*.png',source_page='https://yann.lecun.com/exdb/mnist/',authors_or_owner='LeCun, Cortes, Burges; local computation',paper_or_dataset='MNIST test example 0 (digit 7)',figure_or_caption='Actual recursive forward trajectory; seed 667; linear T=1000',license_or_usage_note='Dataset provenance retained; authored computation',retrieved_date='2026-10-19',transformations='uint8/127.5-1; fixed display scale clipped only for rendering',evidence_type='deterministic simulation',used_in_scene_ids=['s12','s15','s20','s22']))
(R/'asset-manifest.json').write_text(json.dumps(manifest,indent=2))
print('Prepared',len(manifest),'asset records; endpoint',ab[-1])
