from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];A=R/'assets'
im=Image.open(A/'sdedit.jpg')
for name,box in [('sdedit-input',(70,0,174,104)),('sdedit-output-1',(70,110,174,213)),('sdedit-output-2',(70,215,174,317))]:im.crop(box).resize((312,312)).save(A/(name+'.png'))
rec=json.loads((A/'tuple-data.json').read_text())['667-350']
for key,name in [('xt','new-model-input'),('pred','new-model-prediction')]:
 ar=np.array(rec[key][0]);Image.fromarray(np.uint8(np.clip((ar+1)*127.5,0,255))).resize((280,280),Image.Resampling.NEAREST).save(A/(name+'.png'))
z=np.load(R/'models/trajectory.npz');im=z['state_1000'][0,0];Image.fromarray(np.uint8(np.clip((im+1)*127.5,0,255))).resize((280,280),Image.Resampling.NEAREST).save(A/'fresh-gaussian.png')
# Two distinct real digits plus distributions of one coordinate (illustrative overlap).
d=json.loads((A/'demo-data.json').read_text());can=Image.new('RGB',(960,360),'#f7f8fa');dr=ImageDraw.Draw(can)
for i,ix in enumerate([0,2]):
 ar=np.array(d['digits'][ix]);pic=Image.fromarray(np.uint8((ar+1)*127.5)).resize((180,180),Image.Resampling.NEAREST);can.paste(pic,(i*200,80));dr.text((i*200+10,275),'MNIST test '+str(ix),fill='black')
for mu,col in [(650,'#3478e5'),(750,'#b98224')]:
 pts=[(xx,280-150*np.exp(-.5*((xx-mu)/65)**2)) for xx in range(440,940)];dr.line(pts,fill=col,width=4)
dr.line([(440,280),(940,280)],fill='#566171',width=2);dr.text((475,305),'Overlapping noisy distributions (one-coordinate schematic)',fill='#566171');can.save(A/'ambiguity.png')
# Standalone original schedule panel with actual data and fixed axes.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ab=np.array(d['alpha_bar']);ts=np.arange(1001)
fig,axs=plt.subplots(1,2,figsize=(11,3.4));fig.patch.set_facecolor('#f7f8fa')
for ax in axs:ax.set_facecolor('#f7f8fa');ax.set_xlabel('timestep t');ax.grid(alpha=.2)
axs[0].plot(ts,ab,color='#3478e5',label='cumulative retention');axs[0].set_ylim(0,1);axs[0].set_ylabel('alpha-bar')
axs[1].semilogy(ts[1:],ab[1:]/(1-ab[1:]),color='#b98224');axs[1].set_ylabel('SNR (log scale)');axs[1].set_title('t=0: infinity, omitted')
fig.tight_layout();fig.savefig(A/'schedule.png',dpi=160);plt.close(fig)
print('Refined application crops, model provenance, ambiguity and schedule plots')
