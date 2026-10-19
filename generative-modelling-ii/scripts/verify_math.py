from pathlib import Path
import json,math,sys
import numpy as np
import torch
from train_mnist_ddpm import TinyDenoiser,make_schedule,training_loss,extract,reverse_step,sample_ddpm,load_mnist
R=Path(__file__).resolve().parents[1];torch.set_num_threads(4);torch.manual_seed(667);checks={}
def check(k,v):
 checks[k]=bool(v)
 if not v:raise AssertionError(k)
s=make_schedule(dtype=torch.float64);b,a,ab,v=s
check('schedule_length',all(len(x)==1001 for x in s));check('sentinel',ab[0]==1 and b[0]==0);check('finite_schedule',all(torch.isfinite(x).all() for x in s));check('monotone_retention',(ab[1:]<ab[:-1]).all());check('endpoint',abs(ab[-1].item()-4.035829765375676e-5)<1e-12);check('posterior_endpoint',v[1]==0)
x2=math.sqrt(.64)+math.sqrt(.36)*(-.5);loss=(-.5+.4)**2;xhat=(x2-math.sqrt(.36)*(-.4))/math.sqrt(.64);mean=(x2-.2*(-.4)/math.sqrt(.36))/math.sqrt(.8);var=.2*(1-.8)/(1-.64);sample=mean+math.sqrt(var)*.3
fixtures=dict(x2=x2,loss=loss,xhat=xhat,mean=mean,variance=var,sample=sample)
for k,w in dict(x2=.5,loss=.01,xhat=.925,mean=.7080881929,variance=1/9,sample=.8080881929).items():check(k,abs(fixtures[k]-w)<1e-9)
Q=np.eye(2);K=np.array([[1,0],[0,1],[1,1]]);V=np.array([[1,0],[0,2],[1,1]]);score=Q@K.T/np.sqrt(2);att=np.exp(score-score.max(1,keepdims=True));att/=att.sum(1,keepdims=True);AV=att@V
check('attention_row_sums',np.allclose(att.sum(1),1));check('attention_values',np.allclose(AV,[[.802224,.796664],[.598888,1.203336]],atol=1e-6));check('2x2_MSE',np.isclose(np.mean((np.array([-.5,1,0,-1])-np.array([-.4,.8,.1,-.7]))**2),.0375))
x=torch.randn(3,1,28,28);t=torch.tensor([7,350,920]);check('broadcast',extract(ab,t,x).shape==(3,1,1,1));m=TinyDenoiser();pred=m(x,t);check('model_shape',pred.shape==x.shape);l=training_loss(m,x,make_schedule()[2]);l.backward();check('finite_loss_gradients',torch.isfinite(l) and sum(p.grad.abs().sum() for p in m.parameters() if p.grad is not None)>0)
y1=reverse_step(x,pred.detach(),1,make_schedule(),torch.zeros_like(x));y2=reverse_step(x,pred.detach(),1,make_schedule(),torch.ones_like(x)*1000);check('final_step_ignores_noise',torch.equal(y1,y2))
# Algebraic independence check: posterior-mean expression equals epsilon mean without clipping.
t=350;eps=torch.randn_like(x);xt=torch.randn_like(x);bh,ah,abh,vh=make_schedule();xh=(xt-(1-abh[t]).sqrt()*eps)/abh[t].sqrt();pm=abh[t-1].sqrt()*bh[t]/(1-abh[t])*xh+ah[t].sqrt()*(1-abh[t-1])/(1-abh[t])*xt;em=(xt-bh[t]*eps/(1-abh[t]).sqrt())/ah[t].sqrt();check('two_mean_parameterisations',torch.allclose(pm,em,atol=3e-6))
# Recorded stochastic oracle repeatability, implemented twice from fresh seed.
def oracle(seed):
 rng=np.random.default_rng(seed);x=rng.normal(size=(32,2));mu=np.stack([2*np.cos(np.arange(8)*np.pi/4),2*np.sin(np.arange(8)*np.pi/4)],1)
 for t in range(1000,970,-1):
  vv=float(ab[t])*.18**2+1-float(ab[t]);d=np.sqrt(float(ab[t]))*mu[None]-x[:,None];log=-np.sum(d*d,2)/(2*vv);r=np.exp(log-log.max(1,keepdims=True));r/=r.sum(1,keepdims=True);sc=(r[:,:,None]*d).sum(1)/vv;ep=-np.sqrt(1-float(ab[t]))*sc;x=(x-float(b[t])*ep/np.sqrt(1-float(ab[t])))/np.sqrt(float(a[t]))+np.sqrt(float(v[t]))*rng.normal(size=x.shape)
 return x
u=oracle(667);check('oracle_finite',np.isfinite(u).all());check('oracle_seed_repeatability',np.array_equal(u,oracle(667)));check('oracle_seed_changes',not np.array_equal(u,oracle(668)))
report=dict(checks=checks,fixtures=fixtures,attention=att.tolist(),attention_output=AV.tolist(),parameter_count=sum(p.numel() for p in m.parameters()),execution='CPU forward/backward and semantic arithmetic; recorded trained-model run separately documented')
(R/'qa/math.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
