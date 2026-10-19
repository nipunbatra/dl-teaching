from pathlib import Path
import json,sys,io,contextlib,traceback
import nbformat as nbf
R=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook();nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
md=lambda s:nb.cells.append(nbf.v4.new_markdown_cell(s));code=lambda s:nb.cells.append(nbf.v4.new_code_cell(s))
md('# Generative Modelling II: executable DDPM companion\n\nCPU exercises use the same one-based schedule and fixed posterior variance as the lecture. Optional full training is separate. Normalisation is uint8/127.5 - 1. All small-number fixtures are analytical; recorded image predictions come from the documented model. Run from this notebook directory or the lecture root.')
code("from pathlib import Path\nimport sys, torch, numpy as np\nROOT = Path.cwd()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent\nif not (ROOT/'scripts/train_mnist_ddpm.py').exists():\n    ROOT = ROOT/'generative-modelling-ii'\nsys.path.insert(0, str(ROOT/'scripts'))\nfrom train_mnist_ddpm import make_schedule, extract, TinyDenoiser, training_loss, reverse_step, sample_ddpm\ntorch.set_num_threads(4)\ntorch.manual_seed(667)\nprint('Torch', torch.__version__)")
md('## 1. A schedule with a real clean sentinel\nThe endpoint must have little signal remaining before starting a sampler from a standard Gaussian is sensible.')
code("beta, alpha, abar, posterior_var = make_schedule()\nassert abar[0] == 1 and posterior_var[1] == 0\nassert torch.all(abar[1:] < abar[:-1])\nprint('arrays:', len(beta), 'terminal retention:', abar[-1].item())")
md('## 2. Manufacture a training tuple\nA batch may contain different timesteps. This numerical exercise uses a small authored image tensor, not an MNIST sample.')
code("x0 = torch.linspace(-1, 1, 3*28*28).reshape(3,1,28,28)\nt = torch.tensor([7,350,920], dtype=torch.long)\neps = torch.randn_like(x0)\na = extract(abar,t,x0)\nxt = a.sqrt()*x0 + (1-a).sqrt()*eps\nprint('coefficients', tuple(a.shape), 'input', tuple(xt.shape), 'target', tuple(eps.shape))")
md('## 3. Inspect the named-layer denoiser and gradients\nOne CPU forward/backward batch checks execution; it does not train a useful model.')
code("model = TinyDenoiser()\nprediction = model(xt,t)\nloss = torch.nn.functional.mse_loss(prediction, eps)\nloss.backward()\nassert prediction.shape == x0.shape\ngrad = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)\nassert grad > 0 and torch.isfinite(loss)\nprint('parameters', sum(p.numel() for p in model.parameters()), 'loss', loss.item(), 'gradient L1', grad.item())")
md('## 4. The same scalar in forward, loss and reverse calculations')
code("import math\nx2 = .8*1 + .6*(-.5)\neps_hat = -.4\nxhat = (x2-.6*eps_hat)/.8\nmean = (x2-(.2/.6)*eps_hat)/math.sqrt(.8)\nvariance = .2*(1-.8)/(1-.64)\nsample = mean+math.sqrt(variance)*.3\nprint(dict(x2=x2, loss=(-.5-eps_hat)**2, xhat=xhat, mean=mean, variance=variance, sample=sample))\nassert abs(sample-.8080881928749334)<1e-12")
md('## 5. The final DDPM update ignores fresh sampling noise')
code("schedule=make_schedule()\nx=torch.randn(2,1,28,28); prediction=torch.randn_like(x)\na=reverse_step(x,prediction,1,schedule,torch.zeros_like(x))\nb=reverse_step(x,prediction,1,schedule,torch.full_like(x,999))\nassert torch.equal(a,b)\nprint('No extra noise at t=1 verified')")
md('## 6. Cross-attention with hand-designed features\nThe row axis enumerates image queries. Each row normalises over text keys.')
code("Q=np.eye(2); K=np.array([[1,0],[0,1],[1,1]]); V=np.array([[1,0],[0,2],[1,1]])\nscores=Q@K.T/np.sqrt(2)\nA=np.exp(scores-scores.max(axis=1,keepdims=True)); A/=A.sum(axis=1,keepdims=True)\nassert np.allclose(A.sum(axis=1),1)\nprint('attention:',A,'output:',A@V,sep='\\n')")
md('## 7. The exact known-mixture score\nThis is an analytical calculation, not a learned neural score.')
code("centres=np.stack([2*np.cos(np.arange(8)*np.pi/4),2*np.sin(np.arange(8)*np.pi/4)],1)\nx=np.array([[1.,.5],[-1.,0.]])\nat=float(abar[350]); means=np.sqrt(at)*centres; v=at*.18**2+1-at\ndelta=means[None]-x[:,None]\nlogits=-np.sum(delta**2,axis=-1)/(2*v)\nr=np.exp(logits-logits.max(axis=1,keepdims=True)); r/=r.sum(axis=1,keepdims=True)\nscore=(r[:,:,None]*delta).sum(axis=1)/v\nprint('score',score,'oracle noise',-np.sqrt(1-at)*score,sep='\\n')")
md('## 8. Guidance and masked-token loss are different operations')
code("eu=np.array([.2,-.3]); ec=np.array([.6,.1])\nfor s in [0,1,3]: print('guidance',s,eu+s*(ec-eu))\nmask=np.array([False,True,False,False,True]); p_true=np.array([.9,.7,.8,.9,.6]); t=.5\nce_sum=-np.log(p_true[mask]).sum()\nprint('illustrative CE sum',ce_sum,'time-weighted sum',ce_sum/t)\n# No division by sequence length in this displayed time-weighted convention.")
md('## 9. Inspect the actual saved experiment\nThis is a record of training already performed, not a claim that this notebook reruns it.')
code("import json,hashlib\nmeta=json.loads((ROOT/'assets/new-model-provenance.json').read_text())\nprint({k:meta[k] for k in ['steps','batch_size','parameters','seed','alpha_bar_T','sampler']})\ncheckpoint=ROOT/'models/model.pt'\nassert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==meta['checkpoint_sha256']\nprint('Checkpoint hash verified')")
md('## Optional model training and sampling\nRun these commands in the lecture folder. The default device is CPU; use `--device mps` or `--device cuda` only when available. No expensive training occurs automatically in this notebook.\n\n```sh\npython scripts/train_mnist_ddpm.py --mode smoke\npython scripts/train_mnist_ddpm.py --mode train --steps 2000 --device mps\npython scripts/train_mnist_ddpm.py --mode sample --checkpoint models/model.pt --device mps\npython scripts/export_evidence.py\n```\n\nThe final trajectory arrays distinguish chain states from clean estimates. Display clips are not training or sampler clamps. The sampler has no arbitrary timestep skipping. Advanced image and language systems are sourced explanations, not models trained here.')
nbf.write(nb,R/'notebooks/ddpm-companion.ipynb')
# Execute every default code cell in an isolated namespace and preserve actual output.
ns={'__name__':'__main__'};import os;os.chdir(R);n=0
for cell in nb.cells:
 if cell.cell_type!='code':continue
 n+=1;capture=io.StringIO()
 with contextlib.redirect_stdout(capture):exec(compile(cell.source,f'notebook-cell-{n}','exec'),ns)
 cell.execution_count=n;cell.outputs=[nbf.v4.new_output('stream',name='stdout',text=capture.getvalue())]
nbf.write(nb,R/'notebooks/ddpm-companion.executed.ipynb');(R/'qa/notebook.json').write_text(json.dumps({'executed_code_cells':n,'errors':0,'training_in_notebook':'one backward batch only; full run logged separately'},indent=2));print('Executed',n,'notebook cells')
