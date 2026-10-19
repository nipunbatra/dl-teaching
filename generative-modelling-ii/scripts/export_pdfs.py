"""Raster stage exports with explicit live external and internal PDF links."""
from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from pypdf import PdfReader
R=Path(__file__).resolve().parents[1];OUT=R/'exports';OUT.mkdir(exist_ok=True);scenes=json.loads((R/'src/manifest.json').read_text())
for kind in ['reading','all-builds']:
 target=OUT/f'generative-modelling-ii-{kind}.pdf';c=canvas.Canvas(str(target),pagesize=(1280,720),pageCompression=1);c.setTitle('Generative Modelling II: '+kind);c.setAuthor('Nipun Batra · ES 667 · IIT Gandhinagar');count=0
 for s in scenes:
  builds=[s['builds']] if kind=='reading' else range(1,s['builds']+1)
  for b in builds:
   image=R/f'qa/builds/{s["id"]}-{b}.png';assert image.exists(),image
   if b==(s['builds'] if kind=='reading' else 1):c.bookmarkPage(s['id']);c.addOutlineEntry(s['id']+' '+s['title'],s['id'],level=0)
   c.drawImage(ImageReader(image),0,0,width=1280,height=720)
   c.linkURL(s['source'],(45,17,230,50),relative=0,thickness=0)
   c.linkRect('Course map','s10',(560,17,740,50),relative=0,thickness=0)
   c.setFillColor(colors.HexColor('#F7F8FA'));c.rect(995,16,240,34,fill=1,stroke=0);c.setFillColor(colors.HexColor('#566171'));c.setFont('Helvetica',11);c.drawRightString(1230,28,f'{s["id"]} · build {b}/{s["builds"]} · ES 667');c.showPage();count+=1
 c.save();print(kind,count)
# Two-page recap is selectable text, separate from raster classroom-stage exports.
font=Path('/Library/Fonts/Arial Unicode.ttf')
if not font.exists():font=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
if font.exists():pdfmetrics.registerFont(TTFont('Recap',str(font)));family='Recap'
else:family='Helvetica'
body=ParagraphStyle('body',fontName=family,fontSize=11,leading=16,textColor=colors.HexColor('#202633'),spaceAfter=9)
head=ParagraphStyle('head',parent=body,fontSize=22,leading=27,spaceAfter=18,textColor=colors.HexColor('#178F82'))
sub=ParagraphStyle('sub',parent=body,fontSize=14,leading=19,spaceBefore=8,spaceAfter=7,textColor=colors.HexColor('#3478E5'))
story=[]
def para(t,sty=body):story.append(Paragraph(t,sty))
para('Diffusion: from training pairs to generation',head)
para('ES 667 · Generative Modelling II · Two-page recap')
para('1. Define the corruption process',sub)
para('x<sub>0</sub> is clean data; x<sub>t</sub> is a noisy state. Choose variance β<sub>t</sub>, set α<sub>t</sub>=1−β<sub>t</sub>, and cumulative retention ᾱ<sub>t</sub>=∏<sub>s=1…t</sub>α<sub>s</sub>, with ᾱ<sub>0</sub>=1.')
para('One transition: x<sub>t</sub> = √α<sub>t</sub> x<sub>t−1</sub> + √β<sub>t</sub> η<sub>t</sub>, with independent η<sub>t</sub> ~ N(0,I).')
para('Direct marginal: x<sub>t</sub> = √ᾱ<sub>t</sub> x<sub>0</sub> + √(1−ᾱ<sub>t</sub>) ε. Sharing ε across times gives correct marginals, but a different joint coupling from a recursive trajectory.')
para('2. Manufacture a supervised target',sub)
para('Sample x<sub>0</sub> from the dataset, integer t uniformly from 1…T, and independent ε ~ N(0,I). Construct x<sub>t</sub>. Feed only (x<sub>t</sub>,t) to the unconditional denoiser; compare its prediction with ε. Average squared errors over examples and coordinates. Backpropagate through the denoiser and update θ.')
para('Population MSE estimates E[ε | x<sub>t</sub>,t], not each hidden random draw. The network does not receive the clean image or target noise as extra inputs.')
para('3. Keep three values distinct',sub)
para('Noisy state x<sub>t</sub>; estimated clean image x̂<sub>0</sub>; sampled previous state x<sub>t−1</sub>. The estimate x̂<sub>0</sub>=(x<sub>t</sub>−√(1−ᾱ<sub>t</sub>)ε̂)/√ᾱ<sub>t</sub> is not the next chain state.')
para('Scalar fixture: β<sub>1</sub>=β<sub>2</sub>=0.2, x<sub>0</sub>=1, ε=−0.5 → x<sub>2</sub>=0.5. For ε̂=−0.4: loss=0.01, x̂<sub>0</sub>=0.925, reverse mean=0.708088, variance=1/9, standard deviation=1/3. Fresh z=0.3 gives x<sub>1</sub>=0.808088.')
para('4. Remember the ELBO qualification',sub)
para('The negative variational bound contains a terminal prior KL, expected local reverse KLs, and a negative endpoint log likelihood. For t&gt;1, fixed reverse covariance gives a weighted noise error with w<sub>t</sub>=β<sub>t</sub>²/[2σ<sub>t</sub>²α<sub>t</sub>(1−ᾱ<sub>t</sub>)]. Unweighted MSE changes that weighting; it is not the entire exact negative ELBO.')
para('<link href="https://arxiv.org/abs/2006.11239" color="#178F82">Source: Ho, Jain &amp; Abbeel, DDPM (2020)</link>')
story.append(PageBreak())
para('Generation, conditioning and evidence',head)
para('5. Execute the sampler',sub)
para('Start from x<sub>T</sub> ~ N(0,I). For t=T…1, predict ε̂=ε<sub>θ</sub>(x<sub>t</sub>,t). Compute μ=[x<sub>t</sub>−β<sub>t</sub>ε̂/√(1−ᾱ<sub>t</sub>)]/√α<sub>t</sub>. Set σ²=β<sub>t</sub>(1−ᾱ<sub>t−1</sub>)/(1−ᾱ<sub>t</sub>). For t&gt;1 draw x<sub>t−1</sub>=μ+σz; at t=1 return μ without new noise. Model parameters stay fixed.')
para('This recurrence uses one-step α in the denominator. Skipping arbitrary steps is not a defined faster sampler. The teaching model uses 1,000 linear-schedule steps and terminal ᾱ ≈ 0.00004036.')
para('6. Assemble text-to-image components',sub)
para('A spatial autoencoder may map [B,3,512,512] to [B,4,64,64]: 64× fewer spatial positions but 48× fewer scalars. Those are representation ratios, not guaranteed runtime gains. Train the autoencoder, freeze it in the standard example, and train diffusion on correctly scaled latents. At ordinary text-to-image inference: latent noise → repeated denoising → decode once.')
para('In the selected cross-attention block, image features supply Q; text features supply K,V. A=softmax<sub>text positions</sub>(QK<super>T</super>/√d<sub>k</sub>); AV retains one row per image query. Original DiT is class-conditioned; different systems use different conditioning designs.')
para('Classifier-free guidance: ε<sub>cfg</sub>=ε<sub>u</sub>+s(ε<sub>c</sub>−ε<sub>u</sub>). s=0 is unconditional; s=1 is ordinary conditional; s&gt;1 extrapolates. Conditional dropout teaches both predictions using shared weights. Define the null condition explicitly.')
para('7. Change the state space for masked language',sub)
para('Token IDs are arbitrary lookup indices. A discrete mask process replaces positions with [MASK]; the model predicts vocabulary logits [B,L,V]. Simple masked-position CE is pedagogical. The stated LLaDA-style uniformly sampled-time objective adds a 1/t weight. Specify response-slot count, observed prompt tokens, reveal/remask policy and sampling schedule. Parallel logits do not alone establish faster generation.')
para('8. Read the evidence correctly',sub)
para('The new MNIST run: 814,145 parameters, 2,000 steps, seed 667, batch 64. Its unselected samples show malformed and similar digits. The older guidance example uses a separate 4,000-step class-conditioned cosine-schedule model. The eight-cluster demo is analytical; text traces and attention features are illustrative. A plausible scientific reconstruction is not a measured observation. No FID or downstream gain is asserted.')
para('<link href="https://arxiv.org/abs/2112.10752" color="#178F82">Latent diffusion</link> · <link href="https://arxiv.org/abs/2207.12598" color="#178F82">Classifier-free guidance</link> · <link href="https://arxiv.org/abs/2502.09992" color="#178F82">LLaDA</link>')
SimpleDocTemplate(str(OUT/'diffusion-two-page-recap.pdf'),pagesize=(595.28,841.89),leftMargin=43,rightMargin=43,topMargin=35,bottomMargin=35).build(story)
report={}
for f in OUT.glob('*.pdf'):
 reader=PdfReader(f);anns=[a.get_object() for p in reader.pages for a in p.get('/Annots',[])];report[f.name]={'pages':len(reader.pages),'external_links':sum('/A' in a and a['/A'].get('/S')=='/URI' for a in anns),'internal_links':sum('/Dest' in a for a in anns),'text_characters':sum(len(p.extract_text() or '') for p in reader.pages)}
(R/'qa/pdf.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
