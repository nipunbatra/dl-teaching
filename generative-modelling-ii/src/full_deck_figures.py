"""Scene-specific visual treatments for every scene outside the first figure pass.

All crops are display viewports into attributed assets, not new model outputs.
Mathematical sketches are labelled illustrative; recorded evidence stays intact.
"""
from teaching_figures import *
from html import unescape
import re


def plate(content,label,height=400):
    return svg(content,label).replace('0 0 1120 320',f'0 0 1120 {height}')


_crop_serial=0

def raster(name,x,y,w,h,crop=None):
    global _crop_serial
    _crop_serial+=1
    if crop:
        cx,cy,cw,ch,iw,ih=crop
        return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="{cx} {cy} {cw} {ch}" preserveAspectRatio="xMidYMid meet" overflow="hidden"><defs><clipPath id="crop-{_crop_serial}" clipPathUnits="userSpaceOnUse"><rect x="{cx}" y="{cy}" width="{cw}" height="{ch}"/></clipPath></defs><image href="assets/{name}" width="{iw}" height="{ih}" clip-path="url(#crop-{_crop_serial})"/></svg>'
    return f'<image href="assets/{name}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet"/>'


def label(x,y,head,sub='',c=B):
    return txt(x,y,head,c,26,weight=650)+(txt(x,y+30,sub,M,20) if sub else '')


def line(x1,y1,x2,y2,c=M,dash=False):
    return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{c}" stroke-width="2"'+(' stroke-dasharray="5 5"' if dash else '')+'/>'


def dot(x,y,c=B,r=6):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}"/>'


def plot_curve(mu,sd,x0=70,x1=1060,y=280,height=150,c=B):
    pts=[(x,y-height*math.exp(-.5*((x-mu)/sd)**2)) for x in range(x0,x1,3)]
    return '<path d="'+' '.join(('M' if i==0 else 'L')+f'{x},{z:.1f}' for i,(x,z) in enumerate(pts))+f'" stroke="{c}" stroke-width="3" fill="none"/>'


def applications(n):
    if n==1:
        s=label(230,30,'Faces',c=B)+raster('ddpm.png',15,65,430,275,(0,0,510,513,1085,513))
        s+=label(690,30,'Rooms',c=T)+raster('ddpm.png',485,65,410,275,(510,0,470,256,1085,513))
        s+=label(1010,30,'Cats',c=P)+raster('ddpm.png',930,65,160,275,(855,256,125,257,1085,513))
        s+=txt(560,388,'Published unconditional samples · Ho, Jain & Abbeel (2020)',M,19)
    elif n==2:
        s=txt(560,34,'“A basket full of cherries”',P,30)
        for i in range(4):s+=group(1 if i==0 else 2,raster(f'ldm-{i+1}.png',15+i*280,80,250,250)+txt(140+i*280,347,f'Sample {i+1}',M,22))
        s+=txt(560,390,'Published LDM outputs · the source does not specify matched random seeds',M,18)
    elif n==3:
        s=txt(560,30,'“Space view of a planet and its sun”',P,28)
        s+=label(160,77,'Supplied layout',c=P)+raster('gligen.png',30,105,260,235,(25,25,450,430,2880,1160))
        s+=arrow(310,222,390,222,P)
        for i,k in enumerate([1,2,4]):s+=group(2,raster('gligen.png',420+i*230,105,210,235,(25+k*473,25,450,430,2880,1160)))
        s+=group(2,label(755,77,'Published samples',c=T))
        s+=txt(560,387,'GLIGEN · red box: planet · green box: sun',M,20)
    elif n==4:
        s=label(130,38,'Observed pixels',c=B)+raster('repaint-input.png',25,78,210,235,(3,6,165,174,174,198))
        s+=arrow(252,190,310,190,T)
        for i in range(5):
            s+=group(2,raster('repaint-outputs.png',330+i*155,92,145,195,(31+i*181,6,172,174,935,198))+txt(402+i*155,324,str(i+1),M,20))
        s+=group(2,txt(715,38,'Five possible fillings of the same missing region',T,24))
        s+=txt(560,384,'RePaint · blue marks the missing region · outputs are samples, not successive steps',M,19)
    elif n==5:
        s=label(160,35,'Input strokes',c=B)+raster('sdedit-input.png',20,70,280,280)+arrow(325,210,405,210,T)
        for i in range(2):s+=group(2,label(590+i*330,35,f'Published output {i+1}',c=T)+raster(f'sdedit-output-{i+1}.png',450+i*330,70,280,280))
        s+=txt(560,390,'SDEdit · the supplied image sets the starting point before corruption and denoising',M,19)
    elif n==6:
        names=['Bicubic','Regression','SR3 sample','Paired reference']
        s=''
        for i,name in enumerate(names):
            x=10+i*280
            s+=group(1 if i<2 else 2,label(x+130,28,name,c=T if i==2 else B)+raster('sr3-top.png',x,55,260,295,(15+i*362,56,348,702,1460,765)))
        s+=txt(560,382,'SR3 source comparison · the lower row enlarges the eye region',M,18)
    elif n==7:
        s=label(245,35,'Canny edge condition',c=B)+raster('control-pair.png',65,75,360,280,(6,64,357,325,748,394))
        s+=arrow(455,215,655,215,P)+txt(555,185,'ControlNet',P,25)
        s+=group(2,label(880,35,'Generated image',c=T)+raster('control-pair.png',700,75,360,280,(439,64,299,325,748,394)))
        s+=txt(560,393,'Published input–output pair · the edge map supplies spatial structure',M,19)
    elif n==8:
        s=txt(200,29,'ADAPTATION',B,22)+raster('dreambooth-input.png',10,60,260,285)+arrow(285,202,345,202,B)+box(355,150,220,100,'Adapted model','subject-specific weights',T)
        s+=group(2,txt(845,29,'GENERATION',P,22)+arrow(590,202,640,202,P)+raster('dreambooth-output.png',655,100,200,235,(26,16,400,393,1204,462))+raster('dreambooth-output.png',890,100,210,235,(857,16,330,393,1204,462))+txt(845,62,'New prompts',P,23)+txt(755,361,'At the Acropolis',M,20)+txt(995,361,'Getting a haircut',M,20))
        s+=txt(560,392,'DreamBooth · reference images enter during adaptation',M,20)
    elif n==9:
        s=label(265,29,'Published DiffusionSat samples',c=B)+txt(265,67,'“An electric substation in Finland”',P,22)+raster('sat.png',20,112,230,230,(1787,335,376,376,3479,1950))+raster('sat.png',280,112,230,230,(2188,335,376,376,3479,1950))+txt(135,99,'August',B,23)+txt(395,99,'January',B,23)
        s+=group(2,txt(810,30,'POSSIBLE LAB TASK',A,22)+grid(610,88,5,5,25,B)+rect(660,88,72,72,A,A+'50')+arrow(755,150,820,150,A)+grid(850,88,5,5,25,T)+txt(672,265,'Partial observation',B,23)+txt(915,265,'Candidate filling',T,23)+txt(800,317,'Compare with withheld measurements',A,22))
        s+=txt(265,385,'Same caption; month condition changed',M,18)+group(2,txt(817,385,'Proposed task; no local result claimed',A,18))
    return plate(s,'Application '+str(n)+': source evidence with explicit condition and generated outputs.')


def recap_generators():
    s=''
    for step,y,name,model in [(1,58,'VAE / GAN','Decoder / generator'),(2,238,'DIFFUSION','Shared denoiser')]:
        s+=group(step,txt(20,y-24,name,B if step==1 else T,24,'start')+grid(20,y,3,3,21,A)+arrow(102,y+30,235,y+30,A)+box(250,y-12,305,86,model,'',T)+arrow(570,y+30,700,y+30,T))
        if step==1:s+=group(step,image(730,y-20,100,'digit.png','sample',B))
        else:s+=group(step,grid(735,y-10,4,4,20,B)+arrow(780,y+83,365,y+81,P,'V368 H365')+txt(563,361,'sampled state returns',P,22)+txt(1000,y+38,'…  x₀',B,34))
    return plate(s,'Compare a one-pass latent generator with repeated state updates by a shared diffusion denoiser.')


def forward_strip():
    s=arrow(70,55,1065,55,A)+txt(560,28,'Same digit · one recorded recursive trajectory',B,26)
    for i,t in enumerate([0,100,300,500,700,1000]):
        s+=group(1 if i<2 else 2,raster(f'forward-{t}.png',10+i*187,105,170,170)+txt(95+i*187,315,f't = {t}',B if t==0 else A,24))
    return plate(s+txt(560,383,'MNIST test image 0 · seed 667 · fixed display range [−1,1]',M,20),'The same digit under six stages of a recorded forward corruption chain.')


def ambiguity():
    s=group(1,image(25,20,105,'digit.png','clean digit 7',B)+raster('clean-1.png',25,185,105,105)+txt(80,319,'clean digit 2',P,22))
    s+=arrow(160,75,275,150,B)+arrow(160,250,275,195,P)
    s+=plot_curve(500,112,300,1060,285,170,B)+plot_curve(740,112,300,1060,285,170,P)+arrow(300,285,1080,285)
    s+=txt(490,80,'q(xₜ | digit A)',B,24)+txt(835,80,'q(xₜ | digit B)',P,24)+txt(700,335,'One noisy coordinate',M,22)
    s+=group(2,line(620,140,620,285,T,True)+dot(620,285,T)+txt(620,118,'observed value',T,23))
    s+=txt(560,391,'Illustrative overlapping densities; the two full image arrays are not asserted to be equal.',M,18)
    return plate(s,'Two illustrative corruption densities overlap at a possible observed coordinate.')


def tensor():
    s=image(20,65,205,'digit.png','MNIST digit 7',B)+arrow(245,167,340,167,B)
    s+=grid(365,65,8,8,24,B)+txt(460,301,'28 × 28 positions',B,26)+rect(365+5*24,65+2*24,21,21,R,R+'30')
    s+=group(1,arrow(570,123,700,123,R)+txt(890,85,'One coordinate',R,26)+txt(890,138,'uint8 / 127.5 − 1',I,28)+line(730,225,1050,225)+dot(730,225,B)+dot(1050,225,B)+txt(730,266,'0 → −1',B,24)+txt(1050,266,'255 → +1',B,24))
    s+=group(2,txt(560,370,'Batch shape [B,1,28,28] · one channel per image',B,27))
    return plate(s,'A real digit, a schematic coordinate grid and the exact clean pixel normalisation.')


def combine_noise():
    s=group(1,box(25,48,235,78,'η₁ ~ N(0,1)','',A)+arrow(270,87,365,87,A)+box(375,48,265,78,'× √(α₂β₁)','variance α₂β₁',A)+box(25,204,235,78,'η₂ ~ N(0,1)','',P)+arrow(270,243,365,243,P)+box(375,204,265,78,'× √β₂','variance β₂',P))
    s+=group(2,arrow(650,88,760,163,A)+arrow(650,243,760,186,P)+box(775,135,315,90,'One Gaussian','variance 1 − ᾱ₂',T))
    s+=group(3,txt(560,331,'Signal mean: √ᾱₜ x₀',B,27))
    return plate(s,'Independent scaled noise terms combine by adding their variances.',350)


def batch():
    s=txt(155,29,'Three examples',B,25)+txt(435,29,'Gather coefficients',A,25)+txt(855,29,'One batched call',T,25)
    for i,t in enumerate([7,350,920]):
        y=65+i*90
        s+=group(1,raster(f'batch-{t}.png',110,y,74,74)+txt(285,y+42,f't={t}',A,23))
        s+=group(2,rect(355,y+9,158,58,A)+txt(434,y+46,f'ᾱ[{t}]',A,26)+arrow(525,y+38,650,196,A))
    s+=group(3,box(665,145,260,100,'model(xₜ, t)','[3,1,28,28]',T)+arrow(937,195,992,195,T)+grid(1008,150,4,4,20,T))
    s+=txt(560,381,'Gather [B] → reshape [B,1,1,1] → broadcast only across that example',M,23)
    return plate(s,'A separate timestep and coefficient per example, reshaped for correct batched broadcasting.')


def data_labels():
    s=label(170,32,'Clean examples',c=B)+''.join(raster(f'clean-{i}.png',15+i*80,60,72,90) for i in range(4))+txt(170,192,'The data determine coverage',B,23)
    s+=group(1,arrow(350,110,435,110,B)+box(450,65,270,90,'Draw Gaussian noise','known numerical target',A)+arrow(730,110,820,110,A)+grid(845,65,5,5,20,A))
    s+=group(2,txt(560,287,'If training contains only sevens…',P,29)+txt(560,335,'extra corruption draws do not supply examples of other digits.',M,24))
    return plate(s,'Automatically constructed targets do not replace a representative clean dataset.')


def bound_path():
    s=txt(20,28,'FIXED FORWARD PROCESS q',A,22,'start')
    xs=[100,350,600,850,1040]; labs=['x₀','x₁','x₂','…','xT']
    for i,(x,l) in enumerate(zip(xs,labs)):
        s+=rect(x-48,60,96,70,B)+txt(x,106,l,B,32)
        if i<4:s+=arrow(x+58,81,xs[i+1]-60,81,A)
        if i>0:s+=group(2,arrow(x-58,113,xs[i-1]+60,113,T))
    s+=group(2,txt(560,175,'LEARNED REVERSE PROCESS pθ',T,22))
    for k,(x,w,title,sub,c) in enumerate([(15,250,'Endpoint','reconstruct x₀',B),(300,510,'Local reverse terms','compare q(xₜ₋₁ | xₜ,x₀) with pθ(xₜ₋₁ | xₜ)',T),(845,260,'Prior term','match q(xT | x₀) to p(xT)',A)]):
        s+=group(3,rect(x,215,w,91,c)+txt(x+w/2,249,title,c,25)+txt(x+w/2,284,sub,M,16 if w<300 else 18))
    return plate(s,'One latent path with forward and learned reverse arrows, grouped into the three bound contributions.',330)


def gaussian_loss():
    s=plot_curve(390,92,80,1040,230,150,B)+plot_curve(650,92,80,1040,230,150,T)+arrow(60,230,1070,230)
    s+=txt(390,40,'Training posterior',B,26)+txt(650,78,'Model transition',T,26)+txt(390,275,'μ̃ₜ',B,30)+txt(650,275,'μθ',T,30)
    s+=group(1,line(390,245,650,245,R)+txt(520,310,'mean error',R,23))
    return plate(s+txt(1000,310,'fixed variances',M,18),'With fixed covariance, Gaussian KL depends quadratically on the difference of means.',330)


def clean_estimate():
    s=txt(150,43,'Current coordinate',B,26)+txt(150,130,'0.5',B,45)+txt(150,175,'xₜ',B,25)
    s+=group(1,arrow(230,119,345,119,A)+box(355,65,310,112,'Subtract noise','− 0.6 × (−0.4)',A)+txt(510,222,'0.74 remains',A,28))
    s+=group(2,arrow(675,119,775,119,T)+box(785,65,300,112,'Undo signal scaling','÷ 0.8',T))
    s+=group(3,txt(935,247,'0.925',T,45)+txt(935,290,'estimated clean value',T,22))
    return plate(s,'Worked inverse of the forward marginal: subtract estimated noise, then divide by the signal coefficient.',320)


def mean_step():
    s=group(1,txt(120,60,'Current xₜ',B,27)+txt(120,130,'0.5',B,42)+txt(120,208,'Predicted ε̂',T,25)+txt(120,271,'−0.4',T,42))
    s+=group(2,arrow(225,245,355,245,T)+box(365,200,300,90,'Scale correction','cₜ = βₜ / √(1−ᾱₜ)',A)+txt(515,164,'c₂ = 1/3',A,25))
    s+=group(3,arrow(225,114,760,114,B)+arrow(675,245,760,133,A)+txt(823,133,'−',R,38))
    s+=group(4,arrow(860,123,920,123,T)+txt(1000,86,'÷ √αₜ',T,30)+txt(1000,172,'μθ',T,38)+txt(1000,221,'0.708088',T,27))
    return plate(s,'An internally consistent beta point two reverse-mean fixture; one-step alpha is used in the final division.',310)


def trace():
    s=txt(50,128,'State',B,23)+txt(50,163,'xₜ',B,30)+group(2,txt(50,288,'Estimate',T,22)+txt(50,323,'x̂₀',T,30))
    for j,t in enumerate([1000,800,600,400,200,50,1]):
        x=145+j*138
        s+=txt(x+63,39,f't={t}',B,23)+raster('new-model-trace.png',x,65,126,126,(100+j*132,28,126,126,1024,320))
        s+=group(2,raster('new-model-trace.png',x,235,126,126,(100+j*132,179,126,126,1024,320)))
    return plate(s+txt(560,394,'Actual checkpoint replay · seed 667 · display clipping only',M,18),'Recorded chain states and clean estimates in separately revealed, aligned rows.')


def cfg_gallery():
    s=''
    for i,scale in enumerate([0,1,3,7]):
        y=35+i*82
        s+=group(1 if i<2 else 2,txt(56,y+48,f's={scale}',P,27)+raster('mnist_cfg_comparison.png',115,y,990,77,(90,90+i*211,1605,186,1715,955)))
    s+=txt(560,393,'Request: digit 7 · identical random draws across rows · old checkpoint, seed 1002',M,19)
    return plate(s,'All eight outputs at each of four guidance scales, aligned by common random draws.')


def observation():
    s=group(1,txt(265,30,'INPAINTING',B,24)+raster('repaint-input.png',20,75,190,230,(3,6,165,174,174,198))+arrow(230,188,310,188,B)+raster('repaint-outputs.png',330,75,190,230,(31,6,172,174,935,198))+txt(265,350,'Known region + missing region',B,24))
    s+=group(2,txt(845,30,'SUPER-RESOLUTION',P,24)+raster('sr3-top.png',610,75,190,230,(15,56,348,350,1460,765))+arrow(815,188,865,188,P)+raster('sr3-top.png',885,75,190,230,(739,56,348,350,1460,765))+txt(845,350,'Low-resolution observation + inferred detail',P,22))
    return plate(s+txt(560,382,'Published RePaint and SR3 examples; each method has its own observation model.',M,19),'Compare missing spatial regions with missing high-frequency detail, using the opening examples.')


def evidence_gallery():
    s=txt(560,35,'One unselected batch',B,27)+raster('new-model-gallery.png',0,78,1120,175)
    for i,(x,a,b,c) in enumerate([(170,'2,000','training steps',B),(560,'667','sampling seed',P),(950,'Not computed','quality metrics',M)]):
        s+=group(min(i+1,3),txt(x,315,a,c,31)+txt(x,352,b,M,23))
    return plate(s,'The complete unselected eight-sample batch from the new model, with its actual budget and evaluation status.')


def token_ids():
    s='';words=['The','dog','chased','the','ball'];a=[8,24,3,9,11];b=[501,2,83,17,4]
    for row,(name,vals,c) in enumerate([('Sentence',words,P),('IDs A',a,B),('IDs B',b,T)]):
        y=35+row*91;s+=txt(110,y+34,name,c,24)
        for j,val in enumerate(vals):s+=rect(230+j*176,y,150,60,c)+txt(305+j*176,y+39,val,c,28)
    s+=group(2,tokens(230,320,['The','?','chased','?','ball'],spacing=176)+txt(110,352,'Masking',A,24))
    return plate(s,'One sentence under two arbitrary vocabulary numberings, followed by discrete masking.')


def recap():
    s='';rows=[('Pixels','partial image','known noise','respect observed region'),('Latents','prompt features','latent noise','denoise, then decode'),('Tokens','visible context','missing tokens','fill response slots')]
    widths=[230,290,260,300];xs=[10,250,550,820]
    for i,(h,c) in enumerate([('State',B),('Condition',P),('Training target',A),('Sampling rule',T)]):s+=group(i+1,txt(xs[i]+widths[i]/2,37,h,c,26))
    for r,row in enumerate(rows):
        y=85+r*95
        for i,(v,c) in enumerate(zip(row,[B,P,A,T])):s+=group(i+1,rect(xs[i],y,widths[i]-10,67,c)+txt(xs[i]+(widths[i]-10)/2,y+41,v,c,21))
    s+=txt(560,382,'A comparison of the teaching constructions; full systems can use other targets and samplers.',M,18)
    return plate(s,'A progressively completed comparison table across pixel, latent and token generation.')


def appendix_figure(n):
    if n==1:
        s=rect(25,40,270,180,A)+txt(160,85,'aX',A,34)+txt(160,133,'mean: a mₓ',A,26)+txt(160,178,'variance: a²vₓ',A,26)+txt(350,145,'+',M,40)
        s+=group(2,rect(410,40,270,180,P)+txt(545,85,'bY',P,34)+txt(545,133,'mean: b mᵧ',P,26)+txt(545,178,'variance: b²vᵧ',P,26)+arrow(700,130,780,130,T)+txt(950,107,'Gaussian sum',T,28)+txt(950,157,'variances add',T,25))
    elif n==2:
        s=arrow(170,235,170,25)+txt(70,35,'value',M,20)+line(200,55,1020,55,B)+txt(900,39,'log evidence',B,25)+group(2,line(200,210,1020,210,T)+txt(900,245,'ELBO',T,25))
        s+=group(3,line(490,55,490,210,P)+line(480,55,500,55,P)+line(480,210,500,210,P)+txt(705,139,'KL gap ≥ 0',P,31))
    elif n==3:return bound_path()
    elif n==4:
        s=txt(200,38,'Known at training time',B,24)+box(70,65,245,75,'Clean x₀','',B)+box(70,185,245,75,'Noisy xₜ','',A)
        s+=group(1,arrow(328,103,445,156,B)+arrow(328,224,445,184,A)+box(460,125,300,85,'Posterior q','xₜ₋₁ | xₜ,x₀',T))
        s+=group(2,plot_curve(945,47,790,1100,245,115,T)+txt(945,70,'Gaussian',T,25)+txt(945,285,'μ̃ₜ, β̃ₜ',T,27))
    elif n==5:
        s=arrow(240,205,240,20,M)+arrow(140,205,520,205,M)+txt(540,213,'x₀',B,26)+txt(216,45,'ε',A,26)
        s+=group(1,arrow(240,205,480,85,T)+txt(503,81,'xₜ',T,27)+arrow(240,205,156,37,P)+txt(127,34,'v',P,27))
        s+=group(2,txt(830,80,'Forward: rotate (x₀, ε)',T,27)+txt(830,132,'ᾱₜ + (1−ᾱₜ) = 1',M,27))
        s+=group(3,txt(830,219,'Inverse: transpose the rotation',P,24))
    elif n==6:
        s=txt(50,25,'DDPM',B,24,'start')
        for i,name in enumerate(['xT','xₜ','xₜ₋₁','…','x₁','x₀']):
            x=80+i*190;s+=dot(x,75,B,9)+txt(x,115,name,B,26)
            if i<5:s+=arrow(x+15,75,x+170,75,B)
        s+=group(2,txt(50,175,'DDIM, η=0',T,24,'start')+dot(80,220,T,9)+dot(650,220,T,9)+dot(1030,220,T,9)+arrow(100,220,630,220,T)+arrow(670,220,1010,220,T)+txt(80,265,'xₜ',T,27)+txt(650,265,'xₛ',T,27)+txt(1030,265,'x₀',T,27)+txt(560,199,'A separately defined jump',T,23))
    elif n==7:
        s=''
        for i in range(8):
            x=245+115*math.cos(i*math.pi/4);y=150+115*math.sin(i*math.pi/4)
            s+=dot(x,y,B,9)
            for j in range(8):s+=dot(x+12*math.cos(j*math.pi/4),y+12*math.sin(j*math.pi/4),B,2)
        s+=dot(280,140,R,7)+txt(300,129,'x',R,25)
        s+=group(2,arrow(280,140,355,150,B)+arrow(280,140,326,68,B)+txt(630,90,'Responsibilities rₖ',P,28)+txt(630,137,'nearby centres get more weight',M,22))
        s+=group(3,arrow(805,147,900,147,T)+txt(1000,139,'Σ rₖ scoreₖ',T,25)+txt(1000,180,'mixture score',T,22))
    elif n==8:
        s=dot(90,220,A,10)+txt(90,268,'xnoise',A,24)+dot(1020,60,B,10)+txt(1020,33,'xdata',B,24)+line(90,220,1020,60,M)
        s+=group(1,dot(462,156,P,9)+txt(462,200,'xτ at τ=0.4',P,25))
        s+=group(2,arrow(462,156,690,117,T)+txt(650,83,'u = xdata − xnoise',T,27))
        s+=group(3,txt(800,267,'Learned field averages compatible pair velocities',M,21))
    elif n==9:
        s=tokens(10,35,['The','dog','?','the','?'],spacing=146)+txt(935,70,'t = 0.5',A,28)
        s+=group(1,arrow(365,95,365,150,A)+txt(365,192,'CE(chased)',T,25)+arrow(657,95,657,150,A)+txt(657,192,'CE(ball)',T,25))
        s+=group(2,line(280,225,740,225,T)+txt(510,268,'sum over masked positions',T,24)+txt(935,187,'× 1/t',A,34)+txt(935,231,'here: × 2',A,25))
    elif n==10:
        s=grid(25,75,5,5,24,B)+txt(85,50,'Image H',B,25)+arrow(155,135,245,135,B)
        s+=group(1,box(255,90,195,90,'Flatten','N = H×W',B))
        s+=group(2,arrow(460,135,535,135,B))
        for i in range(3):s+=group(2,rect(550+i*22,95-i*18,120,100,P)+(txt(610+i*22,153-i*18,'N × L',P,22) if i==2 else ''))
        s+=group(2,txt(630,235,'h attention heads',P,24))
        s+=group(3,arrow(735,135,805,135,T)+box(815,90,280,90,'Mix V, merge heads','restore H × W',T))
    elif n==11:
        s='';rows=[('Published','Exact source + supplied condition','Source figure',B),('Local','Checkpoint + seed + sampler + budget','Recorded run',T),('Proposed','Observation model + validation plan','No measured result',A)]
        for i,(h,m,r,c) in enumerate(rows):
            y=15+i*99;s+=group(i+1,rect(10,y,1100,76,c)+txt(115,y+46,h,c,27)+txt(550,y+46,m,I,24)+txt(975,y+46,r,c,21))
    return plate(s,'Appendix '+str(n)+': graphical support for the mathematical construction.',330 if n in [3,4] else 300)


def code_figure(scene):
    raw=unescape(re.search(r'<pre>([\s\S]*?)</pre>',scene['visual'])[1])
    lines=raw.split('\n'); training=scene['id']=='s27'
    phases=([1,1,2,3,4,4,4,4,4] if training else [1,1,1,1,1,1,1,2,2,2,2])
    code=''
    for i,line_ in enumerate(lines):
        step=phases[min(i,len(phases)-1)]
        code+=group(step,txt(18,38+i*31,f'{i+1:02}',M,17,'start')+txt(66,38+i*31,line_,I,20,'start'))
    code='<g font-family="SF Mono, Menlo, monospace" xml:space="preserve" style="white-space:pre">'+code+'</g>'
    if training:
        side=group(1,box(840,12,260,75,'Draw t and ε','',A))+group(2,arrow(970,95,970,119,A)+box(840,130,260,75,'Build noisy batch','',B))+group(3,arrow(970,215,970,239,T)+box(840,250,260,75,'Predict noise','',T))+group(4,txt(970,374,'loss → update θ',R,26))
    else:
        side=group(1,box(840,12,260,75,'Start from noise','',A)+arrow(970,94,970,118,A)+box(840,130,260,75,'Predict → mean','',T))+group(2,arrow(970,215,970,239,T)+box(840,250,260,75,'Sample state','except t=1',B)+arrow(1107,290,1107,165,P,'H1115 V108 H1107')+txt(970,376,'return state to model',P,22))
    return plate(code+side,'Annotated '+('training update' if training else 'sampling loop')+' code; the right margin tracks data and control flow.')


def repeated_condition():
    s=txt(560,32,'One encoded prompt C',P,29)+grid(445,52,2,9,24,P)
    for i,t in enumerate(['t','t−1','t−2']):
        x=50+i*380
        s+=group(1,box(x,180,260,92,'Denoiser',f'noisy state at {t}',T)+arrow(555,110,x+130,170,P,'V140 H'+str(x+130)))
        if i<2:s+=group(2,arrow(x+270,226,x+365,226,B))
    s+=group(2,txt(560,355,'Image features change at each call',B,28))
    return plate(s,'One text feature sequence branches into three denoising calls with changing image states.')


NEW={f's{i:02}':applications(i) for i in range(1,10)}
NEW.update({'s11':recap_generators(),'s12':forward_strip(),'s13':ambiguity(),'s15':tensor(),'s18':combine_noise(),'s26':batch(),'s29':data_labels(),'s30':bound_path(),'s31':gaussian_loss(),'s33':clean_estimate(),'s35':mean_step(),'s39':trace(),'s53':cfg_gallery(),'s54':observation(),'s55':evidence_gallery(),'s56':token_ids(),'s60':recap(),'s50':repeated_condition()})
NEW.update({f'a{i:02}':appendix_figure(i) for i in range(1,12)})

# Keep only equations that add information beyond the diagram. Original prose and
# all derivation steps remain in the notes, with their original build numbers.
MATH_SCENES={'s16','s17','s18','s24','s30','s31','s33','s35','s36','a01','a02','a03','a04','a05','a06','a07','a08','a09','a10'}


def apply_full_deck(scenes):
    for s in scenes:
        s['explanation_steps']=list(s['steps'])
        if s['id'] in ['s27','s38']:
            s['visual']=code_figure(s);s['layout']='figure-led code-figure'
        if s['id'] in NEW:
            s['visual']=NEW[s['id']];s['layout']='figure-led'
        if s['id'] in MATH_SCENES:
            maths=[''.join(re.findall(r'<div class="math">[\s\S]*?</div>',v)) for v in s['steps']]
            for j in range(1,len(maths)):
                if not maths[j]:maths[j]=maths[j-1]
            s['steps']=maths
            s['layout']='figure-led has-equation'
        else:s['steps']=['' for _ in s['steps']]
        if 'figure-led' in s['layout']:
            s['visual']=re.sub(r'data-step="(\d+)"',lambda m:f'data-step="{min(int(m[1]),s["builds"])}"',s['visual'])
            s['visual_description']=re.search(r'aria-label="([^"]+)"',s['visual'])[1]
        if s['id']=='a03':
            from scenes import eq
            s['steps'][-1]=eq(r'\begin{aligned}\mathcal L_{VLB}={}&D_{KL}(q(x_T|x_0)\Vert p(x_T))-\mathbb E_q\log p_\theta(x_0|x_1)\\&+\sum_{t=2}^T\mathbb E_q D_{KL}(q(x_{t-1}|x_t,x_0)\Vert p_\theta(x_{t-1}|x_t))\end{aligned}')
        if s['id']=='a08':
            from scenes import eq
            s['steps'][-1]=eq(r'\frac{d x_\tau}{d\tau}=v_\theta(x_\tau,\tau),\qquad x_{\tau=0}\sim p_{noise}')
        s['visual_revision']='Whole-deck scene review: source-aware figures, unique visible labels, retained presenter cue and takeaway.'
