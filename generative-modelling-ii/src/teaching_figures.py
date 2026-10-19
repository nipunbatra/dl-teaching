"""Progressive vector figures in the course's Beyond Attention visual language.

Tensor cells, token features and network internals are schematic unless their
values are explicitly shown. Raster panels reuse the recorded lecture assets.
"""
from html import escape
import math

B='#245EDB'; P='#793DB5'; T='#0F766E'; A='#9A5700'; R='#BE123C'; I='#14171F'; M='#4A5160'

def txt(x,y,s,c=I,size=24,anchor='middle',weight=500):
    return f'<text x="{x}" y="{y}" fill="{c}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(s))}</text>'

def rect(x,y,w,h,c=B,fill=None):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" stroke="{c}" stroke-width="2" fill="{fill or c+"10"}"/>'

def box(x,y,w,h,title,sub='',c=T):
    return rect(x,y,w,h,c)+txt(x+w/2,y+h/2-(4 if sub else -8),title,c,27)+ (txt(x+w/2,y+h/2+29,sub,M,20) if sub else '')

def arrow(x,y,xx,yy,c=M,via=None):
    path=f'M{x},{y} '+(via+' ' if via else '')+f'L{xx},{yy}'
    return f'<path d="{path}" fill="none" stroke="{c}" stroke-width="2.5" marker-end="url(#tip)"/>'

def group(step,s):return f'<g data-step="{step}">{s}</g>'

def grid(x,y,rows=4,cols=4,cell=15,c=B,mask=False):
    s=''
    for r in range(rows):
        for k in range(cols):
            fill=c if not mask or k<=r else '#F7F8FA'
            alpha=.35+.12*((r*3+k)%5) if fill==c else 1
            s+=f'<rect x="{x+k*cell}" y="{y+r*cell}" width="{cell-3}" height="{cell-3}" rx="2" fill="{fill}" opacity="{alpha}"/>'
    return s

def image(x,y,size,name,label,c=B):
    return f'<image href="assets/{name}" x="{x}" y="{y}" width="{size}" height="{size}" preserveAspectRatio="xMidYMid meet"/>'+txt(x+size/2,y+size+29,label,c,24)

def svg(s,label):
    return '<div class="teaching-figure"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 320" role="img" aria-label="'+escape(label)+'"><title>'+escape(label)+'</title><defs><marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1,1 L9,5 L1,9" fill="none" stroke="context-stroke" stroke-width="1.6"/></marker></defs><g font-family="Avenir Next, Segoe UI, sans-serif">'+s+'</g></svg></div>'

def training(grad=False):
    s=group(1,image(15,90,115,'digit.png','clean x₀')+arrow(143,148,224,148)+box(235,106,180,86,'Add noise','known ε, chosen t',A)+arrow(420,148,480,148)+image(490,90,115,'new-model-input.png','noisy xₜ')+txt(325,70,'We make the target',A,23))
    s+=group(1 if grad else 2,arrow(620,148,673,148)+box(680,106,200,86,'Denoiser','εθ(xₜ, t)',T)+txt(775,70,'trainable weights θ',T,21)+arrow(888,148,966,148)+txt(1030,150,'ε̂',T,43)+txt(1030,187,'prediction',M,20))
    s+=group(2 if grad else 3,arrow(325,200,1000,245,A,'V245 H960')+txt(605,277,'target branch: ε',A,23)+box(970,218,130,70,'MSE','',R)+arrow(1030,196,1030,212,R))
    if grad:s+=group(1,arrow(1000,213,805,93,R,'V30 H805')+txt(900,22,'gradient → θ',R,21))
    return svg(s,'Training branches: build the noisy input, predict its noise, and compare with the retained target. '+('Gradients update the denoiser.' if grad else ''))

def sampler(condition=False):
    s=image(20,83,125,'fresh-gaussian.png','start xT',A)+arrow(155,145,230,145)+box(240,100,205,90,'Denoiser','same weights θ',T)+arrow(455,145,515,145)+box(525,100,220,90,'Reverse step','mean + fresh noise',B)+arrow(755,145,830,145)+grid(845,83,6,6,21,B)+txt(907,237,'next state xₜ₋₁',B,24)
    s+=arrow(910,225,260,194,P,'V288 H260')+txt(590,280,'t ← t − 1',P,23)+txt(1015,128,'…',M,43)+txt(1015,180,'x₀',B,30)
    if condition:
        s+=box(190,5,300,58,'Optional condition c','',P)+arrow(343,69,343,92,P)+txt(795,40,'text / layout / observed pixels',P,23)
    else:s+=txt(560,40,'Sampling: θ held fixed',T,27)
    s+=txt(560,315,'State grid is schematic; the starting Gaussian panel is a recorded draw.',M,17)
    return svg(s,'Sampling loop: random noise, shared denoiser, reverse transition, feedback of sampled state.')

def corruption():
    s=group(1,grid(45,90,6,6,19,B)+txt(100,65,'Current state',B,26)+txt(100,240,'xₜ₋₁',B,30)+arrow(165,145,225,145,B)+box(235,105,200,85,'Scale signal','× √(1−βₜ)',B)+arrow(447,145,515,145,B))
    s+=group(2,grid(530,33,4,4,15,A)+txt(420,43,'Fresh noise',A,23)+txt(420,77,'√βₜ ηₜ',A,26)+arrow(558,99,558,126,A))
    s+=group(3,f'<circle cx="558" cy="151" r="23" fill="#F7F8FA" stroke="{T}" stroke-width="2"/>'+txt(558,160,'+',T,30)+arrow(587,151,700,151,T)+grid(720,93,6,6,19,T)+txt(775,65,'Next state',T,26)+txt(775,240,'xₜ',T,30)+txt(995,130,'Less signal',B,24)+txt(995,170,'More noise',A,24))
    return svg(s+txt(560,300,'Tensor cells are schematic; the equation defines the actual operation.',M,18),'A branched addition: scale current signal, add independent Gaussian noise, obtain the next state.')

def alpha():
    s=txt(20,32,'Signal variance retention',M,23,'start')
    for k,(label,w,value,step) in enumerate([('Start',500,'1',1),('After step 1',400,'0.8',1),('After step 2',320,'0.64',2)]):
        y=68+k*70
        s+=group(step,txt(185,y+26,label,M,24,'end')+rect(215,y,500,36,'#D9DDE5','#F7F8FA')+rect(215,y,w,36,B,B+'20')+txt(755,y+27,value,B,26,'start'))
    s+=group(2,txt(950,142,'0.8 × 0.8',T,29)+txt(950,181,'multiply the steps',T,22))
    s+=group(3,txt(560,302,'At step t: signal amplitude √ᾱₜ',I,25))
    return svg(s,'Two-step example: one-step alpha is 0.8; cumulative signal variance retention is 0.64.')

def conditional():
    s=txt(560,35,'Several noise values can be compatible with an observation',M,25)
    for mu,col,label in [(290,A,'possible εa'),(660,P,'possible εb')]:
        pts=[(x,240-130*math.exp(-.5*((x-mu)/95)**2)) for x in range(90,1000,4)]
        s+='<path d="'+' '.join(('M' if i==0 else 'L')+f'{x},{y:.1f}' for i,(x,y) in enumerate(pts))+f'" fill="none" stroke="{col}" stroke-width="3"/>'+txt(mu,275,label,col,24)
    s+=arrow(70,241,1030,241,M)
    s+=group(2,f'<path d="M475,95 V241" stroke="{T}" stroke-width="3"/>'+txt(475,75,'conditional mean',T,26))
    return svg(s+txt(560,313,'Schematic uncertainty, not a measured noise distribution.',M,17),'An illustrative conditional distribution with uncertainty; MSE predicts the conditional mean.')

def time_shared():
    s=''
    for y,label in [(60,'t = 7'),(220,'t = 920')]:
        s+=grid(45,y-32,4,4,15,A)+txt(165,y,label,A,27)+arrow(255,y-6,430,148,A)
    s+=group(1,box(440,100,255,96,'One denoiser','one set of weights θ',T))
    s+=group(2,arrow(708,148,835,148,T)+grid(860,102,6,6,16,T)+txt(910,240,'predicted noise',T,25))
    s+=group(3,box(330,240,385,62,'sin / cos → MLP → blocks','',A)+arrow(520,236,520,202,A))
    return svg(s,'Different timesteps enter one shared denoiser; time features condition its residual blocks.')

def next_state():
    s=image(50,85,135,'new-model-input.png','current xₜ')+arrow(195,150,320,150)+box(330,106,200,88,'Predict noise','ε̂ = εθ(xₜ,t)',T)
    s+=group(1,arrow(540,137,678,75,T)+box(690,32,370,80,'Clean estimate x̂₀','What might the original be?',T))
    s+=group(2,arrow(540,165,678,230,B)+box(690,190,370,80,'Sample previous state','Noise level matches t−1',B))
    s+=group(3,arrow(880,280,388,196,P,'V310 H388')+txt(500,296,'Next model call',P,22))
    return svg(s,'The noise prediction gives a clean estimate and a previous noisy state; only the previous state is fed back.')

def time_block():
    s=group(1,box(25,30,170,70,'Time t','',A)+arrow(205,65,260,65,A)+box(270,30,310,70,'sin / cos → MLP','',A)+arrow(590,65,665,65,A)+grid(680,40,1,8,26,A)+txt(900,92,'[B,C,1,1]',A,22))
    s+=group(1,grid(55,178,5,5,19,B)+txt(260,220,'Image features',B,25)+arrow(400,210,654,210,B)+f'<circle cx="680" cy="210" r="23" fill="#F7F8FA" stroke="{T}" stroke-width="2"/>'+txt(680,219,'+',T,30)+arrow(781,103,680,180,A,'V143 H680')+txt(845,137,'broadcast over H × W',A,22)+arrow(710,210,910,210,T)+grid(940,163,5,5,19,T))
    s+=group(2,txt(560,305,'encoder · middle · decoder',T,25))
    return svg(s,'Time embedding is projected to channels and broadcast across spatial positions before addition.')

def patches():
    s=group(1,grid(30,65,8,8,21,B)+txt(114,273,'4 × 32 × 32',B,25)+txt(114,35,'Spatial latent',B,25)+arrow(208,149,285,149,B))
    for i in range(4):s+=group(1,rect(305+i*61,105,47,82,B)+grid(313+i*61,115,4,2,14,B))
    s+=group(1,txt(424,65,'256 patch tokens',B,25)+txt(424,230,'16 entries each',M,23)+arrow(553,149,610,149))
    s+=group(2,box(620,98,220,102,'Transformer','dense token outputs',P)+arrow(850,149,905,149,P)+grid(930,65,8,8,21,T)+txt(1014,273,'4 × 32 × 32',T,25)+txt(1014,35,'Rebuild patches',T,24))
    return svg(s+txt(560,314,'Cells show patch structure schematically; no CLS pooling is used for this dense output.',M,18),'Latent grid to patch tokens, transformer, then unpatchification to a dense latent output.')

def latent(kind):
    if kind=='compression':
        s=grid(35,45,10,10,20,B)+txt(130,284,'3 × 512 × 512',B,25)+arrow(247,142,360,142,B)+box(370,100,245,88,'Image encoder','compress spatially',T)+arrow(630,142,775,142,T)+grid(805,105,4,4,20,P)+txt(846,284,'4 × 64 × 64',P,25)
        s+=group(2,txt(1000,100,'64×',B,33)+txt(1000,130,'fewer positions',M,19))
        s+=group(3,txt(1000,202,'48×',P,33)+txt(1000,231,'fewer entries',M,19))
        return svg(s,'Image compression: 64 times fewer spatial positions, 48 times fewer scalar entries for the stated shapes.')
    if kind=='train':
        s=group(1,txt(25,32,'1  Learn the representation',B,26,'start')+box(20,60,160,70,'Image','',B)+arrow(190,95,230,95)+box(240,60,170,70,'Encoder','',T)+arrow(420,95,465,95)+grid(480,70,3,4,17,P)+arrow(560,95,615,95)+box(625,60,170,70,'Decoder','',T)+arrow(805,95,860,95)+box(870,60,225,70,'Reconstruction','',B)+txt(555,160,'update encoder and decoder',B,23))
        s+=group(2,txt(25,198,'2  Learn to denoise latents',P,26,'start')+box(20,235,160,65,'Image','',B)+arrow(190,268,230,268)+box(240,235,170,65,'Encoder','',M)+arrow(420,268,465,268)+box(475,235,220,65,'Latent + noise','',A)+arrow(705,268,755,268)+box(765,235,330,65,'Train latent denoiser','',P)+txt(325,228,'fixed weights',M,17))
        return svg(s,'Two training stages: train the autoencoder, then freeze it and train the latent denoiser.')
    s=group(1,grid(25,98,6,6,18,A)+txt(80,249,'random zT',A,26)+arrow(145,149,260,149,A)+box(270,100,330,95,'Latent denoiser','repeat T → 1',P)+arrow(430,202,430,94,P,'V285 H650 V50 H430')+txt(578,42,'state feedback',P,21))
    s+=group(2,arrow(665,149,715,149,P)+box(725,100,175,95,'Decoder','run once',T)+arrow(912,149,975,149,T)+grid(990,82,8,8,18,B)+txt(1058,273,'Pixels',B,26))
    s+=group(3,txt(535,312,'For image-to-image, an encoder can prepare a supplied image first.',M,21))
    return svg(s,'Sampling begins with a random latent, repeatedly denoises it, then decodes to pixels once.')

def text_condition(cross=False):
    if not cross:
        s=group(1,txt(210,50,'“A basket full of cherries”',P,27)+box(60,92,300,80,'Tokeniser + text encoder','',P)+arrow(210,182,210,221,P))
        s+=group(2,grid(60,230,2,8,27,P)+txt(450,267,'C: [B,L,dc]',P,28))
        s+=group(3,arrow(565,253,765,198,P)+box(690,85,365,102,'Denoiser','reuse C at every step',T)+arrow(855,199,855,76,B,'V282 H1090 V40 H855')+txt(930,306,'changing image state',B,21))
        return svg(s,'Encode the prompt once to a sequence of text features; reuse them during denoising.')
    s=group(1,grid(20,48,4,5,19,B)+txt(70,30,'Image H',B,24)+arrow(130,85,215,85,B)+box(225,50,155,70,'Q = HWQ','',B)+grid(20,216,3,5,19,P)+txt(70,199,'Text C',P,24)+arrow(130,245,215,245,P)+box(225,210,155,70,'K, V','from text',P))
    s+=group(2,arrow(390,85,480,118,B)+arrow(390,236,480,158,P)+box(490,100,230,84,'QKᵀ / √dk','[N,L] per example',T))
    s+=group(3,arrow(730,142,786,142,T)+grid(800,85,4,3,28,T)+txt(844,229,'A: rows sum to 1',T,22))
    s+=group(4,arrow(900,142,957,142,T)+grid(975,85,4,4,25,B)+txt(1023,229,'AV',B,27)+txt(1023,259,'[N,dv]',M,22)+arrow(380,260,1088,169,P,'L445,302 H1088'))
    return svg(s,'Image queries and text keys produce attention over text positions; mixing text values preserves image query rows.')

def dropout():
    s=group(1,box(20,112,225,90,'Same noisy latent','inputs: zₜ and t',B)+arrow(255,157,350,157,B)+box(360,112,270,90,'One denoiser','shared parameters θ',T)+arrow(642,157,780,157,T)+txt(950,168,'noise prediction',T,28))
    s+=group(1,box(340,12,310,65,'Text condition c','',P)+arrow(495,82,495,104,P))
    s+=group(2,box(340,245,310,65,'Null condition','',A)+arrow(495,236,495,210,A)+txt(840,282,'sometimes drop c',A,25))
    s+=group(3,txt(170,268,'input and target stay fixed',B,20))
    return svg(s,'Train the shared denoiser with the regular condition or a null condition; keep the noisy latent and target unchanged.')

def guidance():
    # Plot exactly the vectors used by the attention/guidance demo.
    X=lambda x:155+x*530; Y=lambda y:240-y*180
    u=(.2,-.3);c=(.6,.1);g=(1.4,.9)
    s=arrow(155,260,155,25,M)+arrow(130,240,1015,240,M)+txt(155,286,'0',M,20)+txt(1065,244,'ε[0]',M,23)+txt(550,38,'εcfg = εu + s(εc − εu)',T,28)+txt(103,35,'ε[1]',M,23)
    for n,(v,col,label,step) in enumerate([(u,A,'s=0: εu',1),(c,P,'s=1: εc',2),(g,T,'s=3: extrapolate',3)]):
        x,y=X(v[0]),Y(v[1]);s+=group(step,arrow(155,240,x,y,col)+f'<circle cx="{x}" cy="{y}" r="6" fill="{col}"/>'+txt(x+10,y-13,label,col,24,'start'))
    s+=group(3,f'<path d="M{X(u[0])},{Y(u[1])} L{X(g[0])},{Y(g[1])}" stroke="{P}" stroke-dasharray="7 6" fill="none" stroke-width="2"/>')
    return svg(s+txt(600,314,'Illustrative vectors: εu = [.2, −.3], εc = [.6, .1]. Axes use different scales.',M,18),'Classifier-free guidance vectors: unconditional at s zero, conditional at s one, extrapolated at s three.')

def tokens(x,y,words,mask=False,spacing=112):
    s=''
    for i,w in enumerate(words):
        c=A if w=='?' else P if i<2 else T
        s+=rect(x+i*spacing,y,spacing-12,47,c)+txt(x+i*spacing+(spacing-12)/2,y+31,w,c,21 if spacing<112 else 23)
    return s

def masked():
    s=txt(20,27,'Observed prompt',P,23,'start')+txt(285,27,'Response slots',A,23,'start')
    s+=group(1,tokens(20,45,['The','dog','?','?','?'])+arrow(630,70,760,70,T)+txt(930,78,'predict a missing token',T,23)+tokens(20,118,['The','dog','chased','?','?'])+txt(790,149,'The prompt stays fixed',P,23))
    s+=group(2,tokens(20,191,['The','dog','chased','the','?'])+tokens(20,264,['The','dog','chased','the','ball'])+txt(817,256,'one slot per iteration',T,24)+txt(817,290,'scripted, no remasking',M,20))
    return svg(s,'Illustrative discrete generation fills three response slots one at a time while retaining the observed prompt.')

def compare_text():
    s=group(1,txt(20,28,'AUTOREGRESSIVE',P,24,'start')+tokens(20,48,['The','dog'],spacing=102)+tokens(20,108,['The','dog','chased'],spacing=102)+tokens(20,168,['The','dog','chased','the'],spacing=102)+tokens(20,228,['The','dog','chased','the','ball'],spacing=102)+txt(260,310,'extend the prefix, one token at a time',P,21))
    s+=group(2,txt(635,28,'MASKED DENOISING',T,24,'start')+tokens(635,48,['?','dog','?','the','ball'],spacing=92)+tokens(635,138,['The','dog','?','the','ball'],spacing=92)+tokens(635,228,['The','dog','chased','the','ball'],spacing=92)+txt(860,310,'use both sides of a gap',T,21))
    return svg(s,'Schematic comparison: autoregressive generation extends a prefix; masked prediction uses context around a gap.')

def unet():
    s=''
    positions=[(30,85),(245,145),(460,205),(675,145),(890,85)]
    for i,((x,y),shape,name) in enumerate(zip(positions,['28² × 32','14² × 64','7² × 128','14² × 64','28² × 32'],['Encoder 1','Encoder 2','Middle','Decoder 1','Decoder 2'])):
        step=1 if i<2 else 2 if i==2 else 3
        s+=group(step,rect(x+9,y-8,155,64,B,B+'08')+rect(x,y,155,64,B,B+'12')+txt(x+77,y+27,shape,B,24)+txt(x+77,y+51,name,M,18))
        if i<4:
            xx,yy=positions[i+1];s+=group(step,arrow(x+170,y+32,xx-10,yy+32,B))
    s+=group(3,arrow(105,75,965,77,T,'V42 H965')+txt(550,26,'concatenate 32 + 32 channels',T,20)+arrow(320,134,755,137,T,'V77 H755')+txt(540,70,'concatenate 64 + 64 channels',T,20)+txt(1010,216,'1 × 1 head',T,22)+txt(1010,252,'32 → 1',T,25)+arrow(968,156,1010,185,T))
    s+=group(2,txt(560,313,'814,145 trainable parameters in this implementation',A,23))
    return svg(s,'U-Net resolution and channel trace with skip concatenations and a linear one-channel output head.')

def loss_grid():
    mats=[([[-.5,1.0],[0.,-1.]],'Target ε',A),([[-.4,.8],[.1,-.7]],'Prediction ε̂',T),([[.01,.04],[.01,.09]],'Squared errors',R)]
    s=''
    for k,(mat,label,c) in enumerate(mats):
        x=30+k*380
        s+=txt(x+110,47,label,c,26)
        for i,row in enumerate(mat):
            for j,value in enumerate(row):
                s+=rect(x+j*112,78+i*74,100,62,c)+txt(x+j*112+50,119+i*74,f'{value:g}',c,28)
        if k<2:s+=txt(x+296,153,'→',M,35)
    s+=group(2,txt(560,285,'0.15 ÷ 4 = 0.0375',R,28))
    return svg(s,'Illustrative two by two target and prediction tensors; squared errors sum to 0.15 and average to 0.0375.')

def scalar_forward():
    s=group(1,txt(45,45,'Illustrative coordinate',M,22,'start')+txt(210,106,'x₀ = 1',B,34)+txt(210,246,'ε = −0.5',A,34))
    s+=group(2,arrow(315,97,425,97,B)+box(435,60,220,75,'× 0.8','signal scaling',B)+arrow(315,237,425,237,A)+box(435,200,220,75,'× 0.6','noise scaling',A))
    s+=group(3,arrow(665,97,755,97,B)+txt(818,106,'0.8',B,34)+arrow(665,237,755,237,A)+txt(818,246,'−0.3',A,34)+arrow(870,97,935,160,B)+arrow(870,237,935,178,A))
    s+=group(4,txt(1015,181,'0.5',T,44)+txt(1015,245,'x₂',T,28))
    return svg(s,'Worked forward coordinate: 0.8 times one plus 0.6 times negative 0.5 equals 0.5.')

def gaussian_step():
    s=txt(560,28,'Sample around the predicted mean',T,27)
    pts=[(x,242-175*math.exp(-.5*((x-540)/125)**2)) for x in range(140,950,4)]
    s+='<path d="'+' '.join(('M' if i==0 else 'L')+f'{x},{y:.1f}' for i,(x,y) in enumerate(pts))+f'" stroke="{T}" stroke-width="3" fill="none"/>'+arrow(110,242,1010,242,M)+txt(540,280,'mean μθ',T,24)
    s+=group(1,f'<path d="M540,67 V242" stroke="{T}" stroke-dasharray="5 5"/>')
    s+=group(2,f'<path d="M685,140 V242" stroke="{A}" stroke-width="3"/>'+txt(775,125,'one sampled state',A,24)+arrow(548,212,678,212,A)+txt(610,197,'σₜ ξ',A,24))
    s+=group(3,txt(560,310,'Schematic Gaussian transition; the spread comes from the chosen reverse variance.',M,18))
    return svg(s,'Schematic Gaussian reverse transition: mean, spread and a fresh sample. This is not a confidence interval.')

def dense_output():
    s=group(1,image(25,70,180,'new-model-input.png','noisy image xₜ',B)+arrow(225,158,390,158,B)+box(405,113,275,90,'Dense predictor','one output per position',T)+arrow(695,158,850,158,T)+image(880,70,180,'new-model-prediction.png','predicted noise ε̂',A))
    s+=group(2,txt(560,35,'The spatial shape stays [B,1,28,28]',B,28)+txt(560,307,'Recorded input and prediction from the same t=350 model call.',M,22))
    return svg(s,'Actual recorded noisy input and predicted noise at t350, each with the same spatial shape.')

FIGURES={
    's10':sampler(True), 's16':corruption(), 's17':alpha(), 's19':scalar_forward(),
    's21':training(), 's23':loss_grid(), 's24':conditional(), 's25':time_shared(), 's28':training(True),
    's32':sampler(), 's34':next_state(), 's36':gaussian_step(), 's40':dense_output(), 's41':unet(), 's42':time_block(), 's43':patches(),
    's44':latent('compression'), 's45':latent('train'), 's46':latent('sample'),
    's47':text_condition(), 's48':text_condition(True), 's50':text_condition(),
    's51':dropout(), 's52':guidance(), 's58':masked(), 's59':compare_text(),
}

def apply_figures(scenes):
    for scene in scenes:
        if scene['id'] in FIGURES:
            scene['visual']=FIGURES[scene['id']]
            # Two-build scenes still reveal the complete construction.
            import re
            scene['visual']=re.sub(r'data-step="(\d+)"',lambda m:f'data-step="{min(int(m[1]),scene["builds"])}"',scene['visual'])
            scene['layout']='figure-led'
            scene['visual_description']='Progressive authored SVG; tensor cells are schematic. '+re.sub('<[^>]+>',' ',scene['visual'])[:600]
