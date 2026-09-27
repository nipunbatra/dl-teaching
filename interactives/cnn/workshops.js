/* Source-backed teaching experiments. No network or UI dependencies. */
(() => {
  'use strict';
  const M=window.CNN,D=window.CNN_MNIST,L=window.CNNLeNet;
  const $=(s,r=document)=>r.querySelector(s),all=(s,r=document)=>[...r.querySelectorAll(s)];
  const fmt=(v,d=3)=>Number.isInteger(v)?v.toLocaleString('en-US'):v.toFixed(d);
  function el(tag,cls,text){const e=document.createElement(tag);e.className=cls||'';if(text!==undefined)e.textContent=text;return e;}
  function select(label,values,value){const l=el('label','',label),s=el('select');values.forEach(([v,t])=>s.add(new Option(t,v)));s.value=value;l.append(s);return [l,s];}
  function eq(s){return el('div','equation',s);}
  function table(head,rows){const t=el('div','table-wrap');t.innerHTML='<table><thead><tr>'+head.map(v=>'<th>'+v+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table>';return t;}
  function grid(x,{label='',marked=()=>false,scale=1,numbers=true,cell=34}={}){
    const box=el('div','matrix-wrap');box.append(el('div','cap',label));const g=el('div','matrix');g.style.setProperty('--cols',x[0].length);g.style.setProperty('--cell',cell+'px');
    x.forEach((row,r)=>row.forEach((v,c)=>{const a=el('span','cell'+(marked(r,c)?' sampled':''),numbers?fmt(v,1):'');const t=Math.min(1,Math.abs(v)/scale);a.style.background=`color-mix(in srgb, ${v<0?'#245edb':'#be123c'} ${t*65}%,white)`;a.title=`[${r},${c}] = ${v}`;g.append(a);}));box.append(g);return box;
  }
  function heat(values,n,{signed=false,size=150,label='',max,lightHigh=false}={}){
    const wrap=el('div','heat-tile'),canvas=el('canvas');canvas.width=n;canvas.height=Math.ceil(values.length/n);canvas.style.width=size+'px';canvas.style.height=(size*canvas.height/n)+'px';canvas.setAttribute('aria-label',label);const ctx=canvas.getContext('2d'),im=ctx.createImageData(n,canvas.height);
    max=max??Math.max(1e-8,...Array.from(values,Math.abs));
    for(let i=0;i<values.length;i++){let v=values[i],t=Math.min(1,Math.abs(v)/max);if(lightHigh)t=1-t;const rgb=signed?(v<0?[36,94,219]:[190,18,60]):[20,23,31];for(let q=0;q<3;q++)im.data[i*4+q]=255*(1-t)+rgb[q]*t;im.data[i*4+3]=255;}
    ctx.putImageData(im,0,0);wrap.append(canvas,el('div','cap',label));return wrap;
  }
  function bars(p){const b=el('div','digit-bars');p.forEach((v,i)=>{const row=el('div','digit-bar');row.innerHTML=`<span>${i}</span><div><i style="width:${v*100}%"></i></div><output>${(v*100).toFixed(1)}%</output>`;b.append(row);});return b;}
  const state={sample:0,epoch:'10',channel:0};let cachedKey='',cached;
  const trace=()=>{const k=state.sample+':'+state.epoch;if(k!==cachedKey){cached=L.forward(D.checkpoints[state.epoch],D.samples[state.sample].pixels);cachedKey=k;}return cached;};
  const renderers=[];
  function sharedControls(host,{epochs=false,channels=0}={}){
    const c=el('div','controls'),[sl,s]=select('Test image',D.samples.map((x,i)=>[String(i),`${x.index} · label ${x.label}${i>9?' · error example':''}`]),String(state.sample));c.append(sl);
    s.oninput=()=>{state.sample=+s.value;renderAll();};
    let e,q;if(epochs){const pair=select('Saved training epoch',[[0,'0 · random'],[1,'1'],[3,'3'],[10,'10 · trained']],state.epoch);e=pair[1];e.oninput=()=>{state.epoch=e.value;renderAll();};c.append(pair[0]);}
    if(channels){const pair=select('Output channel',Array.from({length:channels},(_,i)=>[i,String(i)]),String(state.channel%channels));q=pair[1];q.oninput=()=>{state.channel=+q.value;renderAll();};c.append(pair[0]);}
    host.append(c);return ()=>{s.value=String(state.sample);if(e)e.value=state.epoch;if(q)q.value=String(state.channel%channels);};
  }
  function renderAll(){renderers.forEach(fn=>fn());}
  all('[data-lenet]').forEach(host=>{
    const stage=host.dataset.lenet,sync=sharedControls(host,{epochs:true,channels:['conv1','relu1','pool1','patch'].includes(stage)?6:['conv2','relu2','pool2'].includes(stage)?16:0}),out=el('div','lenet-output');host.append(out);
    function render(){sync();const a=trace(),sample=D.samples[state.sample],w=D.checkpoints[state.epoch];out.replaceChildren();
      if(stage==='overview'||stage==='prediction'){
        const col=el('div','columns'),left=el('div','digit-summary');left.append(heat(a.input,28,{size:220,label:`MNIST test ${sample.index} · label ${sample.label}`,max:1}));
        left.append(eq(`Predicted ${a.prob.indexOf(Math.max(...a.prob))} · true ${sample.label}\nCross-entropy ${(-Math.log(a.prob[sample.label])).toFixed(3)}`));col.append(left,bars(a.prob));out.append(col);
      } else if(stage==='filters'){
        const maps=el('div','feature-gallery six');const max=Math.max(...w['conv1.weight'].map(Math.abs));for(let q=0;q<6;q++)maps.append(heat(w['conv1.weight'].slice(q*25,q*25+25),5,{signed:true,max,size:118,label:`K${q} · b=${w['conv1.bias'][q].toFixed(3)}`}));out.append(maps,el('p','note',`All six 5×5 kernels share one signed colour scale ±${max.toFixed(3)}. These are learned weights, not six input images.`));
      } else if(['conv1','relu1','pool1','conv2','relu2','pool2'].includes(stage)){
        const channels=stage.endsWith('1')?6:16,n={conv1:24,relu1:24,pool1:12,conv2:8,relu2:8,pool2:4}[stage],values=a[stage],q=state.channel%channels,max=Math.max(...Array.from(values,Math.abs)),gallery=el('div','feature-gallery '+(channels===6?'six':'sixteen'));
        for(let c=0;c<channels;c++){const tile=heat(values.slice(c*n*n,(c+1)*n*n),n,{signed:stage.startsWith('conv'),max,size:channels===6?110:58,label:`channel ${c}`});if(c===q)tile.classList.add('selected-map');gallery.append(tile);}
        const col=el('div','columns wide'),focus=el('div');focus.append(heat(values.slice(q*n*n,(q+1)*n*n),n,{signed:stage.startsWith('conv'),max,size:210,label:`Selected ${q} · ${n}×${n}`}));focus.append(eq(`Tensor [1, ${channels}, ${n}, ${n}]\n${channels*n*n} activations · epoch ${state.epoch}`));col.append(gallery,focus);out.append(col,el('p','note',`One common colour scale within this stage: ${stage.startsWith('conv')?'signed ±':'0 to '}${max.toFixed(3)}. This is an activation map, not an attention or saliency map.`));
      } else if(stage==='flat'){
        const tiles=el('div','feature-gallery sixteen flat-gallery');for(let q=0;q<16;q++)tiles.append(heat(a.pool2.slice(q*16,q*16+16),4,{max:Math.max(...a.pool2),size:64,label:`${q*16}–${q*16+15}`}));out.append(tiles,eq(`16 channels × 4 rows × 4 columns = 256 values\nFirst 8: ${Array.from(a.flat.slice(0,8),v=>v.toFixed(3)).join(' · ')}`));
      } else if(stage==='hidden'){
        const row=el('div','feature-gallery');row.append(heat(a.hidden1,12,{size:230,label:'120 features (12×10 display)',max:Math.max(...a.hidden1)}),heat(a.hidden2,12,{size:230,label:'84 features (12×7 display)',max:Math.max(...a.hidden2)}));out.append(row,eq(`256 → Linear + ReLU → 120 → Linear + ReLU → 84\nThese display rows are an arrangement of a vector, not image coordinates.`));
      } else if(stage==='logits'){
        const winner=a.prob.indexOf(Math.max(...a.prob));out.append(table(['Digit','Logit','Probability'],Array.from({length:10},(_,i)=>[`${i}${i===sample.label?' · true':''}${i===winner?' · predicted':''}`,a.logits[i].toFixed(3),(100*a.prob[i]).toFixed(2)+'%'])));
      } else if(stage==='patch'){
        const q=state.channel%6,r=8,c=8,patch=M.grid(5,5,(i,j)=>a.input[(r+i)*28+c+j]),kernel=M.grid(5,5,(i,j)=>w['conv1.weight'][q*25+i*5+j]),row=el('div','matrices');row.append(grid(patch,{label:'Input patch at [8,8]',scale:1}),el('span','operator','·'),grid(kernel,{label:`Learned K${q}`,scale:.5}),el('span','operator','+ b'));out.append(row,eq(`Σ 25 products + ${w['conv1.bias'][q].toFixed(4)} = ${a.conv1[q*24*24+8*24+8].toFixed(4)}\nReLU → ${a.relu1[q*24*24+8*24+8].toFixed(4)}`));
      }
    }
    renderers.push(render);
  });
  all('[data-experiment]').forEach(host=>{
    const kind=host.dataset.experiment;
    if(kind==='coverage'){
      const [l,s]=select('Zero padding',[[0,'0 · valid'],[1,'1 · same']],0);host.append(l);const out=el('div');host.append(out);
      function run(){const p=+s.value,n=5,count=M.grid(5,5),width=M.size(n,3,p);for(let r=0;r<width;r++)for(let c=0;c<width;c++)for(let i=0;i<3;i++)for(let j=0;j<3;j++){const rr=r+i-p,cc=c+j-p;if(rr>=0&&cc>=0&&rr<n&&cc<n)count[rr][cc]++;}out.replaceChildren(grid(count,{label:'How often is each input pixel used?',scale:9,cell:48}),eq(`Corner ${count[0][0]} uses · centre ${count[2][2]} uses\nPadding changes border participation; it does not make all pixels equivalent.`));}s.oninput=run;run();
    }
    if(kind==='shrink'){
      const [l,s]=select('Number of valid 5×5 layers',Array.from({length:8},(_,i)=>[i,i]),4);host.append(l);const out=el('div');host.append(out);
      function run(){const ns=Array.from({length:+s.value+1},(_,i)=>32-4*i);out.replaceChildren(eq(ns.map(x=>x+'×'+x).join(' → ')),el('p','takeaway',ns.at(-1)===4?'The next 5×5 kernel does not fit. This sequence never reaches 1×1.':'Every layer removes four pixels from each spatial dimension.'));}s.oninput=run;run();
    }
    if(kind==='camera'){
      const [l,s]=select('Image resolution',[[32,'32×32 RGB'],[224,'224×224 RGB'],[108000000,'108 megapixels RGB']],224);host.append(l);const out=el('div');host.append(out);
      function run(){const n=+s.value,pixels=n===108000000?n:n*n,params=pixels*3*100+100;out.replaceChildren(eq(`${fmt(pixels)} pixels × 3 channels × 100 hidden units + 100 biases\n= ${fmt(params)} parameters\nFloat32 weights alone: ${(params*4/1e9).toFixed(4)} GB (decimal)`),el('p','note','Gradients, optimizer state, and activations require additional memory. Float32 is four bytes, not 32 bytes.'));}s.oninput=run;run();
    }
    if(kind==='mnist-curves'){
      const [l,s]=select('Show training through epoch',D.history.map(x=>[x.epoch,x.epoch]),10);host.append(l);const out=el('div');host.append(out);
      function run(){const end=+s.value,rows=D.history.slice(0,end+1),mx=2.4,points=key=>rows.map(x=>`${60+x.epoch*70},${270-x[key].loss/mx*240}`).join(' ');out.innerHTML=`<svg viewBox="0 0 820 330" class="learning-plot" role="img" aria-label="Measured MNIST train and validation cross-entropy"><path d="M60 20V270H790" fill="none" stroke="#9ca3af"/><polyline points="${points('train')}" stroke="#0f766e" stroke-width="4" fill="none"/><polyline points="${points('validation')}" stroke="#245edb" stroke-width="4" fill="none"/><text x="4" y="30">2.4</text><text x="25" y="274">0</text><text x="60" y="306">0</text><text x="680" y="306">10 epochs</text><text x="450" y="35" fill="#0f766e">Train</text><text x="570" y="35" fill="#245edb">Validation</text></svg>`;const h=rows.at(-1);out.append(eq(`Epoch ${end} · train loss ${h.train.loss.toFixed(3)} · validation loss ${h.validation.loss.toFixed(3)}\nValidation: ${h.validation.correct} / ${h.validation.total} correct`));}s.oninput=run;run();
    }
    if(kind==='pca'){
      const [l,s]=select('Representation',[["raw","784 raw pixels"],["learned","84 learned features"]],'raw');host.append(l);const out=el('div');host.append(out);const colours=['#245edb','#be123c','#0f766e','#a65310','#7345a5','#b43b16','#326875','#605920','#a53970','#565f70'];
      function run(){const pts=D.pca[s.value],xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);out.innerHTML='<svg viewBox="0 0 900 390" class="pca-plot" role="img" aria-label="PCA projection of 1000 held-out MNIST representations">'+pts.map((p,i)=>`<circle cx="${30+(p[0]-xmin)/(xmax-xmin)*760}" cy="${350-(p[1]-ymin)/(ymax-ymin)*320}" r="3" opacity=".65" fill="${colours[D.pca.labels[i]]}"><title>Test ${i}, label ${D.pca.labels[i]}</title></circle>`).join('')+colours.map((c,i)=>`<text x="830" y="${35+i*32}" fill="${c}">${i}</text>`).join('')+'</svg>';}s.oninput=run;run();
    }
    if(kind==='bottleneck'){
      const [l,s]=select('Intermediate width',[[8,8],[16,16],[32,32],[64,64]],16);host.append(l);const out=el('div');host.append(out);
      function run(){const b=+s.value,rows=[['1×1 reduce',`64 → ${b}`,64*b],['3×3 spatial',`${b} → ${b}`,9*b*b],['1×1 expand',`${b} → 128`,b*128]],weights=rows.reduce((a,r)=>a+r[2],0);out.replaceChildren(table(['Operation','Channels','Weights'],rows.map(r=>[r[0],r[1],fmt(r[2])])),eq(`28² × ${fmt(weights)} = ${fmt(weights*784)} MACs\nStandard 3×3: 57,802,752 MACs · reduction ${(57802752/(weights*784)).toFixed(2)}×`));}s.oninput=run;run();
    }
    if(kind==='residual-number'){
      const label=el('label','','Residual branch coefficient a'),s=el('input');s.type='range';s.min=-1;s.max=1;s.step=.1;s.value=.1;label.append(s);host.append(label);const out=el('div');host.append(out);function run(){const a=+s.value;out.replaceChildren(eq(`x = 2 · F(x) = ${a.toFixed(1)} x = ${(2*a).toFixed(1)}\ny = x + F(x) = ${(2+2*a).toFixed(1)}\ndy/dx = 1 + ${a.toFixed(1)} = ${(1+a).toFixed(1)}`),el('p','note','At a = 0 the block is the identity. At a = −1 the two derivative paths cancel; a shortcut is not a universal gradient guarantee.'));}s.oninput=run;run();
    }
    if(kind==='photo-lab'){
      const controls=el('div','controls'),[a,scene]=select('Original ML tutorial image',[["beach","Beach"],["buildings","Buildings"]],'buildings'),[b,ch]=select('Input plane',[["rgb","RGB"],["gray","Grayscale"],[0,"Red"],[1,"Green"],[2,"Blue"]],'gray'),[c,k]=select('Filter',[["vertical","Vertical edges"],["horizontal","Horizontal edges"],["blur","Mean blur"],["sharpen","Sharpen"]],'vertical');controls.append(a,b,c);host.append(controls);const out=el('div','photo-lab-result');host.append(out);let source=null;
      function run(){if(!source)return;out.replaceChildren();const w=192,h=128,canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;const ctx=canvas.getContext('2d');ctx.drawImage(source,0,0,w,h);const raw=ctx.getImageData(0,0,w,h),channel=ch.value;
        const xs=M.grid(h,w,(r,c)=>{const i=(r*w+c)*4;return channel==='gray'||channel==='rgb'?(.299*raw.data[i]+.587*raw.data[i+1]+.114*raw.data[i+2])/255:raw.data[i+(+channel)]/255;});
        const kernel=k.value==='sharpen'?[[0,-1,0],[-1,5,-1],[0,-1,0]]:M.kernels[k.value],ys=M.conv(xs,kernel,{p:1});
        const pair=el('div','photo-pair');if(channel==='rgb'){canvas.style.width='330px';canvas.style.height='220px';canvas.setAttribute('aria-label','Original colour tutorial photograph');pair.append(canvas);}else{const a=heat(xs.flat(),w,{size:330,lightHigh:true,label:channel==='gray'?'Grayscale input':['Red','Green','Blue'][+channel],max:1});$('canvas',a).style.height='220px';pair.append(a);}
        const result=heat(ys.flat(),w,{signed:!['blur','sharpen'].includes(k.value),size:330,lightHigh:['blur','sharpen'].includes(k.value),label:'Computed '+k.options[k.selectedIndex].text,max:['blur','sharpen'].includes(k.value)?1:3});$('canvas',result).style.height='220px';pair.append(result);out.append(pair,eq(kernel.map(row=>row.map(v=>fmt(v,2)).join('  ')).join('\n')),el('p','note','Same original ML tutorial photographs, resized to 192×128 for this calculation. Cross-correlation, zero padding. RGB preview filters its grayscale conversion.'));
      }
      function load(){source=null;out.textContent='Loading course photograph…';const img=new Image();img.onload=()=>{source=img;run();};img.onerror=()=>out.textContent='Image could not load; check the bundled figures/ml-course folder.';img.src=`figures/ml-course/${scene.value}.jpg`;}
      scene.oninput=load;ch.oninput=k.oninput=run;load();
    }
  });
  renderAll();
  window.CNNWorkshops={state,trace,renderAll};
})();
