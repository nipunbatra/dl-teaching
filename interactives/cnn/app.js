/* Shared article / classroom runtime and small, model-backed worksheets. */
(() => {
  'use strict';
  const M=window.CNN,$=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
  const f=(n,d=2)=>Math.abs(n)<.5*10**(-d)?'0':Number.isInteger(n)?String(n):n.toFixed(d),fmt=n=>n.toLocaleString('en-IN');
  const clamp=(n,a,b)=>Math.min(b,Math.max(a,n));
  $('.topbar').innerHTML='<div class="topbar-inner"><a class="brand" href="index.html">Convolutional neural networks</a><nav aria-label="Lecture navigation"><a href="sources.html">Sources</a></nav><button id="contents">Contents</button><button id="present">Present <kbd>P</kbd></button></div>';
  $('.page-end').innerHTML='<a href="../../slides.html">← Course lectures</a><a href="sources.html">Sources & teaching notes</a><a href="#question">Back to the opening ↑</a>';
  function node(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e;}
  function matrix(x,{title='',selected=()=>false,sampled=()=>false,padded=()=>false,click,gray=false,scale=3,cell=42}={}) {
    const wrap=node('div','matrix-wrap'),cap=node('div','cap',title),g=node('div','matrix');
    g.style.setProperty('--cols',x[0]?.length||1);g.style.setProperty('--cell',cell+'px');
    x.forEach((row,r)=>row.forEach((v,c)=>{
      const e=node(click?'button':'span','cell',f(v));
      const strength=Math.min(1,Math.abs(v)/scale);
      e.style.background=gray?`rgb(${250-Number(v)*175},${251-Number(v)*165},${254-Number(v)*135})`:`color-mix(in srgb, ${v<0?'#245edb':'#be123c'} ${Math.round(strength*28)}%, white)`;
      if(gray&&v>.5)e.style.color='white';
      if(selected(r,c))e.classList.add('selected');if(sampled(r,c))e.classList.add('sampled');if(padded(r,c))e.classList.add('pad');
      if(click){e.type='button';e.setAttribute('aria-label',`${title}: row ${r}, column ${c}, value ${f(v)}`);e.onclick=()=>click(r,c);}
      else e.setAttribute('title',`[${r}, ${c}] = ${f(v)}`);
      g.append(e);
    }));wrap.append(cap,g);return wrap;
  }
  function mount(el,controls='') {el.innerHTML=`<div class="controls">${controls}</div><div class="result"></div>`;return $('.result',el);}
  const select=(id,label,options)=>`<label>${label}<select id="${id}">${options.map(o=>`<option value="${o[0]}">${o[1]}</option>`).join('')}</select></label>`;
  const slider=(id,label,min,max,value,step=1)=>`<label for="${id}">${label} <output id="${id}-value">${value}</output><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}"></label>`;
  function bind(el,render){$$('input,select',el).forEach(e=>e.addEventListener('input',()=>{const o=$('#'+e.id+'-value',el);if(o)o.textContent=e.value;render();}));render();}
  function stats(items){const box=node('div','stats');items.forEach(([value,label])=>{const b=node('div','stat-block');b.append(node('div','stat',value),node('div','stat-label',label));box.append(b);});return box;}
  function equation(text){return node('div','equation',text);}
  function mats(...items){const box=node('div','matrices');items.forEach(x=>box.append(typeof x==='string'?node('span','operator',x):x));return box;}

  const widgets={
    hero(el){el.append(mats(matrix(M.base,{title:'A 5 × 5 image',gray:true,cell:36}),'→',matrix(M.conv(M.base,M.kernels.horizontal),{title:'One response map',cell:36})));el.append(node('p','note','Nine shared weights. Nine computed responses.'));},
    photo(el){
      const out=mount(el,select('photo-k','Filter',[['horizontal','Horizontal contrast'],['vertical','Vertical contrast'],['blur','3 × 3 average'],['identity','Identity']]));
      const pair=node('div','photo-pair'),a=document.createElement('canvas'),b=document.createElement('canvas');a.width=b.width=160;a.height=b.height=160;
      a.setAttribute('aria-label','Grayscale pet photograph');b.setAttribute('aria-label','Computed filter response');
      const left=node('div','matrix-wrap'),right=node('div','matrix-wrap'),caption=node('p','note');left.append(node('div','cap','Grayscale input'),a);right.append(node('div','cap','Computed response'),b);pair.append(left,right);out.append(pair,caption);
      let pixels=null;const img=new Image();
      function render(){if(!pixels)return;const key=$('#photo-k').value,k=M.kernels[key],y=M.conv(pixels,k,{p:1}),ctx=b.getContext('2d'),im=ctx.createImageData(160,160);for(let r=0;r<160;r++)for(let c=0;c<160;c++){const i=(r*160+c)*4,v=y[r][c];if(key==='blur'||key==='identity'){const g=clamp(v,0,1)*255;im.data[i]=im.data[i+1]=im.data[i+2]=g;}else{const t=clamp(Math.abs(v)/3,0,1),color=v<0?[36,94,219]:[190,18,60];for(let q=0;q<3;q++)im.data[i+q]=255*(1-t)+color[q]*t;}im.data[i+3]=255;}ctx.putImageData(im,0,0);caption.textContent=(key==='blur'||key==='identity')?'Grayscale output, fixed range [0, 1]. Zero padding, stride 1.':'Signed output, fixed range [−3, +3]. Blue < 0, white = 0, red > 0. Zero padding, stride 1.';}
      img.onload=()=>{try{const ctx=a.getContext('2d');ctx.drawImage(img,0,0,160,160);const im=ctx.getImageData(0,0,160,160);pixels=M.grid(160,160,(r,c)=>{const i=(r*160+c)*4,v=(.299*im.data[i]+.587*im.data[i+1]+.114*im.data[i+2])/255;im.data[i]=im.data[i+1]=im.data[i+2]=v*255;return v;});ctx.putImageData(im,0,0);render();}catch(e){caption.textContent='Serve this folder over localhost to enable the photograph calculator. See README for the command.';}};
      img.onerror=()=>caption.textContent='Photograph could not load. Check that the figures folder is present.';
      img.src='figures/head-roi-crop.png';$('#photo-k').onchange=render;
    },
    convolution(el){let x=M.clone(M.mlEdge),r=0,c=0;
      const out=mount(el,select('conv-example','Course example',[['ml','ML notebook · edge'],['dl','DL L8 · band image']])+select('conv-k','Shared kernel',[['horizontal','Horizontal contrast'],['vertical','Vertical contrast'],['identity','Identity']])+'<button id="conv-prev" aria-label="Previous output">← Patch</button><button id="conv-next">Next patch →</button><button id="conv-reset">Reset pixels</button>');
      function render(){const k=M.kernels[$('#conv-k').value],y=M.conv(x,k);out.replaceChildren(mats(matrix(x,{title:'Input X · click to edit',gray:true,sampled:(i,j)=>i>=r&&i<r+3&&j>=c&&j<c+3,click:(i,j)=>{x[i][j]=1-x[i][j];render();$$('.matrix button',out)[i*x[0].length+j].focus();}}),'×',matrix(k,{title:'Shared kernel K',scale:1}),'→',matrix(y,{title:'Output Y · choose a cell',selected:(i,j)=>i===r&&j===c,click:(i,j)=>{r=i;c=j;render();}})));
        const terms=k.flatMap((row,i)=>row.map((v,j)=>`${x[r+i][c+j]}×(${v})`));out.append(equation(`Y[${r},${c}] = ${terms.join(' + ')} = ${f(y[r][c])}`));
        $('#conv-prev').disabled=r===0&&c===0;$('#conv-next').disabled=r===y.length-1&&c===y[0].length-1;
      }
      const reset=()=>{x=M.clone($('#conv-example').value==='ml'?M.mlEdge:M.base);r=c=0;render();};
      $('#conv-prev').onclick=()=>{const w=x[0].length-2,n=r*w+c-1;r=Math.floor(n/w);c=n%w;render();};$('#conv-next').onclick=()=>{const w=x[0].length-2,n=r*w+c+1;r=Math.floor(n/w);c=n%w;render();};$('#conv-reset').onclick=reset;
      $('#conv-example').oninput=()=>{$('#conv-k').value=$('#conv-example').value==='ml'?'vertical':'horizontal';reset();};$('#conv-k').oninput=render;$('#conv-k').value='vertical';render();
    },
    geometry(el){let r=0,c=0;
      const out=mount(el,select('geo-p','Zero padding p',[[0,'0'],[1,'1'],[2,'2']])+select('geo-s','Stride s',[[1,'1'],[2,'2']])+select('geo-d','Dilation d',[[1,'1'],[2,'2']]));
      function render(){const p=+$('#geo-p').value,s=+$('#geo-s').value,d=+$('#geo-d').value,k=M.kernels.horizontal,y=M.conv(M.base,k,{p,s,d}),n=y.length;r=Math.min(r,n-1);c=Math.min(c,n-1);
        const padded=M.grid(5+2*p,5+2*p,(i,j)=>M.base[i-p]?.[j-p]??0),sampled=(i,j)=>{const a=i-r*s,b=j-c*s;return a>=0&&b>=0&&a<=2*d&&b<=2*d&&a%d===0&&b%d===0;};
        out.replaceChildren(mats(matrix(padded,{title:`Input + padding · ${5+2*p} × ${5+2*p}`,gray:true,cell:32,sampled,padded:(i,j)=>i<p||j<p||i>=p+5||j>=p+5}),'→',matrix(y,{title:`Output · ${n} × ${n}`,cell:36,selected:(i,j)=>i===r&&j===c,click:(i,j)=>{r=i;c=j;render();}})));
        out.append(equation(`Kernel span = ${d*(3-1)+1} · output = floor((5 + 2×${p} − ${d*(3-1)+1}) / ${s}) + 1 = ${n} · selected Y[${r},${c}] = ${y[r][c]}`));
      }bind(el,render);
    },
    channels(el){const out=mount(el,slider('rgb-r','Red kernel scale',-2,2,1,.5)+slider('rgb-g','Green kernel scale',-2,2,1,.5)+slider('rgb-b','Blue kernel scale',-2,2,1,.5));
      const xs=[[[1,0],[0,1]],[[0,1],[1,0]],[[1,1],[0,0]]],ks=[[[1,0],[0,1]],[[0,-1],[-1,0]],[[.5,.5],[.5,.5]]];
      function render(){const rows=node('div','matrices'),con=[];['r','g','b'].forEach((name,q)=>{const a=+$('#rgb-'+name).value,k=ks[q].map(row=>row.map(v=>v*a)),v=M.conv(xs[q],k)[0][0];con.push(v);const col=node('div','matrix-wrap');col.append(mats(matrix(xs[q],{title:['R input','G input','B input'][q],gray:true,cell:32}),'·',matrix(k,{title:'Kernel slice',scale:2,cell:32})),node('p','note',`Contribution: ${f(v)}`));rows.append(col);});out.replaceChildren(rows,equation(`${con.map(v=>`(${f(v)})`).join(' + ')} + bias 0.5 = ${f(M.sum(con)+.5)} → one output value`));}bind(el,render);
    },
    pooling(el){const out=mount(el,select('pool-mode','Pooling operation',[['max','Max'],['mean','Average']])+select('pool-relu','Apply ReLU first?',[[1,'Yes'],[0,'No']]));const x=[[-3,-1,2,0],[-2,-4,1,3],[1,2,-1,-2],[0,4,-3,-4]];
      function render(){const relu=+$('#pool-relu').value,z=x.map(row=>row.map(v=>relu?Math.max(0,v):v)),y=M.pool(z,$('#pool-mode').value);out.replaceChildren(mats(matrix(x,{title:'Signed feature map',scale:4}),relu?'ReLU →':'→',matrix(z,{title:'Pool input',scale:4,sampled:(r,c)=>r<2&&c<2}),'→',matrix(y,{title:'2 × 2 output',scale:4})));out.append(equation(`Top-left window [${z[0][0]}, ${z[0][1]}, ${z[1][0]}, ${z[1][1]}] → ${f(y[0][0])}`));}bind(el,render);
    },
    shifts(el){const out=mount(el,select('shift-s','Stride',[[1,'1'],[2,'2']])+select('shift-dx','Input shift right',[[1,'1 pixel'],[2,'2 pixels']])+select('shift-boundary','Boundary',[['circular','Circular'],['zero','Zero']]));const x=M.grid(6,6,(r,c)=>(r===2&&c===0)||(r===4&&c===3)?1:0),k=[[0,0,0],[1,2,1],[0,0,0]];
      function render(){const s=+$('#shift-s').value,dx=+$('#shift-dx').value,circular=$('#shift-boundary').value==='circular',a=M.conv(M.shift(x,dx,0,circular),k,{p:1,s,circular}),b=M.shift(M.conv(x,k,{p:1,s,circular}),Math.floor(dx/s),0,circular),err=Math.max(...a.flatMap((row,r)=>row.map((v,c)=>Math.abs(v-b[r][c]))));out.replaceChildren(mats(matrix(x,{title:'Input X',gray:true,cell:30}),'→',matrix(a,{title:'f(shift(X))',cell:34}),dx%s?'?':'vs',matrix(b,{title:`shift(f(X), ${Math.floor(dx/s)})`,cell:34})));out.append(equation(`Maximum absolute difference = ${f(err)}${dx%s?' · input shift / stride = 0.5 output pixels; displayed candidate shift = 0':' · aligned output shift = '+dx/s}`));out.append(node('p','note','Fixed 3 × 3 kernel: center row [1, 2, 1], all other weights zero. Both paths use the selected boundary rule.'));}bind(el,render);
    },
    receptive(el){const presets={course:[{k:3},{k:2,s:2},{k:3}],stack:[{k:3},{k:3}],stride:[{k:3,s:2},{k:3}],dilated:[{k:3,d:2}],holes:[{k:3,d:2},{k:3,d:2}],point:[{k:3},{k:1}]};
      const out=mount(el,select('rf-mode','Layer stack',[['course','DL L8 · conv → pool → conv'],['stack','3×3 → 3×3'],['stride','3×3 stride 2 → 3×3'],['dilated','3×3 dilation 2'],['holes','Two 3×3, both dilation 2'],['point','3×3 → 1×1']]));
      function render(){const layers=presets[$('#rf-mode').value],trace=M.receptive(layers),last=trace.at(-1);let support=new Set([0]),jump=1;layers.forEach(({k,s=1,d=1})=>{const next=new Set();support.forEach(a=>{for(let i=0;i<k;i++)next.add(a+i*d*jump);});support=next;jump*=s;});const centered=new Set([...support].map(v=>v+Math.floor((13-last.r)/2)));
        const picture=matrix(M.grid(13,13,()=>0),{title:'Possible input support · one interior output',cell:24,sampled:(r,c)=>centered.has(r)&&centered.has(c),scale:1});$$('.cell',picture).forEach(e=>{e.textContent='';if(e.classList.contains('sampled'))e.style.background='#c4e5df';});
        const info=node('div');info.append(stats([[`${last.r} × ${last.r}`,'Bounding span'],[`${support.size**2}`,'Possible input pixels'],[`${last.j}`,'Output jump']]));info.append(equation('rₗ = rₗ₋₁ + dₗ(kₗ−1) jₗ₋₁;  jₗ = jₗ₋₁ sₗ'));info.append(node('p','note',`Start r₀=1, j₀=1. ${trace.map((v,i)=>`After layer ${i+1}: r=${v.r}, j=${v.j}`).join(' → ')}.`));const cols=node('div','columns');cols.append(picture,info);out.replaceChildren(cols);
      }bind(el,render);
    },
    cost(el){const opts=[8,16,32,64].map(x=>[x,x]),out=mount(el,select('cost-n','Image width / height',[[16,16],[32,32],[64,64],[128,128]])+select('cost-ci','Input channels',[[3,3],...opts])+select('cost-co','Output channels',opts));$('#cost-n').value=32;$('#cost-co').value=16;
      function render(){const n=+$('#cost-n').value,ci=+$('#cost-ci').value,co=+$('#cost-co').value,a=M.ledger(n,ci,co);out.replaceChildren(stats([[fmt(a.params),'Learned parameters'],[fmt(a.macs),'MACs per image'],[fmt(a.activations),'Output values'],[f(a.activations*4/1024)+' KiB','Output storage · float32']]));out.append(equation(`W: [${co}, ${ci}, 3, 3] · Y: [${co}, ${n}, ${n}]\nParameters = ${co} × (9 × ${ci} + 1)\nMACs = ${n}² × ${co} × 9 × ${ci}`));}bind(el,render);
    },
    lenet(el){const out=mount(el,select('lenet-n','Input size · ML course examples',[[32,'Slides · 32 × 32'],[28,'Notebook · 28 × 28']]));
      function render(){const n=+$('#lenet-n').value,a=M.lenet(n);out.innerHTML='<div class="table-wrap"><table><thead><tr><th>Layer / block</th><th>Output shape</th><th>Parameters</th></tr></thead><tbody>'+a.rows.map(([label,shape,params])=>`<tr><td>${label}</td><td>${shape}</td><td class="num">${fmt(params)}</td></tr>`).join('')+`<tr><td><strong>Total</strong></td><td>10 logits</td><td class="num"><strong>${fmt(a.total)}</strong></td></tr></tbody></table></div>`;}bind(el,render);
    },
    gradient(el){const out=mount(el,select('grad-use','Highlight spatial contribution',[[0,'Output y₀'],[1,'Output y₁'],[2,'Sum both uses']]));
      function render(){const x=[2,1,3],w=[.5,-1],g=[1,2],a=M.gradient1d(x,w,g),i=+$('#grad-use').value;const rows=a.y.map((v,r)=>`<tr ${i===r?'style="background:#e4ecff"':''}><td>y${r} = ${v}</td><td>[${x[r]}, ${x[r+1]}]</td><td>${g[r]}</td><td>[${g[r]*x[r]}, ${g[r]*x[r+1]}]</td></tr>`).join('');out.innerHTML=`<div class="table-wrap"><table><thead><tr><th>Output</th><th>Patch</th><th>∂L/∂y</th><th>Contribution to ∂L/∂w</th></tr></thead><tbody>${rows}<tr ${i===2?'style="background:#d9f2ef"':''}><td>Sum</td><td></td><td></td><td>[${a.dw.join(', ')}]</td></tr></tbody></table></div>`;out.append(equation(`∂L/∂x = [${a.dx.join(', ')}] · ∂L/∂b = ${a.db}${i<2?' · highlighted kernel contribution ['+g[i]*x[i]+', '+g[i]*x[i+1]+']':''}`));}bind(el,render);
    },
    training(el){trainingEl=el;const out=mount(el,'<button class="primary" id="train-100">Train 100 steps</button><button id="train-500">Train 500 steps</button><button id="train-reset">Reset model</button>'+select('train-sample','Inspect example',M.samples.map((s,i)=>[i,`${s.y?'Vertical':'Horizontal'} · position ${s.pos}`])));trainingOut=out;
      $('#train-100').onclick=()=>train(100);$('#train-500').onclick=()=>train(500);$('#train-reset').onclick=()=>{model=M.init();steps=0;history=[{step:0,loss:M.objective(model).loss}];renderTraining();renderForward();renderClassifier();};$('#train-sample').onchange=renderTraining;renderTraining();
    },
    forward(el){forwardEl=el;forwardOut=mount(el,select('forward-sample','Example',M.samples.map((s,i)=>[i,`${s.y?'Vertical':'Horizontal'} · position ${s.pos}`]))+select('forward-filter','Inspect filter',[[0,'0'],[1,'1']]));$$('select',el).forEach(e=>e.onchange=()=>{if($('#classifier-sample'))$('#classifier-sample').value=$('#forward-sample').value;renderForward();renderClassifier();});renderForward();},
    classifier(el){classifierOut=mount(el,select('classifier-sample','Example · shared with forward trace',M.samples.map((s,i)=>[i,`${s.y?'Vertical':'Horizontal'} · position ${s.pos}`])));$('#classifier-sample').onchange=()=>{$('#forward-sample').value=$('#classifier-sample').value;renderForward();renderClassifier();};renderClassifier();}
  };
  let model=M.init(),steps=0,history=[{step:0,loss:M.objective(model).loss}],trainingEl,trainingOut,forwardEl,forwardOut,classifierOut;
  function train(n){for(let i=0;i<n;i++){const a=M.step(model);steps++;if(steps%10===0)history.push({step:steps,loss:a.loss});}renderTraining();renderForward();renderClassifier();}
  function renderTraining(){if(!trainingOut)return;const a=M.objective(model),sample=M.samples[+$('#train-sample').value],p=M.forward(model,sample.x).prob;trainingOut.replaceChildren();
    const lower=node('div','columns'),left=node('div'),right=node('div');left.append(stats([[steps,'Gradient steps'],[a.loss.toFixed(4),'Mean cross-entropy'],[`${Math.round(a.accuracy*6)} / 6`,'Training correct']]));const max=Math.max(.75,...history.map(h=>h.loss)),points=history.map(h=>`${35+330*h.step/Math.max(1,steps)},${125-105*h.loss/max}`).join(' ');left.insertAdjacentHTML('beforeend',`<svg class="plot" viewBox="0 0 390 155" role="img" aria-label="Training loss history"><path d="M35 15V125H370" stroke="#aeb7c4" fill="none"/><polyline points="${points}" fill="none" stroke="#0f766e" stroke-width="2.5"/><text class="chart-axis" x="2" y="20">${max.toFixed(2)}</text><text class="chart-axis" x="18" y="128">0</text><text class="chart-axis" x="35" y="149">0</text><text class="chart-axis" x="240" y="149">${steps} steps</text></svg>`);
    right.append(mats(matrix(sample.x,{title:'Training image',gray:true,cell:29}),matrix(model.k[0],{title:'Kernel 0',scale:2,cell:42}),matrix(model.k[1],{title:'Kernel 1',scale:2,cell:42})));
    ['Horizontal','Vertical'].forEach((name,i)=>right.insertAdjacentHTML('beforeend',`<div class="bar-row"><span>${name}</span><div class="bar-track"><div class="bar-fill" style="width:${100*p[i]}%"></div></div><output>${(100*p[i]).toFixed(1)}%</output></div>`));right.append(node('p','note',`True label: ${sample.y?'vertical':'horizontal'}. Probabilities use the current weights.`));lower.append(right,left);trainingOut.append(lower);
  }
  function renderForward(){if(!forwardOut)return;const sample=M.samples[+$('#forward-sample').value],q=+$('#forward-filter').value,a=M.forward(model,sample.x);forwardOut.replaceChildren(mats(matrix(sample.x,{title:'Input',gray:true,cell:29}),'→',matrix(a.z[q],{title:`Conv channel ${q}`,scale:2,cell:42}),'→',matrix(a.h[q],{title:'After ReLU',scale:2,cell:42}),'→',matrix([[a.g[q]]],{title:'Spatial mean',scale:2,cell:45})));
    forwardOut.append(equation(`Pooled vector g = [${a.g.map(v=>v.toFixed(3)).join(', ')}] · model at step ${steps}`));
    forwardOut.append(node('p','note',`Channel ${q} bias = ${model.b[q].toFixed(3)}. Global average = sum of its nine ReLU responses / 9. Full-precision calculations; labels rounded.`));
  }
  function renderClassifier(){if(!classifierOut)return;const sample=M.samples[+$('#classifier-sample').value],a=M.forward(model,sample.x);
    classifierOut.innerHTML='<div class="table-wrap"><table><thead><tr><th>Class logit</th><th>Feature 0 contribution</th><th>Feature 1 contribution</th><th>+ bias</th><th>= logit</th></tr></thead><tbody>'+model.w.map((w,i)=>`<tr><td>${i?'Vertical':'Horizontal'}</td><td>${w[0].toFixed(3)} × ${a.g[0].toFixed(3)}</td><td>${w[1].toFixed(3)} × ${a.g[1].toFixed(3)}</td><td>${model.a[i].toFixed(3)}</td><td>${a.logits[i].toFixed(3)}</td></tr>`).join('')+'</tbody></table></div>';
    classifierOut.append(equation(`p(class i) = exp(logitᵢ) / Σⱼ exp(logitⱼ)\nProbabilities = [${a.prob.map(v=>v.toFixed(4)).join(', ')}] · true label: ${sample.y?'vertical':'horizontal'}\nCross-entropy for this image = −log p(true class) = ${(-Math.log(a.prob[sample.y])).toFixed(4)}`));
  }
  $$('[data-widget]').forEach(el=>widgets[el.dataset.widget]?.(el));
  // One set of live widgets in reading mode and a fixed 16:9 classroom stage.
  const frames=$$('.frame');let current=0,present=false,build=0;
  const controls=node('div');controls.id='present-controls';controls.innerHTML='<button id="previous" aria-label="Previous slide">←</button><span id="counter"></span><button id="next" aria-label="Next slide">→</button><button id="overview">Overview O</button><button id="notes">Notes S</button><button id="read">Read Esc</button>';document.body.append(controls);
  const overview=document.createElement('dialog'),notes=document.createElement('dialog');
  const chapters=[...new Set(frames.map(f=>f.dataset.chapter))];
  overview.innerHTML='<h2>Lecture overview</h2>'+chapters.map(ch=>'<div class="overview-chapter"><h3>'+ch+'</h3><ol>'+frames.map((e,i)=>e.dataset.chapter===ch?`<li value="${i+1}"><a href="#${e.id}" data-frame="${i}">${e.dataset.title}</a></li>`:'').join('')+'</ol></div>').join('')+'<button>Close</button>';
  document.body.append(overview,notes);$('button',overview).onclick=()=>overview.close();$$('a',overview).forEach(a=>a.onclick=e=>{e.preventDefault();overview.close();show(+a.dataset.frame);if(!present)frames[current].scrollIntoView();});
  function scale(){document.documentElement.style.setProperty('--scale',Math.min(innerWidth/1280,innerHeight/720));}
  const maxBuild=()=>Math.max(0,...$$('[data-build]',frames[current]).map(e=>+e.dataset.build));
  function paintBuild(hash=true){
    $$('[data-build]',frames[current]).forEach(e=>{const pending=present&&+e.dataset.build>build;e.classList.toggle('is-pending',pending);e.inert=pending;if(pending)e.setAttribute('aria-hidden','true');else e.removeAttribute('aria-hidden');});
    $('#counter').textContent=`${current+1} / ${frames.length}${maxBuild()?' · '+build+'/'+maxBuild():''}`;
    $('#previous').disabled=current===0&&build===0;$('#next').disabled=current===frames.length-1&&build===maxBuild();
    if(hash)window.history.replaceState(null,'','#'+frames[current].id+(build?'/'+build:''));
  }
  function show(i,hash=true,step=0){current=clamp(i,0,frames.length-1);build=Math.min(step,maxBuild());frames.forEach((e,j)=>e.classList.toggle('live',j===current));paintBuild(hash);}
  function advance(){if(build<maxBuild()){build++;paintBuild();}else show(current+1);}
  function retreat(){if(build>0){build--;paintBuild();}else{show(current-1);build=maxBuild();paintBuild();}}
  function setPresent(v){present=v;document.body.classList.toggle('present',v);if(v){scale();show(current,true,build);}else{$$('[data-build]').forEach(e=>{e.classList.remove('is-pending');e.inert=false;e.removeAttribute('aria-hidden');});frames[current].scrollIntoView({behavior:'instant'});}const u=new URL(location.href);v?u.searchParams.set('present',''):u.searchParams.delete('present');window.history.replaceState(null,'',u);}
  function openNotes(){const source=$('.notes',frames[current]);notes.innerHTML='<h2>Teaching notes</h2><p>'+(source?.textContent||'')+'</p><p class=small>Source: '+frames[current].dataset.source+'</p><button>Close</button>';$('button',notes).onclick=()=>notes.close();notes.showModal();}
  $('#present').onclick=()=>setPresent(true);$('#previous').onclick=retreat;$('#next').onclick=advance;$('#read').onclick=()=>setPresent(false);$('#overview').onclick=()=>overview.showModal();$('#contents').onclick=()=>overview.showModal();$('#notes').onclick=openNotes;
  const fromHash=()=>{const [id,step]=location.hash.slice(1).split('/');const i=frames.findIndex(e=>e.id===id);if(i>=0)show(i,false,+step||0);};fromHash();addEventListener('hashchange',fromHash);addEventListener('resize',scale);
  if('IntersectionObserver' in window){const io=new IntersectionObserver(entries=>{if(!present)for(const e of entries)if(e.isIntersecting)current=frames.indexOf(e.target);},{rootMargin:'-20% 0px -60% 0px'});frames.forEach(e=>io.observe(e));}
  addEventListener('keydown',e=>{if(overview.open||notes.open)return;const form=/INPUT|SELECT|TEXTAREA/.test(e.target.tagName),key=e.key.toLowerCase();if(e.altKey||e.ctrlKey||e.metaKey)return;if(e.target.tagName==='BUTTON'&&key===' ')return;if(form){if(key==='escape'){e.target.blur();e.preventDefault();}else if(key==='n'&&present){advance();e.preventDefault();}return;}if(key==='p'){setPresent(!present);e.preventDefault();}else if(key==='o'){overview.showModal();e.preventDefault();}else if(key==='s'&&present){openNotes();e.preventDefault();}else if(present&&(key==='arrowright'||key==='n'||key==='pagedown'||key===' ')){advance();e.preventDefault();}else if(present&&(key==='arrowleft'||key==='pageup')){retreat();e.preventDefault();}else if(key==='escape'&&present)setPresent(false);});
  if(new URLSearchParams(location.search).has('present'))setPresent(true);
  window.CNNLesson={show,setPresent,advance,retreat,revealAll(){build=maxBuild();paintBuild();},get build(){return build;},get model(){return model;},get steps(){return steps;},frames};
})();
