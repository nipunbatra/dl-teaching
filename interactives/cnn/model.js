/* Numerical source of truth for the CNN lessons. No DOM or dependencies. */
(function (root) {
  'use strict';
  const grid = (h, w, f = () => 0) => Array.from({length:h}, (_,r) => Array.from({length:w}, (_,c) => f(r,c)));
  const clone = x => JSON.parse(JSON.stringify(x));
  const sum = x => x.reduce((a,b) => a+b, 0);
  const base = [[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1],[1,1,1,1,1],[0,0,0,0,0]];
  // Exact binary edge input from ml-teaching/notebooks/cnn-edge.ipynb.
  const mlEdge = grid(6,6,(_,c)=>Number(c<3));
  const kernels = {
    horizontal: [[1,1,1],[0,0,0],[-1,-1,-1]],
    vertical: [[1,0,-1],[1,0,-1],[1,0,-1]],
    blur: grid(3,3,()=>1/9),
    identity: [[0,0,0],[0,1,0],[0,0,0]]
  };
  function size(n,k,p=0,s=1,d=1) { return Math.floor((n+2*p-d*(k-1)-1)/s)+1; }
  function conv(x,k,{p=0,s=1,d=1,b=0,circular=false}={}) {
    const h=x.length,w=x[0].length, oh=size(h,k.length,p,s,d),ow=size(w,k[0].length,p,s,d);
    if(oh<1||ow<1) return [];
    return grid(oh,ow,(r,c)=>{
      let y=b;
      k.forEach((row,i)=>row.forEach((v,j)=>{
        let rr=r*s+i*d-p,cc=c*s+j*d-p;
        if(circular){rr=(rr%h+h)%h;cc=(cc%w+w)%w;}
        y+=v*(x[rr]?.[cc]??0);
      }));
      return y;
    });
  }
  function shift(x,dx,dy=0,circular=false) {
    const h=x.length,w=x[0].length;
    return grid(h,w,(r,c)=>{
      let rr=r-dy,cc=c-dx;
      if(circular){rr=(rr%h+h)%h;cc=(cc%w+w)%w;}
      return x[rr]?.[cc]??0;
    });
  }
  function pool(x,mode='max',k=2,s=2) {
    return grid(size(x.length,k,0,s),size(x[0].length,k,0,s),(r,c)=>{
      const a=grid(k,k,(i,j)=>x[r*s+i][c*s+j]).flat();
      return mode==='max'?Math.max(...a):sum(a)/a.length;
    });
  }
  function receptive(layers) {
    let r=1,j=1,start=.5;
    return layers.map(({k,s=1,d=1,p=0})=>{
      const effective=d*(k-1)+1;
      start+=((effective-1)/2-p)*j;
      r+=(effective-1)*j;j*=s;
      return {r,j,start};
    });
  }
  function ledger(n,ci,co,k=3,p=1,s=1,d=1,g=1) {
    if(ci%g||co%g) throw Error('Channels must be divisible by groups');
    const out=size(n,k,p,s,d),weights=co*(ci/g)*k*k;
    return {out,weights,params:weights+co,macs:out*out*weights,activations:out*out*co};
  }
  // One-dimensional valid cross-correlation, squared loss, with overlapping patches.
  function gradient1d(x,w,upstream) {
    const y=Array.from({length:x.length-w.length+1},(_,i)=>sum(w.map((v,j)=>v*x[i+j])));
    const g=upstream??y;
    if(g.length!==y.length)throw Error('One upstream gradient is required per output');
    const dw=w.map((_,j)=>sum(g.map((v,i)=>v*x[i+j]))),dx=x.map(()=>0);
    g.forEach((v,i)=>w.forEach((a,j)=>dx[i+j]+=v*a));
    return upstream?{y,dw,dx,db:sum(g)}:{y,dw,dx,loss:sum(y.map(v=>v*v))/2};
  }
  function lenet(n=32) {
    const c1=size(n,5),p1=size(c1,2,0,2),c2=size(p1,5),p2=size(c2,2,0,2),flat=16*p2*p2;
    const rows=[['Conv 5×5, 1→6 → pool 2',`6 × ${p1} × ${p1}`,156],
      ['Conv 5×5, 6→16 → pool 2',`16 × ${p2} × ${p2}`,2416],
      ['Flatten',`${flat}`,0],['Linear → 120','120',flat*120+120],
      ['Linear → 84','84',120*84+84],['Linear → 10','10',84*10+10]];
    return {c1,p1,c2,p2,flat,rows,total:sum(rows.map(r=>r[2]))};
  }
  // Six deliberately tiny training examples: horizontal/vertical bars at 3 locations.
  const samples=[];
  for(let y=0;y<2;y++) for(let pos=1;pos<=3;pos++) samples.push({y,pos,x:grid(5,5,(r,c)=>(y===0?r===pos:c===pos)?1:0)});
  function init(seed=7) {
    const rand=()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed/4294967296;};
    return {k:Array.from({length:2},()=>grid(3,3,()=>rand()*.6-.3)),b:[.15,.15],w:grid(2,2,()=>rand()*.6-.3),a:[0,0]};
  }
  function forward(m,x) {
    const z=m.k.map((k,i)=>conv(x,k,{b:m.b[i]}));
    const h=z.map(a=>a.map(row=>row.map(v=>Math.max(0,v))));
    const g=h.map(a=>sum(a.flat())/9);
    const logits=m.w.map((row,i)=>sum(row.map((v,j)=>v*g[j]))+m.a[i]);
    const exp=logits.map(v=>Math.exp(v-Math.max(...logits))),den=sum(exp),prob=exp.map(v=>v/den);
    return {z,h,g,logits,prob};
  }
  function objective(m,data=samples,withGrad=true) {
    const grad={k:grid(2,1,()=>grid(3,3)).map(a=>a[0]),b:[0,0],w:grid(2,2),a:[0,0]};
    let loss=0,correct=0;
    for(const {x,y} of data) {
      const f=forward(m,x);loss-=Math.log(f.prob[y]);correct+=Number((f.prob[1]>f.prob[0]?1:0)===y);
      if(!withGrad) continue;
      const dl=f.prob.map((v,i)=>(v-Number(i===y))/data.length);
      dl.forEach((v,i)=>{grad.a[i]+=v;f.g.forEach((a,j)=>grad.w[i][j]+=v*a);});
      for(let q=0;q<2;q++) {
        const dg=sum(dl.map((v,i)=>v*m.w[i][q]));
        for(let r=0;r<3;r++) for(let c=0;c<3;c++) {
          const dz=f.z[q][r][c]>0?dg/9:0;
          grad.b[q]+=dz;
          for(let i=0;i<3;i++) for(let j=0;j<3;j++) grad.k[q][i][j]+=dz*x[r+i][c+j];
        }
      }
    }
    return {loss:loss/data.length,accuracy:correct/data.length,grad};
  }
  function step(m,lr=.4) {
    const {grad}=objective(m);
    for(let q=0;q<2;q++) {
      m.b[q]-=lr*grad.b[q];m.a[q]-=lr*grad.a[q];
      for(let i=0;i<2;i++) m.w[q][i]-=lr*grad.w[q][i];
      for(let i=0;i<3;i++) for(let j=0;j<3;j++) m.k[q][i][j]-=lr*grad.k[q][i][j];
    }
    return objective(m,samples,false);
  }
  const api={grid,clone,sum,base,mlEdge,kernels,size,conv,shift,pool,receptive,ledger,gradient1d,lenet,samples,init,forward,objective,step};
  if(typeof module!=='undefined') module.exports=api;
  if(root) root.CNN=api;
})(typeof window!=='undefined'?window:null);
