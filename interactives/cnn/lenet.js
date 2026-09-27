/* The ML course's LeNet forward pass, with every activation available. */
(function(root){
  'use strict';
  function conv(x,ci,n,w,b,co,k){
    const m=n-k+1,y=new Float64Array(co*m*m);
    for(let o=0;o<co;o++)for(let r=0;r<m;r++)for(let c=0;c<m;c++){
      let v=b[o];
      for(let q=0;q<ci;q++)for(let i=0;i<k;i++)for(let j=0;j<k;j++)
        v+=x[(q*n+r+i)*n+c+j]*w[((o*ci+q)*k+i)*k+j];
      y[(o*m+r)*m+c]=v;
    }
    return y;
  }
  const relu=x=>Float64Array.from(x,v=>Math.max(0,v));
  function pool(x,channels,n){
    const m=n/2,y=new Float64Array(channels*m*m);
    for(let q=0;q<channels;q++)for(let r=0;r<m;r++)for(let c=0;c<m;c++)
      y[(q*m+r)*m+c]=Math.max(x[(q*n+2*r)*n+2*c],x[(q*n+2*r)*n+2*c+1],x[(q*n+2*r+1)*n+2*c],x[(q*n+2*r+1)*n+2*c+1]);
    return y;
  }
  function linear(x,w,b){return Float64Array.from(b,(v,i)=>{for(let j=0;j<x.length;j++)v+=w[i*x.length+j]*x[j];return v;});}
  function forward(weights,pixels){
    const input=Float64Array.from(pixels,v=>v/255);
    const conv1=conv(input,1,28,weights['conv1.weight'],weights['conv1.bias'],6,5),relu1=relu(conv1),pool1=pool(relu1,6,24);
    const conv2=conv(pool1,6,12,weights['conv2.weight'],weights['conv2.bias'],16,5),relu2=relu(conv2),pool2=pool(relu2,16,8);
    const flat=pool2,hidden1=relu(linear(flat,weights['fc1.weight'],weights['fc1.bias']));
    const hidden2=relu(linear(hidden1,weights['fc2.weight'],weights['fc2.bias']));
    const logits=linear(hidden2,weights['fc3.weight'],weights['fc3.bias']);
    const mx=Math.max(...logits),exp=Array.from(logits,v=>Math.exp(v-mx)),den=exp.reduce((a,b)=>a+b,0),prob=exp.map(v=>v/den);
    return {input,conv1,relu1,pool1,conv2,relu2,pool2,flat,hidden1,hidden2,logits,prob};
  }
  const api={conv,relu,pool,linear,forward};
  if(typeof module!=='undefined')module.exports=api;
  if(root)root.CNNLeNet=api;
})(typeof window!=='undefined'?window:null);
