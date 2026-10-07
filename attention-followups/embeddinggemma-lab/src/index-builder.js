import { Engine, prepare } from './runtime.js';
import { REVISION } from './config.js';
const $=s=>document.querySelector(s);
const collection=new URLSearchParams(location.search).get('collection')||'gallery';
const course=collection==='course', questions=collection==='questions';
const [gallery,data]=await Promise.all([
 fetch(course?'./course-corpus.json':questions?'./course-questions.json':'./gallery.json').then(r=>r.json()),
 fetch(course?'./course-embeddings.json':questions?'./course-query-embeddings.json':'./embeddings.json').then(r=>r.json())
]);
if(data.revision!==REVISION)throw Error('Wrong model revision');
const db=await new Promise((resolve,reject)=>{const req=indexedDB.open('embeddinggemma-index-builder',1);req.onupgradeneeded=()=>req.result.createObjectStore('vectors');req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);});
const cacheKey=id=>`${REVISION}:${collection}:${id}`;
const transaction=(mode,work)=>new Promise((resolve,reject)=>{const tx=db.transaction('vectors',mode);work(tx.objectStore('vectors'));tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);});
const keys=new Set(gallery.map(x=>cacheKey(x.id)));
await new Promise((resolve,reject)=>{const req=db.transaction('vectors').objectStore('vectors').openCursor();req.onsuccess=()=>{const c=req.result;if(!c)return resolve();if(keys.has(c.key))data.items[c.key.split(':').at(-1)]=c.value;c.continue();};req.onerror=()=>reject(req.error);});
const pending=()=>gallery.filter(x=>!data.items[x.id] || (x.type==='text'&&data.items[x.id].info?.text!==(questions?`task: ${x.task} | query: ${x.text}`:`title: ${x.group==='caption'?'none':x.title} | text: ${x.text}`)));
$('#status').textContent=`${collection} · ${gallery.length} items · ${pending().length} missing`;
const engine=new Engine(p=>{if(p.status==='progress')$('#status').textContent=`Loading ${p.file} · ${Math.round(p.progress||0)}%`;});
let stopped=false;
$('#start').onclick=async()=>{
 $('#start').disabled=true;stopped=false;$('#stop').disabled=false;
 try {
  await engine.load();const missing=pending();let n=0;
  for(const item of missing){
   if(stopped)break;
   $('#status').textContent=`Encoding ${++n}/${missing.length}: ${item.title}`;
   const {input,info}=await prepare(item,item.task||'search result',questions?'query':'document');
   const result={...(await engine.embed(input)),info,contentHash:item.contentHash||item.sha256};
   data.items[item.id]=result;
   await transaction('readwrite',store=>store.put(result,cacheKey(item.id)));
   $('#log').textContent=(`${item.id} · ${result.vector.length} dimensions · ${Math.round(result.elapsed)} ms\n`+$('#log').textContent).slice(0,16000);
  }
  data.date=new Date().toISOString();$('#status').textContent=`${stopped?'Paused':'Complete'} · ${gallery.length-pending().length}/${gallery.length} real embeddings · ${pending().length} missing`;
 }catch(e){$('#status').textContent=(stopped?'Paused: ':'ERROR: ')+e.message;}
 finally{$('#start').disabled=false;$('#stop').disabled=true;engine.stop();}
};
$('#stop').onclick=()=>{stopped=true;engine.stop();};
$('#download').onclick=()=>{
 const out={...data,date:new Date().toISOString(),items:Object.fromEntries(gallery.filter(i=>data.items[i.id]).map(i=>[i.id,data.items[i.id]]))};
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out)],{type:'application/json'}));a.download=course?'course-embeddings.json':questions?'course-query-embeddings.json':'expanded-embeddings.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
};
