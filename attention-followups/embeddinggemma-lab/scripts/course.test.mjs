import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { rank } from '../src/math.js';
import { distinctResults } from '../src/knowledge.js';
import { REVISION } from '../src/config.js';
const read = f => JSON.parse(fs.readFileSync(new URL('../public/'+f,import.meta.url)));
const items=read('course-corpus.json'),data=read('course-embeddings.json'),questions=read('course-questions.json'),queries=read('course-query-embeddings.json');
assert.equal(data.revision,REVISION);assert.equal(queries.revision,REVISION);
assert.equal(new Set(items.map(x=>x.id)).size,items.length);
for(const item of items){
 const record=data.items[item.id];assert(record,`Missing ${item.id}`);
 assert.equal(record.vector.length,768);assert(record.vector.every(Number.isFinite));assert(Math.abs(Math.hypot(...record.vector)-1)<1e-5);
 assert.equal(record.info.text,`title: ${item.title} | text: ${item.text}`,`Stale text: ${item.id}`);
 assert.equal(crypto.createHash('sha256').update(item.text).digest('hex'),item.contentHash);
 assert(/^https:\/\/(nipunbatra.github.io|github.com)\//.test(item.source));
 if(item.page){assert(item.source.endsWith('#page='+item.page));assert(fs.existsSync(new URL('../public/'+item.thumbnail,import.meta.url)));}
 if(item.group==='code')assert(item.cell>0 && item.colab.startsWith('https://colab.research.google.com/'));
}
const checks=[/dropout/i,/square root|scaling|variance/i,/loss/i,/EOT|attention mask/i,/activations|activation|nonlinear/i,/chain rule|backprop/i,/gradient|training step|autograd/i,/contrastive|loss|CLIP/i,/normaliz|unit|encode/i,/dropout/i,/finite.difference|numerical/i,/patch|position/i];
const report=[];
questions.forEach((q,i)=>{
 const vector=queries.items[q.id];assert.equal(vector.info.text,`task: ${q.task} | query: ${q.text}`);
 const results=distinctResults(rank(vector.vector,items.filter(x=>x.group===q.group),data.items));
 assert(results.slice(0,3).some(x=>checks[i].test(x.item.title+' '+x.item.text)),q.text);
 report.push({question:q.text,top:results.map(x=>({score:x.score,id:x.item.id,title:x.item.title,source:x.item.source}))});
});
fs.mkdirSync('output/verification',{recursive:true});fs.writeFileSync('output/verification/course-retrieval.json',JSON.stringify(report,null,2));
console.log(`Passed: ${items.length} genuine course vectors, exact encoded text, source links, PDF thumbnails, and 12 retrieval spot checks. These spot checks are not a retrieval benchmark.`);
