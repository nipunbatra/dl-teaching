import assert from 'node:assert/strict';
import fs from 'node:fs';
import {chapters,chapterScores} from '../src/lesson-data.js';
import {unit,dot,difference} from '../src/math.js';
const gallery=JSON.parse(fs.readFileSync('public/gallery.json'));const vectors=JSON.parse(fs.readFileSync('public/embeddings.json')).items;
const original=JSON.stringify(vectors);
const reports=[];
for(const c of chapters.filter(c=>c.candidates)){
 for(const id of [c.a,c.b,...c.candidates,...(c.alternate||[])].filter(Boolean))assert.ok(gallery.some(x=>x.id===id)&&vectors[id],id);
 const scores=chapterScores(c,vectors);assert.ok(scores.every(x=>Number.isFinite(x.a)&&Math.abs(x.a)<=1.0000001));
 if(c.kind==='compare')for(const s of scores)assert.equal(s.b,dot(unit(vectors[c.b].vector),unit(vectors[s.id].vector)));
 if(c.kind==='delta'){
  const flipped=chapterScores(c,vectors,{reversed:true});
  const norm=Math.hypot(...vectors[c.b].vector.map((x,k)=>unit(vectors[c.b].vector)[k]-unit(vectors[c.a].vector)[k]));
  for(const s of scores){assert.ok(Math.abs(s.a+flipped.find(x=>x.id===s.id).a)<1e-12);const v=unit(vectors[s.id].vector);assert.ok(Math.abs(s.a-(dot(unit(vectors[c.b].vector),v)-dot(unit(vectors[c.a].vector),v))/norm)<1e-12);}
 }
 if(c.kind==='missing'){const removed=chapterScores(c,vectors,{removed:true});assert.equal(removed.length,scores.length-1);for(const s of removed)assert.equal(s.a,scores.find(x=>x.id===s.id).a);}
 if(c.kind==='dimensions')for(const s of scores)assert.ok(Math.abs(s.b-dot(unit(vectors[c.a].vector,128),unit(vectors[s.id].vector,128)))<1e-12);
 reports.push({chapter:c.id,scores});
}
assert.equal(JSON.stringify(vectors),original,'demonstrations must never mutate the saved index');
// A difference at d dimensions normalizes each truncated input before subtracting.
const shortDelta=difference(unit(vectors['portrait-hat'].vector,128),unit(vectors.portrait.vector,128));
assert.equal(shortDelta.length,128);assert.ok(Math.abs(Math.hypot(...shortDelta)-1)<1e-12);
fs.mkdirSync('output/verification',{recursive:true});fs.writeFileSync('output/verification/lesson-math.json',JSON.stringify(reports,null,2));
console.log('Passed: nine numeric chapters, candidate removal invariance, delta decomposition/sign reversal, dimension comparisons, immutable index.');
