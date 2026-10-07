import { rank, unit } from './math.js';
import { REVISION } from './config.js';
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let loading;
const localItems = [], localVectors = {};
async function loadIndex() {
  if (!loading) loading = Promise.all(['course-corpus.json', 'course-embeddings.json', 'course-questions.json', 'course-query-embeddings.json'].map(async file => {
    const response = await fetch(`./${file}`);
    if (!response.ok) throw Error('The course index could not load. Check your connection and try again.');
    return response.json();
  })).then(([items, embeddings, questions, queries]) => {
    if (embeddings.revision !== REVISION || queries.revision !== REVISION) throw Error('The course index uses a different model revision. Please refresh.');
    return {items, vectors: embeddings.items, questions, queries: queries.items};
  }).catch(e => { loading = null; throw e; });
  return loading;
}
export function sourceLabel(item) {
  return item.page ? `PDF page ${item.page}` : item.cell ? `Notebook cell ${item.cell}` : item.anchor ? 'HTML slide' : 'Your local note';
}
// Limit repeated animation builds of the same explanation, without changing scores.
export function distinctResults(results, limit = 5) {
  const seen = [];
  const words = text => new Set(text.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []);
  const chosen = [];
  for (const result of results) {
    const tokens = words(result.item.text);
    const duplicate = seen.some(old => {
      if (old.item.source && old.item.source === result.item.source && old.item.cell === result.item.cell) return true;
      if (old.item.lecture !== result.item.lecture) return false;
      const intersection = [...tokens].filter(x => old.tokens.has(x)).length;
      return intersection / Math.max(1, Math.min(tokens.size, old.tokens.size)) > .88;
    });
    if (!duplicate) { seen.push({...result, tokens}); chosen.push(result); }
    if (chosen.length === limit) break;
  }
  return chosen;
}
export function mountKnowledge(root, kind, services) {
  const code = kind === 'code', group = code ? 'code' : 'document';
  let alive = true, data, activeQuery, lastRows = [], working = false;
  const $ = s => root.querySelector(s);
  root.innerHTML = `<div class="knowledge"><p class="knowledge-count" role="status">Loading the course index…</p>
  <form id="knowledge-form"><label for="knowledge-query">${code ? 'Describe the code you need' : 'What would you like to understand?'}</label><textarea id="knowledge-query" maxlength="2000" rows="2" placeholder="${code ? 'e.g. compute gradients and update model weights' : 'e.g. Why does dropout help with overfitting?'}"></textarea><div class="knowledge-ideas"></div><div class="knowledge-actions"><button class="primary" id="knowledge-search" disabled>${code ? 'Find code' : 'Find slide excerpts'}</button><label>${code ? 'Notebook' : 'Lecture'}<select id="knowledge-filter"><option value="all">All ${code ? 'notebooks' : 'lectures'}</option></select></label><label>Vector size<select id="knowledge-dimension"><option>768</option><option>512</option><option>256</option><option>128</option></select></label></div><p class="hint">Example questions use saved query embeddings. Your own question runs locally with WebGPU.${code?" Open the notebook for imports and earlier cells.":""}</p></form>
  <p class="knowledge-status" role="status" aria-live="polite">The results will be excerpts from the original ${code ? 'notebooks' : 'slides'}.</p><p class="knowledge-error" role="alert" hidden></p>
  <section class="knowledge-results" aria-label="${code ? 'Code results' : 'Cited slide excerpts'}"></section>
  <details class="knowledge-how"><summary>How does this search work?</summary><div class="knowledge-flow"><span><b>1. Prepare once</b>Extract slide text or notebook cells.<br>Encode each passage and keep its source link.</span><span><b>2. Ask now</b>Encode your question with the same model.<br>No need to encode the course again.</span><span><b>3. Compare & read</b>Compare unit vectors by dot product.<br>Open the matching original source.</span></div><p>Here the index is a small collection of vectors. The browser can compare them all directly; a database server is not needed. Only changed passages need new embeddings when the course changes.</p><p>EmbeddingGemma retrieves excerpts. It does not write an answer or check that a passage answers your question. Diagrams and equation layouts remain in the linked slides; the searchable index uses extracted text.</p><p class="hint">Results use cosine similarity in the chosen dimension, then near-duplicate builds of a slide are skipped. Scores are not probabilities. There is no reliable automatic “I don’t know” threshold in this demo.</p></details>
  <details class="knowledge-add"><summary>Try indexing your own ${code ? 'code snippet' : 'passage'} right now</summary><p>Paste a small ${code ? 'function' : 'note'}, encode it once, then ask a question about it. It joins this tab’s search collection. Nothing is uploaded or saved after a reload.</p><label>Title<input id="knowledge-local-title" maxlength="100" placeholder="My ${code ? 'function' : 'note'}"></label><label>${code ? 'Code' : 'Passage'}<textarea id="knowledge-local-text" maxlength="6000" rows="5"></textarea></label><button class="quiet" id="knowledge-add" disabled>Encode and add to this tab</button><button class="quiet" id="knowledge-clear">Remove my local items</button><p id="knowledge-local-count" class="hint"></p></details></div>`;
  function message(text) { if (alive) $('.knowledge-status').textContent = text; }
  function setWorking(value) {
    working = value;
    root.setAttribute('aria-busy', String(value));
    root.querySelectorAll('button,input,select,textarea').forEach(n => n.disabled = value);
    services.busy(value);
  }
  const task = code ? 'code retrieval' : 'question answering';
  function candidates() { return [...data.items, ...localItems].filter(x => x.group === group && ($('#knowledge-filter').value === 'all' || x.lecture === $('#knowledge-filter').value)); }
  function vectorDetails(item, score) {
    const dimensions = +$('#knowledge-dimension').value, record = ({...data.vectors, ...localVectors})[item.id];
    const v = unit(record.vector, dimensions), q = unit(activeQuery.vector, dimensions);
    return `<details class="knowledge-vectors"><summary>Inspect the embeddings and score</summary><p>Question [${dimensions}] · passage [${dimensions}] → one cosine: <b>${score.toFixed(6)}</b></p><p class="hint">Both vectors are normalized. The score sums q[k] × v[k] over all ${dimensions} coordinates.</p><div class="knowledge-vector-grid"><details><summary>Full query vector</summary><pre>${esc(JSON.stringify(q.map(x => +x.toFixed(6))))}</pre></details><details><summary>Full passage vector</summary><pre>${esc(JSON.stringify(v.map(x => +x.toFixed(6))))}</pre></details></div><details><summary>Exact model inputs</summary><pre>${esc(activeQuery.info?.text || '')}\n\n${esc(record.info?.text || '')}</pre></details></details>`;
  }
  function nearby(item) {
    if (item.group !== 'document' || !item.source) return '';
    const ordered=data.items.filter(x=>x.lecture===item.lecture && x.part===1), index=ordered.findIndex(x=>x.source===item.source);
    if(index<0)return '';
    const neighbours=[ordered[index-1], ordered[index+1]].filter(Boolean);
    return neighbours.length ? `<details class="knowledge-nearby"><summary>Read the neighbouring slides</summary>${neighbours.map(x=>`<p><a href="${esc(x.source)}" target="_blank" rel="noopener">${esc(x.title)} ↗</a></p><blockquote>${esc(x.text.slice(0,500))}${x.text.length>500?'…':''}</blockquote>`).join('')}</details>` : '';
  }
  function preview(text) {
    const limit=code?650:550;
    const lines=text.split('\n');
    const part=lines.slice(0,code?14:12).join('\n').slice(0,limit);
    return part;
  }
  function showResults() {
    if (!activeQuery || !alive) return;
    const pool = candidates();
    lastRows = distinctResults(rank(activeQuery.vector, pool, {...data.vectors, ...localVectors}, +$('#knowledge-dimension').value));
    $('.knowledge-results').innerHTML = `<div class="knowledge-results-head"><h3>${code ? 'From the course notebooks' : 'From your course slides'}</h3><span>${lastRows.length} matches · ${pool.length.toLocaleString()} candidates</span></div>` + (lastRows.length ? lastRows.map(({item, score}, i) => `<article class="knowledge-result" data-source-id="${esc(item.id)}"><div class="knowledge-result-top"><span class="knowledge-rank">${String(i + 1).padStart(2,'0')}</span><div><p class="eyebrow">${esc(item.lecture)} · ${esc(sourceLabel(item))}</p><h3>${esc(item.title.replace(item.lecture + ' · ', '').replace(/ [VDI] (VISUAL|DERIVATION|INTERACTIVE)$/, ''))}</h3></div><span class="knowledge-score" title="Cosine similarity, not a probability">${score.toFixed(3)}<small>cosine</small></span></div><div class="knowledge-excerpt ${code ? 'is-code' : ''}">${item.thumbnail && !code ? `<a href="${esc(item.source)}" target="_blank" rel="noopener"><img src="${esc(item.thumbnail)}" alt="Preview of the source slide" loading="lazy"></a>` : ''}<div><${code ? 'pre' : 'blockquote'}>${esc(preview(item.text)+(preview(item.text).length<item.text.length?'…':''))}</${code ? 'pre' : 'blockquote'}>${preview(item.text).length<item.text.length ? `<details><summary>Read the complete ${code ? 'code chunk' : 'extracted passage'}</summary><pre>${esc(item.text)}</pre></details>` : ''}</div></div><div class="knowledge-source-links">${item.source ? `<a href="${esc(item.source)}" target="_blank" rel="noopener">${code ? 'Open notebook on GitHub' : 'Open original slide'} ↗</a>` : '<span>Local item · this tab only</span>'}${item.colab ? `<a href="${esc(item.colab)}" target="_blank" rel="noopener">Open in Colab ↗</a>` : ''}${!code && item.sourceKind === 'html' ? `<small>Includes diagrams and surrounding slides</small>` : ''}</div>${nearby(item)}${vectorDetails(item, score)}</article>`).join('') : '<p>No passages in this selection. Choose another lecture or add a local item.</p>');
    root.querySelectorAll('.knowledge-excerpt img').forEach(img => img.onerror = () => img.closest('a').hidden = true);
    message(`${activeQuery.origin} · ${activeQuery.question}. Read the excerpts and check the source.`);
  }
  async function search(event) {
    event?.preventDefault(); if (!data || working) return;
    const text = $('#knowledge-query').value.trim(); if (!text) { $('#knowledge-query').focus(); return; }
    $('.knowledge-error').hidden = true; setWorking(true); message('Encoding your question…');
    try {
      const saved = data.questions.find(x => x.group === group && x.text === text);
      const record = saved && data.queries[saved.id];
      const result = record || await services.embed({id: 'course-query',type:'text',title:'Your question',text}, {role:'query',task});
      activeQuery = {...result, question:text, origin:record ? 'Saved example query · same pinned WebGPU model' : 'Question encoded on this device'};
      showResults();
    } catch(e) { if(alive) { $('.knowledge-error').textContent=e.message; $('.knowledge-error').hidden=false; message('Search could not finish. The previous results remain below.'); } }
    finally { if (alive) setWorking(false); }
  }
  $('#knowledge-form').onsubmit = search;
  $('#knowledge-filter').onchange = showResults;
  $('#knowledge-dimension').onchange = showResults;
  function localCount() { $('#knowledge-local-count').textContent=`${localItems.filter(x=>x.group===group).length} local ${code?'code snippets':'passages'} in this tab.`; }
  $('#knowledge-add').onclick = async () => {
    const text=$('#knowledge-local-text').value.trim(),title=$('#knowledge-local-title').value.trim() || `My ${code?'function':'note'}`;
    if (!text || working) return;
    $('.knowledge-error').hidden=true; setWorking(true); message('Encoding your new passage once…');
    try {
      const item={id:'local-'+crypto.randomUUID(),type:'text',group,title,text,lecture:'Your local items'};
      const result=await services.embed(item,{role:'document',task});
      localItems.push(item);localVectors[item.id]=result;
      if (!Array.from($('#knowledge-filter').options).some(x=>x.value===item.lecture)) $('#knowledge-filter').add(new Option(item.lecture,item.lecture));
      localCount();message('Added. Ask a question above to search the course and your new item.');$('#knowledge-local-text').value='';
    } catch(e){if(alive){$('.knowledge-error').textContent=e.message;$('.knowledge-error').hidden=false;}}
    finally{if(alive)setWorking(false);}
  };
  $('#knowledge-clear').onclick=()=>{
    for(let i=localItems.length-1;i>=0;i--)if(localItems[i].group===group){delete localVectors[localItems[i].id];localItems.splice(i,1);}
    if($('#knowledge-filter').value==='Your local items')$('#knowledge-filter').value='all';
    $('#knowledge-filter').querySelector('option[value="Your local items"]')?.remove();localCount();showResults();
  };
  loadIndex().then(result=>{
    if(!alive)return; data=result;
    const items=data.items.filter(x=>x.group===group), lectures=[...new Set(items.map(x=>x.lecture))];
    $('.knowledge-count').textContent=`${items.length.toLocaleString()} ${code?'code chunks':'slide passages'} · ${lectures.length} ${code?'notebooks':'lecture decks'} · original sources linked`;
    lectures.forEach(name=>$('#knowledge-filter').add(new Option(name,name)));
    if(localItems.some(x=>x.group===group))$('#knowledge-filter').add(new Option('Your local items','Your local items'));
    const examples=data.questions.filter(x=>x.group===group);
    $('.knowledge-ideas').innerHTML=examples.map(x=>`<button type="button" class="chip" data-question="${esc(x.id)}">${esc(x.text)}</button>`).join('');
    root.querySelectorAll('[data-question]').forEach(button=>button.onclick=()=>{$('#knowledge-query').value=examples.find(x=>x.id===button.dataset.question).text;search();});
    $('#knowledge-query').value=examples[0]?.text || '';$('#knowledge-search').disabled=false;$('#knowledge-add').disabled=false;localCount();
    message('Choose an example to search instantly, or type your own question.');
  }).catch(e=>{if(alive){$('.knowledge-count').textContent='Course index unavailable';$('.knowledge-error').hidden=false;$('.knowledge-error').textContent=e.message;}});
  return ()=>{alive=false;root.removeAttribute('aria-busy');root.innerHTML='';};
}
