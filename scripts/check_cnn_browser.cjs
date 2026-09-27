const assert=require('node:assert/strict');
const http=require('node:http');
const fs=require('node:fs');
const path=require('node:path');
const puppeteer=require('puppeteer');
const root=path.resolve(__dirname,'..'),out=path.join(root,'tmp/cnn-review');
fs.mkdirSync(out,{recursive:true});
const types={'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.json':'application/json','.py':'text/plain'};
const server=http.createServer((req,res)=>{
  const p=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));
  if(!p.startsWith(root+path.sep)){res.writeHead(403).end();return;}
  fs.readFile(p,(err,data)=>{if(err){res.writeHead(404).end();return;}res.setHeader('Content-Type',types[path.extname(p)]||'text/plain');res.end(data);});
});
let browser;
(async()=>{
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const url=`http://127.0.0.1:${server.address().port}/interactives/cnn/`;
  const chrome='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
  browser=await puppeteer.launch({headless:true,...(fs.existsSync(chrome)?{executablePath:chrome}:{}),timeout:20000});
  const page=await browser.newPage(),errors=[],failed=[],overflows=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('response',r=>{if(r.status()>=400&&!r.url().endsWith('favicon.ico'))failed.push(r.url());});
  await page.setViewport({width:1440,height:900});
  await page.goto(url+'index.html?present',{waitUntil:'networkidle0'});
  const frames=await page.$$eval('.frame',xs=>xs.map(x=>x.id));
  assert.equal(frames.length,31);assert.equal(new Set(frames).size,31);
  assert.equal(await page.$$eval('h1',xs=>xs.length),1);
  assert.equal(await page.$$eval('a',xs=>xs.filter(a=>/part[123]\.html/.test(a.getAttribute('href'))).length),0);
  async function show(id){await page.evaluate(id=>{CNNLesson.setPresent(true);CNNLesson.show(CNNLesson.frames.findIndex(f=>f.id===id));},id);}
  async function inspect(state){
    const clipped=await page.evaluate(()=>{
      const f=document.querySelector('.frame.live'),box=f.getBoundingClientRect();
      return [...f.children].filter(e=>getComputedStyle(e).display!=='none').filter(e=>{
        const r=e.getBoundingClientRect();return r.bottom>box.bottom-10||r.top<box.top+10||r.right>box.right+1||r.left<box.left-1;
      }).map(e=>({tag:e.tagName,class:e.className,top:e.getBoundingClientRect().top,bottom:e.getBoundingClientRect().bottom,frameBottom:box.bottom}));
    });
    if(clipped.length)overflows.push({state,clipped});
  }
  for(const id of frames){
    await show(id);
    await page.evaluate(()=>document.querySelectorAll('.frame.live details').forEach(d=>d.open=true));
    await inspect(id);
    if(['question','convolution','channels','receptive','lenet','gradient','trace'].includes(id))await page.screenshot({path:path.join(out,`unified-${id}.png`)});
  }
  await page.evaluate(()=>CNNLesson.setPresent(false));
  const before=await page.$eval('[data-widget=convolution] .equation',e=>e.textContent);
  await page.click('#conv-next');assert.notEqual(await page.$eval('[data-widget=convolution] .equation',e=>e.textContent),before);
  await page.click('[data-widget=convolution] .matrix button');await page.click('#conv-reset');
  assert.equal(await page.$eval('[data-widget=convolution] .equation',e=>e.textContent),before);
  await page.select('#conv-example','ml');assert.equal(await page.$eval('#conv-k',e=>e.value),'vertical');
  assert.equal(await page.$$eval('[data-widget=convolution] .matrix button',es=>es.length),52);
  await page.click('#conv-next');assert.match(await page.$eval('[data-widget=convolution] .equation',e=>e.textContent),/= 3$/);
  await show('convolution');await inspect('ML 6×6 edge');await page.screenshot({path:path.join(out,'unified-ml-edge.png')});
  await page.evaluate(()=>CNNLesson.setPresent(false));
  await page.select('#geo-d','2');assert.match(await page.$eval('[data-widget=geometry] .equation',e=>e.textContent),/Kernel span = 5/);
  await page.select('#geo-p','2');await page.select('#geo-s','2');
  const responseBefore=await page.$eval('[data-widget=photo] .photo-pair > div:last-child canvas',e=>e.toDataURL());
  await page.select('#photo-k','vertical');
  assert.notEqual(await page.$eval('[data-widget=photo] .photo-pair > div:last-child canvas',e=>e.toDataURL()),responseBefore);
  await page.select('#shift-s','2');assert.match(await page.$eval('[data-widget=shifts] .equation',e=>e.textContent),/0.5 output pixels/);
  await page.select('#shift-dx','2');assert.match(await page.$eval('[data-widget=shifts] .equation',e=>e.textContent),/difference = 0/);
  assert.equal(await page.$$eval('[data-widget=receptive] .cell.sampled',xs=>xs.length),64);
  await page.select('#rf-mode','holes');assert.match(await page.$eval('[data-widget=receptive]',e=>e.textContent),/9 × 9/);
  assert.equal(await page.$$eval('[data-widget=receptive] .cell.sampled',xs=>xs.length),25);
  await page.select('#lenet-n','28');assert.match(await page.$eval('[data-widget=lenet]',e=>e.textContent),/44,426/);
  await page.select('#lenet-n','32');assert.match(await page.$eval('[data-widget=lenet]',e=>e.textContent),/61,706/);
  await page.select('#grad-use','2');assert.match(await page.$eval('[data-widget=gradient]',e=>e.textContent),/\[4, 7\]/);
  assert.match(await page.$eval('[data-widget=gradient] .equation',e=>e.textContent),/\[0.5, 0, -2\]/);
  await page.click('#train-100');await page.click('#train-500');
  assert.equal(await page.evaluate(()=>CNNLesson.steps),600);
  assert.ok(await page.evaluate(()=>CNN.objective(CNNLesson.model).loss)<.003);
  assert.match(await page.$eval('[data-widget=forward] .equation',e=>e.textContent),/step 600/);
  await page.select('#forward-sample','4');assert.equal(await page.$eval('#classifier-sample',e=>e.value),'4');
  await show('question');await show('train');assert.equal(await page.evaluate(()=>CNNLesson.steps),600);
  await inspect('Trained model');await page.screenshot({path:path.join(out,'unified-trained.png')});
  await page.evaluate(()=>CNNLesson.setPresent(false));await page.click('#train-reset');assert.equal(await page.evaluate(()=>CNNLesson.steps),0);
  await page.setViewport({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'mobile horizontal overflow');
  await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.join(out,'unified-mobile.png')});
  await page.setViewport({width:1440,height:900});
  await page.goto(url+'index.html?present#channels',{waitUntil:'networkidle0'});
  assert.equal(await page.$eval('.frame.live',e=>e.id),'channels');
  await page.focus('#rgb-r');await page.keyboard.press('ArrowRight');assert.equal(await page.$eval('.frame.live',e=>e.id),'channels');
  await page.keyboard.press('n');assert.equal(await page.$eval('.frame.live',e=>e.id),'tensor');
  await page.evaluate(()=>document.activeElement.blur());
  await page.keyboard.press('s');assert.equal(await page.$eval('dialog:last-of-type',e=>e.open),true);await page.keyboard.press('Escape');
  await page.keyboard.press('o');assert.equal(await page.$eval('dialog:first-of-type',e=>e.open),true);await page.keyboard.press('Escape');
  await page.keyboard.press('Escape');assert.equal(await page.evaluate(()=>document.body.classList.contains('present')),false);
  await page.click('#contents');assert.equal(await page.$eval('dialog:first-of-type',e=>e.open),true);await page.keyboard.press('Escape');
  for(const [part,id] of [[1,'question'],[2,'channels'],[3,'gradient']]){
    await page.goto(url+`part${part}.html?present`,{waitUntil:'networkidle0'});
    assert.equal(new URL(page.url()).pathname,'/interactives/cnn/index.html');assert.equal(await page.$eval('.frame.live',e=>e.id),id);
  }
  await page.goto(url+'sources.html',{waitUntil:'networkidle0'});
  assert.deepEqual(errors,[]);assert.deepEqual(failed,[]);
  const report={passed:!overflows.length,frames:frames.length,canonical:'interactives/cnn/index.html',browserErrors:errors,failedRequests:failed,overflows,mobileWidth:390};
  fs.writeFileSync(path.join(out,'browser-report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
  assert.deepEqual(overflows,[]);
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();server.close();});
