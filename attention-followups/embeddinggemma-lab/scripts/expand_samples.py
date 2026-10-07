"""Add Commons photographs and ESC-10 examples; preserve the teaching split.
Requires requests; credits are stored per item and in MORE-MEDIA-CREDITS.md.
"""
from pathlib import Path
import requests,csv,io,json,re,html,hashlib,time,concurrent.futures
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'public';M=P/'media';W=ROOT/'work/more-media';W.mkdir(parents=True,exist_ok=True)
items={i['id']:i for i in json.loads((P/'gallery.json').read_text())};session=requests.Session();session.headers['User-Agent']='EmbeddingGemmaCourseLab/1.0 (https://nipunbatra.github.io)'
def get(url,**kw):
 for attempt in range(3):
  r=session.get(url,timeout=60,**kw)
  if r.ok:return r
  time.sleep(1+attempt)
 r.raise_for_status()
def plain(t):return html.unescape(re.sub('<[^>]+>','',t)).strip()
def add(x):
 if x.get('src'):x['sha256']=hashlib.sha256((P/x['src']).read_bytes()).hexdigest()
 items[x['id']]=x
base='https://raw.githubusercontent.com/karolpiczak/ESC-50/master/'
rows=list(csv.DictReader(io.StringIO(get(base+'meta/esc50.csv').text)))
license=(P/'ESC-50-LICENSE.txt').read_text();categories=sorted({r['category']for r in rows if r['esc10']=='True'})
existing={Path(i.get('src','')).name for i in items.values() if not i['id'].startswith('esc-extra-')}
additions=[]
for cat in categories:
 selected=[r for r in rows if r['category']==cat and r['esc10']=='True' and r['filename']not in existing][:6]
 for r in selected:
  filename=r['filename'];dest=M/filename
  additions.append((r,dest))
def sound(pair):
 r,dest=pair
 if not dest.exists():dest.write_bytes(get(base+'audio/'+r['filename']).content)
 return r,dest
for r,dest in concurrent.futures.ThreadPoolExecutor(4).map(sound,additions):
 key='-'.join(r['filename'].split('-')[:-1])+'.ogg';attr=next((l for l in license.splitlines()if '['+key+']'in l),'ESC-10 · CC BY 3.0; see complete license')
 add(dict(id='esc-extra-'+dest.stem,type='audio',title=r['category'].replace('_',' ').capitalize()+' · example '+dest.stem,src='media/'+dest.name,duration=5,group='sound',label=r['category'],split='explore',sourceRecording=r['src_file'],credit='ESC-10 / Karol J. Piczak. '+attr,license='CC BY 3.0',source='https://github.com/karolpiczak/ESC-50'))
print('Added',len(additions),'recordings',flush=True)
# These searches choose photographs; each exact file's author and license are preserved.
queries=['banana fruit','elephant wild','tiger animal','snow mountain','sunflower flower','laptop computer','violin instrument','sailboat sea','airplane aircraft','fire engine','watermelon fruit','butterfly flower','owl bird','wind turbine','tractor field','bowl soup','bicycle street','snowman snow','lighthouse coast','basketball ball']
credits=[]
for q in queries:
 cache=W/(q.replace(' ','-')+'.json')
 if cache.exists():data=json.loads(cache.read_text())
 else:
  data=get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','generator':'search','gsrsearch':q+' filetype:bitmap','gsrnamespace':6,'gsrlimit':8,'prop':'imageinfo','iiprop':'url|extmetadata|size','iiurlwidth':640}).json();cache.write_text(json.dumps(data))
 pages=sorted(data.get('query',{}).get('pages',{}).values(),key=lambda x:x.get('index',0));count=0
 for page in pages:
  if count==2:break
  if not page.get('imageinfo'):continue
  info=page['imageinfo'][0];meta=info.get('extmetadata',{});lic=plain(meta.get('LicenseShortName',{}).get('value',''))
  if not (lic.startswith('CC BY')or lic in ['CC0','Public domain']):continue
  if info.get('width',0)<300 or info.get('height',0)<200:continue
  title=page['title'][5:];url=info.get('thumburl',info['url'])
  if not re.search(r'\.(jpg|jpeg|png)(?:$|\?)',url,re.I):continue
  id='commons-extra-'+str(page['pageid']);dest=M/(id+('.png'if '.png'in url.lower()else'.jpg'))
  try:
   if not dest.exists():dest.write_bytes(get(url).content)
  except requests.RequestException:continue
  author=plain(meta.get('Artist',{}).get('value','Wikimedia Commons contributor'));source=info['descriptionurl'];label=re.sub(r'\.(jpg|jpeg|png)$','',title,flags=re.I).replace('_',' ')
  description=plain(meta.get('ImageDescription',{}).get('value',''));caption=description if 20<len(description)<240 else label
  add(dict(id=id,type='image',title=label[:110],src='media/'+dest.name,group='photo',credit=author+' · '+lic+' · Commons thumbnail, resized.',source=source,license=lic,licenseURL=meta.get('LicenseUrl',{}).get('value',''),description=description[:500]))
  add(dict(id='caption-'+id,type='text',title='Caption: '+label[:90],text=caption,group='caption',credit='Caption from Wikimedia Commons file description/title; exact source linked.',source=source))
  credits.append(f'- [{label}]({source}) — {author}; {lic}. Resized thumbnail.');count+=1
 print(q,count,flush=True)
(P/'gallery.json').write_text(json.dumps(list(items.values()),ensure_ascii=False,indent=2)+'\n')
(P/'MORE-MEDIA-CREDITS.md').write_text('# More media examples\n\nAdditional audio comes from ESC-10 (CC BY 3.0). Full recording attribution is in ESC-50-LICENSE.txt and each sample. These examples are for exploration; they do not change the training/test split.\n\n## Photographs\n\n'+'\n'.join(credits)+'\n')
from collections import Counter
print('TOTAL',len(items),Counter(i['type']for i in items.values()),flush=True)
