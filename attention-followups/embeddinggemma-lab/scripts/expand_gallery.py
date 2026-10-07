"""Add attributed photographs, real videos, sounds and multilingual captions.
Run after prepare_gallery.py and prepare_video.py. Requires requests and ffmpeg.
Original downloads stay in work/; redistributed media retain their own licenses.
"""
import csv,io,json,pathlib,re,hashlib,subprocess,time,html
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1];P=ROOT/'public';M=P/'media';W=ROOT/'work/media-sources';W.mkdir(parents=True,exist_ok=True)
s=requests.Session();s.headers['User-Agent']='EmbeddingGemmaTeachingLab/1.0 (https://nipunbatra.github.io)'
def get(url,**kwargs):
 for attempt in range(4):
  r=s.get(url,timeout=90,**kwargs)
  if r.ok:return r
  time.sleep(2+attempt*2)
 r.raise_for_status()
def run(args):subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y']+args,check=True)
def clean(v):return html.unescape(re.sub('<[^>]*>','',v)).strip()
items={x['id']:x for x in json.loads((P/'gallery.json').read_text())}
def add(x):
 if x.get('src'):x['sha256']=hashlib.sha256((P/x['src']).read_bytes()).hexdigest()
 items[x['id']]=x
photos=[('fire','Campfire and sparks in Anttoora 3.jpg','Sparks above a campfire','A campfire burning at night with orange sparks.'),('waterfall','Phu Sang Waterfall 01.jpg','Waterfall in a forest','A waterfall flowing down a rocky slope in a green forest.'),('bicycle','Parked bicycle with graffitied building facade and doors in Amsterdam.jpg','Bicycle by a painted wall','A bicycle parked against a wall with graffiti.'),('train','UBTZ 2TE116UM-022 Cagaan Had - Sumangijn Zoo.jpg','Train on the railway','A locomotive pulling a train through open countryside.'),('beach','Rügen, Beach at Sellin -- 2009 -- 1173.jpg','Beach at Sellin','A sandy beach beside the sea.'),('guitar','Guitar May 2009-1.jpg','Acoustic guitar','An acoustic guitar with strings and a wooden body.'),('pizza','Vegetarian Pizza.jpg','Vegetarian pizza','A pizza covered with vegetables and cheese.'),('rooster','A 95 year old woman with her pet rooster, Havana, Cuba.jpg','Woman with her rooster','A woman holding a rooster in Havana.')]
videos=[('puppy','Puppy playing.webm','Puppy playing indoors','A golden retriever puppy playing inside a house.',0),('coffee','Coffee machine.ogv','Coffee machine at work','Coffee pouring from a coffee machine into a metal cup.',4),('waves','Water Waves Greece.ogv','Waves in Greece','Gentle waves moving across blue water.',0),('geese','Geese-1280669, Dingle Peninsula, Co. Kerry, Ireland.webm','Geese by the water','Geese moving beside the water in Ireland.',0)]
titles=['File:'+x[1] for x in photos+videos]
cache=W/'commons-metadata.json'
if cache.exists():pages=json.loads(cache.read_text())
else:
 pages=get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','titles':'|'.join(titles),'prop':'imageinfo','iiprop':'url|extmetadata|size','iiurlwidth':640}).json()['query']['pages'];cache.write_text(json.dumps(pages,indent=2))
meta={x['title'][5:]:x['imageinfo'][0] for x in pages.values() if 'imageinfo'in x}
credits=[]
for kind,entries in [('image',photos),('video',videos)]:
 for entry in entries:
  key,name,title,caption=entry[:4];x=meta[name];e=x['extmetadata'];author=clean(e.get('Artist',{}).get('value',''));license=clean(e['LicenseShortName']['value']);licenseURL=e.get('LicenseUrl',{}).get('value', 'https://creativecommons.org/publicdomain/mark/1.0/')
  if not (license.startswith('CC BY') or license=='Public domain'):raise ValueError(license)
  credit=f'{author} · {license}. '+('Resized to 640 px.' if kind=='image' else '12-second excerpt; resized, silent H.264 conversion.');source=x['descriptionurl']
  original=W/name
  if not original.exists():original.write_bytes(get(x.get('thumburl',x['url']) if kind=='image' else x['url']).content)
  dest=M/f'commons-{key}.{ "jpg" if kind=="image" else "mp4"}'
  if not dest.exists():
   if kind=='image':run(['-i',str(original),'-vf','scale=640:-2','-frames:v','1','-q:v','3',str(dest)])
   else:run(['-ss',str(entry[4]),'-i',str(original),'-t','12','-vf','scale=640:-2','-r','15','-an','-c:v','libx264','-crf','26','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)])
  common=dict(type=kind,title=title,src='media/'+dest.name,credit=credit,source=source,license=license,licenseURL=licenseURL)
  if kind=='image':add(dict(id='photo-'+key,**common))
  else:
   duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(dest)]))
   add(dict(id='film-'+key,**common,start=0,end=duration,duration=duration,group='video'))
   for a,b in [(0,4),(4,8)]:add(dict(common,id=f'moment-{key}-{a}',title=f'{title} · {a}–{b} s',start=a,end=min(b,duration),duration=min(b,duration)-a,group='moment',parent='film-'+key))
   frame=M/f'frame-{key}.jpg'
   if not frame.exists():run(['-ss','2','-i',str(dest),'-frames:v','1','-q:v','3',str(frame)])
   add(dict(common,id='frame-'+key,type='image',title=title+' · still frame',src='media/'+frame.name,credit=credit+' Still extracted at 2 s.',group='frame'))
  add(dict(id='caption-commons-'+key,type='text',title='Caption: '+title,text=caption,group='caption',credit='Course teaching caption'))
  credits.append(f'## {title}\n\n- Source: [{name}]({source})\n- Author: {author}\n- License: [{license}]({licenseURL})\n- Changes: {credit}\n')
  print('Prepared',title,flush=True)
base='https://raw.githubusercontent.com/karolpiczak/ESC-50/master/'
rows=list(csv.DictReader(io.StringIO(get(base+'meta/esc50.csv').text)));license=(P/'ESC-50-LICENSE.txt').read_text()
categories=['dog','rooster','rain','sea_waves','crackling_fire','crying_baby','sneezing','clock_tick','helicopter','chainsaw']
for category in categories:
 seen=set();chosen=[]
 for r in rows:
  if r['category']==category and r['esc10']=='True' and r['src_file'] not in seen:
   seen.add(r['src_file']);chosen.append(r)
  if len(chosen)==4:break
 for n,x in enumerate(chosen):
  f=x['filename'];id='sound-'+category+(f'-{n+1}' if n else '');dest=M/f
  if not dest.exists():dest.write_bytes(get(base+'audio/'+f).content)
  key='-'.join(f.split('-')[:-1])+'.ogg';attr=next((line for line in license.splitlines() if '['+key+']' in line),'ESC-10 · CC BY 3.0; see full license')
  add(dict(id=id,type='audio',title=category.replace('_',' ').capitalize()+f' · recording {n+1}',src='media/'+f,duration=5,credit='ESC-10 / Karol J. Piczak. '+attr,source='https://github.com/karolpiczak/ESC-50',group='sound',label=category,sourceRecording=x['src_file'],split='test' if n==3 else 'train'))
 print('Prepared 4 recordings:',category,flush=True)
texts=[('cat-hi','बिल्ली घर में आराम कर रही है।','Hindi · resting cat'),('cat-es','Un gato descansando en casa.','Spanish · resting cat'),('cat-fr','Un chat se repose à la maison.','French · resting cat'),('cat-en','A cat resting at home.','English · resting cat'),('dog-hi','एक कुत्ते के भौंकने की आवाज़।','Hindi · dog barking'),('dog-es','El sonido de un perro ladrando.','Spanish · dog barking'),('dog-en','A dog barking loudly.','English · dog barking'),('coffee-gu','કોફીનો ગરમ કપ.','Gujarati · hot coffee'),('coffee-fr','Une tasse de café chaud.','French · hot coffee'),('coffee-en','A warm cup of coffee.','English · hot coffee'),('wave-hi','समुद्र के किनारे लहरों की आवाज़।','Hindi · sea waves'),('wave-en','Waves washing onto the shore.','English · sea waves'),('fire-en','The crackling sound of a fire.','English · crackling fire'),('rain-en','Rain tapping against a window.','English · rain'),('guitar-en','Someone playing an acoustic guitar.','English · guitar'),('train-en','A train passing on a railway track.','English · train'),('clock-en','A clock ticking steadily.','English · clock'),('bird-en','A rooster crowing in the morning.','English · rooster'),('heli-en','The thudding blades of a helicopter.','English · helicopter'),('pizza-en','A pizza with vegetables and melted cheese.','English · pizza')]
for key,caption,title in texts:add(dict(id='phrase-'+key,type='text',title=title,text=caption,group='caption',credit='Course teaching phrase; translations for exploration'))
(P/'EXPANDED-MEDIA-CREDITS.md').write_text('# Additional media\n\nEach item keeps its original license, including the share-alike terms for adapted clips and stills. These licenses apply to the media, not to the application code.\n\n'+'\n'.join(credits))
(P/'gallery.json').write_text(json.dumps(list(items.values()),ensure_ascii=False,indent=2))
from collections import Counter
print(len(items),Counter(x['type'] for x in items.values()))
