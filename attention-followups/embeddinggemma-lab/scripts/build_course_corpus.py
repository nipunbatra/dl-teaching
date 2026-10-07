"""Extract cited passages and code from published course checkouts.
Run: python build_course_corpus.py --course-repo PATH --attention-repo PATH
Build dependencies: beautifulsoup4==4.13.4, pypdf==5.4.0; Poppler for thumbnails.
No summaries are generated. Source text, original links and hashes are retained.
"""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,unicodedata
from bs4 import BeautifulSoup
from pypdf import PdfReader
p=argparse.ArgumentParser();p.add_argument('--course-repo',type=Path,required=True);p.add_argument('--attention-repo',type=Path,required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'public';items=[];decks=[]
C='https://nipunbatra.github.io/dl-teaching/';A='https://nipunbatra.github.io/attention/'
def sha(b):return hashlib.sha256(b).hexdigest()
def clean(t):
 t=unicodedata.normalize('NFKC',t).replace('\u00ad','')
 return re.sub(r'[ \t]+',' ',re.sub(r'\n{3,}','\n\n',t)).strip()
def chunks(t,size=1900):
 lines=t.splitlines();out=[];buf=''
 for line in lines:
  if len(buf)+len(line)>size and len(buf)>150:out.append(buf.strip());buf=''
  while len(line)>size:
   cut=line.rfind(' ',0,size)
   if cut<100:cut=size
   if buf:out.append(buf.strip());buf=''
   out.append(line[:cut]);line=line[cut:].lstrip()
  buf+=line+'\n'
 if buf.strip():out.append(buf.strip())
 return out

def add(ident,title,text,meta):
 for i,part in enumerate(chunks(text)):
  if len(re.sub(r'\W','',part))<45:continue
  items.append(dict(id=ident+(f'-part-{i+1}' if i else ''),type='text',group='document',title=title,text=part,part=i+1,contentHash=sha(part.encode()),**meta))

pdfs=[('L1','Likelihood and loss'),('L2','Linear models and neural networks'),('L3','Backpropagation and autograd'),('L3A','Calculus for deep learning'),('L5','Optimization'),('L5B','Valid outputs by design'),('L5C','Learning-rate schedules'),('L7','Generalization and regularization')]
thumbs=OUT/'course-thumbnails';thumbs.mkdir(exist_ok=True)
for key,label in pdfs:
 f=a.course_repo/'slides-pdf'/f'{key}.pdf'; reader=PdfReader(f);seen=set();before=len(items)
 # Exact PDF pages also serve as visual context when extraction flattens equations.
 prefix=thumbs/key
 if not list(thumbs.glob(key+'-*.jpg')):
  subprocess.run(['pdftoppm','-jpeg','-jpegopt','quality=65','-scale-to','640',str(f),str(prefix)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 pics=sorted(thumbs.glob(key+'-*.jpg'))
 for n,page in enumerate(reader.pages,1):
  text=clean(page.extract_text()or'');lines=text.splitlines()
  if lines and re.fullmatch(r'\d+',lines[-1].strip()):lines.pop()
  text='\n'.join(lines)
  if sha(text.encode())in seen:continue
  seen.add(sha(text.encode()));heading=lines[0][:150]if lines else label
  add(f'course-{key.lower()}-p{n}',f'{label} · {heading}',text,dict(lecture=label,lectureId=key,page=n,source=C+f'slides-pdf/{key}.pdf#page={n}',sourceFile=f'slides-pdf/{key}.pdf',sourceHash=sha(f.read_bytes()),sourceKind='pdf',thumbnail='course-thumbnails/'+pics[n-1].name,credit='Nipun Batra · published course handout'))
 decks.append(dict(id=key,title=label,source=C+f'slides-pdf/{key}.pdf',passages=len(items)-before,pages=len(reader.pages)))
 print(label,len(items)-before,flush=True)

htmls=[(a.attention_repo,'part1.html',A,'Tokens','Characters and next-token prediction'),(a.attention_repo,'attention.html',A,'Attention','Self-attention'),(a.attention_repo,'vision1.html',A,'ViT','Vision Transformers'),(a.attention_repo,'clip/index.html',A,'CLIP','CLIP'),(a.attention_repo,'from-clip-to-vlm/index.html',A,'VLM','Vision-language models'),(a.attention_repo,'object-detection/index.html',A,'Detection','Object detection'),(a.attention_repo,'beyond-boxes/index.html',A,'Dense','Segmentation, pose and depth'),(a.course_repo,'interactives/from-attention-to-applications/index.html',C,'Architectures','Transformer architectures'),(a.course_repo,'interactives/cnn/index.html',C,'CNN','Convolutional neural networks')]
for repo,path,base,key,label in htmls:
 f=repo/path;soup=BeautifulSoup(f.read_text(),'html.parser');frames=soup.select('.frame[id], section.slide[id]')
 if len(frames)<5:frames=soup.select('section.sec[id]')
 if not frames:frames=soup.select('section[id]')
 before=len(items);seen=set()
 for n,frame in enumerate(frames,1):
  anchor=frame['id'];heading=frame.get('data-title')or(frame.find(['h2','h3']).get_text(' ',strip=True)if frame.find(['h2','h3'])else anchor)
  for x in frame.select('script,style,button,input,select,nav,.credit,.credits,.source-note,.source-notes,.katex-html,.mobile-labels'):x.decompose()
  for x in frame.select('annotation'):
   math=x.find_parent('math')
   if math:math.replace_with(' '+x.get_text()+' ')
  lines=[];seenlines=set()
  for line in frame.get_text('\n',strip=True).splitlines():
   line=clean(line)
   if line and line not in seenlines:lines.append(line);seenlines.add(line)
  text=clean('\n'.join(lines))
  if sha(text.encode()) in seen:continue
  seen.add(sha(text.encode()))
  source=base+path.replace('index.html','')+'#'+anchor
  image=frame.find('img',src=True);thumb=None
  if image and not image['src'].startswith('data:'):
   from urllib.parse import urljoin
   thumb=urljoin(base+path,image['src'])
  add(f'course-{key.lower()}-{anchor}',f'{label} · {heading}',text,dict(lecture=label,lectureId=key,slide=n,anchor=anchor,source=source,sourceFile=path,sourceKind='html',sourceHash=sha(f.read_bytes()),thumbnail=thumb,credit='Nipun Batra · published HTML lecture'))
 decks.append(dict(id=key,title=label,source=base+path.replace('index.html',''),passages=len(items)-before,slides=len(frames)))
 print(label,len(items)-before,flush=True)

# Published notebook cells; preserve exact source and notebook links.
notebooks=[]
for folder in ['L01','L02','L03','L03A','L05','L07','L08']:
 notebooks+=sorted((a.course_repo/'notebooks'/folder).glob('*.ipynb'))
notebooks += [a.course_repo/'notebooks/12-attention-by-hand.ipynb',a.course_repo/'notebooks/L17/01_patch_embedding.ipynb',a.attention_repo/'clip/notebooks/clip-applications.ipynb',a.attention_repo/'clip/notebooks/clip-simple.ipynb']
code_count=0
for f in notebooks:
 if not f.exists():continue
 repo=a.attention_repo if f.is_relative_to(a.attention_repo)else a.course_repo
 reponame='attention'if repo==a.attention_repo else'dl-teaching';branch='main'if reponame=='attention'else'master';relative=f.relative_to(repo).as_posix();notebook=json.loads(f.read_text());heading=f.stem.replace('_',' ');seen=set()
 for n,cell in enumerate(notebook['cells'],1):
  source=''.join(cell.get('source',[]))
  if cell['cell_type']=='markdown':
   h=re.search(r'^#{1,4}\s+(.+)',source,re.M)
   if h:heading=h.group(1)
   continue
  if cell['cell_type']!='code' or len(source.strip())<90 or len(source.splitlines())<3:continue
  if not re.search(r'\b(def |class |torch|loss|backward|optimizer|softmax|np\.|gradient|train|predict)',source):continue
  if source.strip().startswith(('!pip','%pip','!wget')):continue
  digest=sha(source.encode())
  if digest in seen:continue
  seen.add(digest)
  link=f'https://github.com/nipunbatra/{reponame}/blob/{branch}/{relative}'
  colab=f'https://colab.research.google.com/github/nipunbatra/{reponame}/blob/{branch}/{relative}'
  cellid=cell.get('metadata',{}).get('id')or cell.get('id')
  if cellid:colab+='#scrollTo='+cellid
  # Keep complete cells if possible; long cells are split at line boundaries.
  for i,part in enumerate(chunks(source,2400)):
   if len(part)<70:continue
   ident='code-'+sha((relative+str(n)+str(i)).encode())[:16]
   items.append(dict(id=ident,type='text',group='code',title=f'{heading} · {f.stem} · cell {n}',text=part,lecture=f.stem.replace('_',' '),lectureId='Notebooks',source=link,colab=colab,sourceFile=relative,sourceKind='notebook',cell=n,cellId=cellid,part=i+1,contentHash=sha(part.encode()),sourceHash=sha(f.read_bytes()),credit='Nipun Batra · published course notebook'))
   code_count+=1
print('Code snippets',code_count,flush=True)
assert len({i['id']for i in items})==len(items)
(OUT/'course-corpus.json').write_text(json.dumps(items,ensure_ascii=False,indent=2)+'\n')
meta={'date':'2026-10-07','passages':sum(i['group']=='document'for i in items),'code':code_count,'items':len(items),'decks':decks,'notebooks':len(notebooks),'extraction':'Exact slide text and notebook cells; diagrams and equation layout remain in the linked originals.'}
(OUT/'course-manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
f=OUT/'course-embeddings.json'
if not f.exists():
 data=json.loads((OUT/'embeddings.json').read_text());data['items']={};data['collection']='course';f.write_text(json.dumps(data)+'\n')
print('TOTAL',len(items),flush=True)
