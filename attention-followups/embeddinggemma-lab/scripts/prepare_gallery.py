"""Rebuild the small, attributed teaching collection from public sources."""
import csv, io, json, pathlib, shutil, urllib.request, re, hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]; P=ROOT/'public'; MEDIA=P/'media'
SOURCE=ROOT.parent/'clip/lab/clip-mini-gallery'
items=[]
REMOTE='https://nipunbatra.github.io/attention/clip/lab/clip-mini-gallery/'
def source_bytes(name):
 local=SOURCE/name
 return local.read_bytes() if local.exists() else urllib.request.urlopen(REMOTE+name).read()
for x in json.loads(source_bytes('manifest.json')):
 (MEDIA/x['file']).write_bytes(source_bytes(x['file']))
 items.append(dict(id=x['id'],type='image',title=x['title'],src='media/'+x['file'],credit=x['credit'],source='https://nipunbatra.github.io/attention/clip/lab/clip-mini-gallery/CREDITS.md'))
 items.append(dict(id='caption-'+x['id'],type='text',title='Caption: '+x['title'],text=x['caption'],credit='Course teaching caption',group='caption'))
(P/'IMAGE-CREDITS.md').write_bytes(source_bytes('CREDITS.md'))
base='https://raw.githubusercontent.com/karolpiczak/ESC-50/master/'
rows=list(csv.DictReader(io.StringIO(urllib.request.urlopen(base+'meta/esc50.csv').read().decode())))
license=(P/'ESC-50-LICENSE.txt').read_text()
for category in ['dog','rooster','rain','sea_waves','crackling_fire','crying_baby','sneezing','clock_tick','helicopter','chainsaw']:
 x=next(r for r in rows if r['category']==category and r['esc10']=='True'); f=x['filename']; dest=MEDIA/f
 if not dest.exists():dest.write_bytes(urllib.request.urlopen(base+'audio/'+f).read())
 key='-'.join(f.split('-')[:-1])+'.ogg'; attr=next((line for line in license.splitlines() if '['+key+']' in line),'ESC-10 · CC BY 3.0; see full license')
 items.append(dict(id='sound-'+category,type='audio',title=category.replace('_',' ').capitalize(),src='media/'+f,duration=5,credit='ESC-10 / Karol J. Piczak. '+attr,source='https://github.com/karolpiczak/ESC-50',group='sound'))
 items.append(dict(id='caption-sound-'+category,type='text',title='Sound description: '+category.replace('_',' '),text='The sound of '+category.replace('_',' ')+'.',credit='Course teaching caption',group='caption'))
# Short passages are authored for the class. Retrieval exposes the passage, not a generated answer.
docs=[('normalization','Why normalize?','Divide an embedding by its L2 norm to make its length one. The dot product of two unit vectors equals their cosine similarity.'),('clip-loss','The CLIP training loss','CLIP compares each image with every caption in a batch. Row and column softmaxes form two distributions. The loss averages the negative log probabilities of the observed image-caption partners.'),('retrieval','Retrieval and generation','An embedding model finds relevant source passages. A separate generative model can use those passages to write an answer. A high retrieval score is not a guarantee that the passage answers the question.'),('causal','The text readout in original CLIP','Original CLIP uses causal self-attention in its text Transformer. The end-of-text token can attend to the whole preceding description. Its final state provides the text representation.'),('mrl','Shorter vectors','With Matryoshka embeddings, keep the first 512, 256, or 128 coordinates, then normalize again. Shorter vectors save index memory. Multimodal retrieval at 128 dimensions can lose quality.'),('solar','Solar power','Photovoltaic panels convert sunlight into electrical energy. Solar farms arrange many panels in rows. Power production varies with sunlight and cloud cover.')]
for id,title,text in docs:items.append(dict(id='doc-'+id,type='text',title=title,text=text,group='document',credit='Passage written for this course'))
codes=[('normalize.py','def unit(x):\n    return x / np.linalg.norm(x)'),('search.py','scores = gallery @ query\norder = np.argsort(-scores)\nreturn order[:5]'),('contrastive.py','logits = image_vectors @ text_vectors.T / temperature\ny = torch.arange(len(image_vectors))\nloss = (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2'),('frames.py','for second in range(0, duration, 4):\n    clip = video.subclip(second, second + 4)\n    index.append((second, embed(clip)))')]
for title,text in codes:items.append(dict(id='code-'+title[:-3],type='text',title=title,text=text,group='code',credit='Code example written for this course'))
for x in items:
 if x.get('src'):x['sha256']=hashlib.sha256((P/x['src']).read_bytes()).hexdigest()
(P/'gallery.json').write_text(json.dumps(items,indent=2,ensure_ascii=False))
print(len(items),'items',sum(x['type']=='image' for x in items),'images',sum(x['type']=='audio' for x in items),'audio clips')
