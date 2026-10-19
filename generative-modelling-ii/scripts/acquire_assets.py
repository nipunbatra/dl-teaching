from pathlib import Path
import urllib.request,json,hashlib,concurrent.futures
from PIL import Image,ImageOps,ImageDraw
R=Path(__file__).resolve().parents[1]; A=R/'assets'; A.mkdir(exist_ok=True)
items=[
('ddpm','https://hojonathanho.github.io/diffusion/assets/img/denoising_diffusion_all.png','https://hojonathanho.github.io/diffusion/','Ho, Jain, Abbeel','DDPM published unconditional mosaic',[1,32,60]),
('ldm','https://ommer-lab.com/wp-content/uploads/2022/08/article-Figure17-1.png','https://ommer-lab.com/research/latent-diffusion-models/','Rombach et al.','LDM Figure 17; prompt-conditioned samples; 100 DDIM steps eta=0; seeds unspecified',[2,47,50,60]),
('gligen','https://gligen.github.io/images/groundtext2img.png','https://gligen.github.io/','Li et al.','GLIGEN grounding and generated outputs',[3]),
('repaint','https://user-images.githubusercontent.com/11280511/150803812-a4729ef8-6ad4-46aa-ae99-8c27fbb2ea2e.png','https://github.com/andreas128/RePaint','Lugmayr et al.','RePaint; blue regions are missing; multiple sampled fillings',[4,54,60]),
('sdedit','https://sde-image-editing.github.io/images/stroke_based_generation.jpg','https://sde-image-editing.github.io/','Meng et al.','SDEdit stroke-based generation',[5,46]),
('sdedit-edit','https://sde-image-editing.github.io/images/stroke_edit.jpg','https://sde-image-editing.github.io/','Meng et al.','SDEdit stroke editing',[5]),
('sr3','https://iterative-refinement.github.io/images/super_res_examples.png','https://iterative-refinement.github.io/','Saharia et al.','SR3 super-resolution comparison',[6,54,55,60]),
('control','https://raw.githubusercontent.com/lllyasviel/ControlNet/main/github_page/p1.png','https://github.com/lllyasviel/ControlNet','Zhang et al.','ControlNet Canny edge condition and outputs',[7,50]),
('dreambooth','https://dreambooth.github.io/DreamBooth_files/teaser_static.jpg','https://dreambooth.github.io/','Ruiz et al.','DreamBooth reference subjects and generated contexts',[8,45]),
('sat','https://www.samarkhanna.com/DiffusionSat/static/images/diffusion_sat_samples.png','https://www.samarkhanna.com/DiffusionSat/','Khanna et al.','DiffusionSat metadata-conditioned imagery',[9,55]),
('sat-sr','https://www.samarkhanna.com/DiffusionSat/static/images/diffusion_sat_superres.png','https://www.samarkhanna.com/DiffusionSat/','Khanna et al.','DiffusionSat super-resolution',[9])]
def get(v):
 key,url,page,authors,caption,scenes=v;p=A/(key+Path(url).suffix)
 try:
  if not p.exists():
   req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
   with urllib.request.urlopen(req,timeout=45) as f:p.write_bytes(f.read())
  im=Image.open(p);im.verify()
  return dict(id=key,local_path=str(p.relative_to(R)),source_page=page,asset_url=url,authors_or_owner=authors,paper_or_dataset=caption.split(';')[0],figure_or_caption=caption,license_or_usage_note='Author-published research figure, attributed for local classroom discussion. No blanket license for embedded photographs is asserted; review permissions before external redistribution.',retrieved_date='2026-10-19',transformations='Unmodified source; layout may display an explicitly documented crop.',evidence_type='published model output',used_in_scene_ids=[f's{i:02}' for i in scenes],sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 except Exception as e:return {'id':key,'error':str(e),'asset_url':url}
rows=list(concurrent.futures.ThreadPoolExecutor(6).map(get,items));(R/'asset-manifest.json').write_text(json.dumps(rows,indent=2))
canvas=Image.new('RGB',(1200,320*((len(rows)+2)//3)),'#f7f8fa');d=ImageDraw.Draw(canvas)
for i,r in enumerate(rows):
 x=(i%3)*400;y=(i//3)*320;d.text((x+10,y+5),r['id'],fill='black')
 if 'local_path' in r:
  im=Image.open(R/r['local_path']).convert('RGB');im.thumbnail((390,285));canvas.paste(im,(x+5,y+28))
print(json.dumps([{k:v for k,v in r.items() if k in ['id','error','local_path']} for r in rows],indent=2));canvas.save(R/'qa/source-contact.jpg')
