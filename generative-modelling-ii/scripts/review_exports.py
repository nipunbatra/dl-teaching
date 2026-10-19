from pathlib import Path
import json,hashlib
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
from pypdf import PdfReader
R=Path(__file__).resolve().parents[1];out=R/'qa/pdf-renders';out.mkdir(exist_ok=True)
for name in ['generative-modelling-ii-reading','diffusion-two-page-recap']:
 pdf=pdfium.PdfDocument(str(R/'exports'/f'{name}.pdf'))
 for start in range(0,len(pdf),12):
  contact=Image.new('RGB',(1536,1220),'#d9dee6');d=ImageDraw.Draw(contact)
  for j,n in enumerate(range(start,min(start+12,len(pdf)))):
   im=pdf[n].render(scale=.4 if 'reading' in name else 1.4).to_pil().convert('RGB')
   if 'recap' in name:im.save(out/f'recap-{n+1}.png')
   im.thumbnail((500,280));x=j%3*512;y=j//3*305;contact.paste(im,(x,y+22));d.text((x+8,y+5),f'{name} page {n+1}',fill='black')
  contact.save(out/f'{name}-contact-{start//12+1}.jpg')
 for n in ([4,17,26,30,36,40,48,56,62] if 'reading' in name else []):pdf[n].render(scale=1).to_pil().save(out/f'reading-{n+1}.png')
 pdf.close()
# Verify destination and actual URL independently of annotation counts.
for kind in ['reading','all-builds']:
 rr=PdfReader(R/'exports'/f'generative-modelling-ii-{kind}.pdf');ann=[a.get_object() for a in rr.pages[0]['/Annots']]
 assert any(a.get('/A',{}).get('/URI')=='https://hojonathanho.github.io/diffusion/' for a in ann)
 dest=next(a['/Dest'] for a in ann if '/Dest'in a)
 assert dest[0].get_object() is not None
 print(kind,'verified source URI and resolved course-map page destination')
