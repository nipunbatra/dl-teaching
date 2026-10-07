"""Rebuild the attributed teaching slideshow and add its four index entries."""
import subprocess,json,pathlib,hashlib
root=pathlib.Path(__file__).resolve().parents[1];p=root/'public'
args=['ffmpeg','-hide_banner','-loglevel','error','-y']
for f in ['chelsea.jpg','coffee.jpg','rocket.jpg']:args+=['-loop','1','-t','4','-i',str(p/'media'/f)]
filters=';'.join(f'[{i}:v]scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]' for i in range(3))+';[v0][v1][v2]concat=n=3:v=1:a=0[v]'
args+=['-filter_complex',filters,'-map','[v]','-t','12','-r','12','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(p/'media/three-scenes.mp4')]
subprocess.run(args,check=True)
items=[x for x in json.loads((p/'gallery.json').read_text()) if not x['id'].startswith('video-')]
for i,(title,start,end) in enumerate([('Three scenes: cat, coffee, rocket',0,12),('Moment 1 · 0–4 seconds',0,4),('Moment 2 · 4–8 seconds',4,8),('Moment 3 · 8–12 seconds',8,12)]):
 items.append(dict(id=f'video-{i}',type='video',title=title,src='media/three-scenes.mp4',start=start,end=end,duration=end-start,group='moment' if i else 'video',credit='Teaching slideshow. Cat: Stefan van der Walt; coffee: Rachel Michetti; rocket: SpaceX. All CC0.',source='IMAGE-CREDITS.md',sha256=hashlib.sha256((p/'media/three-scenes.mp4').read_bytes()).hexdigest()))
(p/'gallery.json').write_text(json.dumps(items,indent=2,ensure_ascii=False))
