from pathlib import Path
import json,html,re,subprocess,shutil,base64,mimetypes
from scenes import SCENES,steps
import appendix
from presenter_lines import LINES,TITLES
from teaching_figures import apply_figures
from full_deck_figures import apply_full_deck
R=Path(__file__).resolve().parents[1]
# Preserve house-style 1280x720 stage and reveal semantics, with a scene/build adapter.
for s in SCENES:
 if s['visual'].startswith('<div class="diagram">'):s['layout']='wide'
 assert s['id'] in LINES,s['id']
 s['talk_through'],s['takeaway']=LINES[s['id']]
 s['title']=TITLES.get(s['id'],s['title'])
apply_figures(SCENES)
apply_full_deck(SCENES)
# Export each authored diagram as a portable, editable SVG as well as inline HTML.
(R/'figures').mkdir(exist_ok=True)
for s in SCENES:
 if 'figure-led' in s['layout']:
  vector=re.search(r'<svg\b[\s\S]*</svg>',s['visual']).group()
  def embed_asset(m):
   asset=R/m[1];mime=mimetypes.guess_type(asset)[0] or 'application/octet-stream'
   return 'href="data:'+mime+';base64,'+base64.b64encode(asset.read_bytes()).decode()+'"'
  vector=re.sub(r'href="(assets/[^\"]+)"',embed_asset,vector)
  (R/'figures'/f'{s["id"]}.svg').write_text(vector)
meta=[{k:v for k,v in s.items() if k not in ['visual','steps']} for s in SCENES]
D=dict(scenes=meta,data=json.loads((R/'assets/demo-data.json').read_text()),tuples=json.loads((R/'assets/tuple-data.json').read_text()),oracle=json.loads((R/'assets/oracle-data.json').read_text()))
parts=[]
for s in SCENES:
 visual=s['visual'].replace('id="tip"',f'id="tip-{s["id"]}"').replace('url(#tip)',f'url(#tip-{s["id"]})')
 if s['id'] in ['s27','s38']:
  def reveal_code(m):
   lines=m.group(1).split('\n'); groups=([1,1,2,3,4,4,4,4,4] if s['id']=='s27' else [1,1,1,1,1,1,1,2,2,2,2]);return '<pre>'+''.join(f'<span class="code-line" data-step="{groups[min(i,len(groups)-1)]}">{line}</span>' for i,line in enumerate(lines))+'</pre>'
  visual=re.sub(r'<pre>([\s\S]*?)</pre>',reveal_code,visual)
 body=f'<div class="visual">{visual}</div>'+(steps(s['steps']) if any(s['steps']) else '')
 explanation='<div class="explanation">'+''.join(f'<div class="explanation-step"><span>Build {j+1}</span>{v}</div>' for j,v in enumerate(s['explanation_steps']))+'</div>'
 explanation=explanation.replace('id="tip"',f'id="tip-notes-{s["id"]}"').replace('url(#tip)',f'url(#tip-notes-{s["id"]})')
 notes=f'<details class="notes"><summary>Instructor notes and discussion answer</summary><p><strong>Talk through:</strong> {s["talk_through"]}</p><p><strong>Takeaway:</strong> {s["takeaway"]}</p><p>{s["narration"]}</p><p><strong>Question:</strong> {s["question"] or s["title"]}</p><p><strong>Answer:</strong> {s["answer"] or s["narration"]}</p>{explanation}</details>'
 cue=f'<aside class="teaching-line" aria-label="Presenter cue"><span class="cue-label">Takeaway</span><p>{html.escape(s["takeaway"])}</p></aside>'
 parts.append(f'<section class="slide" id="{s["id"]}" data-builds="{s["builds"]}"><header><div class="eyebrow">{s["section"]} · {s["id"]}</div><h2>{s["title"]}</h2></header><div class="body {s["layout"]}">{body}</div>{cue}<footer><a href="{s["source"]}">Primary source ↗</a><a href="#s10/2">Course map</a><span>{s["id"]} · {s["builds"]} builds · ES 667</span></footer>{notes}</section>')
h='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Generative Modelling II | From Denoising to Text-to-Image</title><link rel="stylesheet" href="src/lecture.css"><link rel="stylesheet" href="src/visual-refinement.css"></head><body><div class="mast"><p class="eyebrow">ES 667 · IIT Gandhinagar · Nipun Batra</p><h1>From denoising to text-to-image generation</h1><p>Generative Modelling II · 60 core scenes, 158 builds. Full route: 120 minutes. Compressed overview: 90 minutes. Appendix readings follow the main ending.</p><div class="links"><a href="?present#s01/1">Present lecture</a><a href="exports/generative-modelling-ii-reading.pdf">Reading PDF</a><a href="exports/generative-modelling-ii-all-builds.pdf">All builds PDF</a><a href="../slides/generative2.html">Older diffusion lecture</a><a href="INSTRUCTOR_NOTES.md">Instructor notes</a><a href="notebooks/ddpm-companion.ipynb">Notebook</a></div></div><main class="stage">'''+''.join(parts)+'''</main><nav class="toolbar" aria-label="Lecture controls"><span class="brand">Generative Modelling II</span><button id="previous" aria-label="Previous build">←</button><span class="progress" id="progress"></span><button id="next" aria-label="Next build">→</button><button id="finish">Show all</button><button id="mode">Present</button><button id="overview">Overview</button><button id="notes">Notes</button><label>Route <select id="route"><option value="full">Full · 120 min</option><option value="compressed">Compressed · 90 min</option></select></label></nav><dialog id="dialog"><button class="close" id="close">Close</button><h3 id="dialog-title"></h3><div id="dialog-body"></div></dialog><script type="application/json" id="lecture-data">'''+json.dumps(D,separators=(',',':')).replace('</','<\\/')+'''</script><script src="src/lecture.js"></script></body></html>'''
# SVG markers have document-unique IDs even with repeated figures.
counter=0
def marker(m):
 global counter
 counter+=1
 return m.group().replace('id="tip"',f'id="tip-{counter}"').replace('url(#tip)',f'url(#tip-{counter})')
h=re.sub(r'<svg\b[\s\S]*?</svg>',marker,h)
(R/'index.html').write_text(h)
(R/'src/manifest.json').write_text(json.dumps(meta,indent=2))
(R/'STORYBOARD.md').write_text('# Generative Modelling II storyboard\n\n60 core scenes / 158 core builds. '+str(len(SCENES)-60)+' appendix scenes / '+str(sum(s['builds'] for s in SCENES[60:]))+' appendix builds.\n\n'+ '\n'.join(f'- {s["id"]}: {s["title"]} ({s["builds"]} builds) → {s["section"]}' for s in SCENES))
routes='''Full route: 120 minutes (12 / 20 / 23 / 21 / 15 / 17 / 12). Compressed overview: 90 minutes (8 / 15 / 18 / 16 / 11 / 14 / 8). The compressed navigation jumps over intermediate forward-expansion builds in s18, shows the final weighted-objective state in s31, and summarises s41 and s49. Present the application gallery briefly; reduce pauses and use the retained notes for details. This does not give the same depth as the full route. Both PDFs include all content irrespective of route. Appendix is outside timings. Neither pacing estimate has been classroom rehearsed.'''
notes='# Instructor notes\n\n'+routes+'\n\nKeyboard: ←/→, PageUp/PageDown, Space; P toggles present; O overview; N notes; Escape closes dialog or leaves presentation. Inputs keep their own keyboard events.\n\n'
for i,s in enumerate(SCENES):
 notes+=f'## {s["id"]} · {s["title"]}\n\nTalk through: {s["talk_through"]}\n\nTakeaway: {s["takeaway"]}\n\n{s["narration"]}\n\nBuild explanations:\n'+''.join(f'{j+1}. '+re.sub('<[^>]+>',' ',html.unescape(v)).strip()+'\n' for j,v in enumerate(s['explanation_steps']))+f'\nAsk: {s["question"] or s["title"]}\n\nExpected answer: {s["answer"] or s["narration"]}\n\nTransition: '+(SCENES[i+1]['title'] if i+1<len(SCENES) else 'Return to the core recap.')+f'\n\nSource: {s["source"]}\n\n'
(R/'INSTRUCTOR_NOTES.md').write_text(notes)
(R/'requirements-to-scenes.json').write_text(json.dumps([{**q,'implemented_scene':q['id'],'html':'index.html#'+q['id'],'qa_status':'pending rendered verification'} for q in json.loads((R/'src/requirements.json').read_text())],indent=2))
subprocess.run(['node',str(R/'src/render_math.cjs')],check=True)
print(f'Built {len(SCENES)} scenes, {sum(s["builds"] for s in SCENES)} builds')
