"""Small, actual waveform previews. Does not change model inputs or embeddings."""
from pathlib import Path
import array,json,wave,sys
root=Path(__file__).resolve().parents[1]
result={}
for item in json.loads((root/'public/gallery.json').read_text()):
 if item['type']!='audio':continue
 with wave.open(str(root/'public'/item['src'])) as wav:
  assert wav.getsampwidth()==2,'Expected PCM16 samples'
  samples=array.array('h',wav.readframes(wav.getnframes()))
  if sys.byteorder!='little':samples.byteswap()
  channels=wav.getnchannels()
  samples=[sum(samples[i:i+channels])/channels/32768 for i in range(0,len(samples),channels)]
  peaks=[]
  for k in range(80):
   chunk=samples[k*len(samples)//80:(k+1)*len(samples)//80]
   peaks.append([round(min(chunk),4),round(max(chunk),4)])
  result[item['id']]=peaks
(root/'src/waveforms.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
print('Computed real waveform previews for',len(result),'recordings')
