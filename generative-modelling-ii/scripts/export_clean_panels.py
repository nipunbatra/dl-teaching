"""Export the existing clean MNIST examples without altering the stored arrays."""
from pathlib import Path
import json
from PIL import Image
R=Path(__file__).resolve().parents[1]
for i,a in enumerate(json.loads((R/'assets/demo-data.json').read_text())['digits'][:4]):
 image=Image.new('L',(28,28))
 image.putdata([round(max(0,min(255,(v+1)*127.5))) for row in a for v in row])
 image.resize((280,280),Image.Resampling.NEAREST).save(R/f'assets/clean-{i}.png')
