import waveforms from './waveforms.json';
export function waveform(id){
 const bins=waveforms[id];if(!bins)return '';
 const peak=Math.max(...bins.flat().map(Math.abs),0.001);
 return `<svg class="audio-waveform" viewBox="0 0 320 90" role="img" aria-label="Recorded waveform, scaled to its own peak; time runs left to right"><path d="M0 45H320" stroke="currentColor" opacity=".18"/>${bins.map(([lo,hi],i)=>`<path d="M${i*4+1} ${45-hi/peak*40}V${45-lo/peak*40}" stroke="currentColor" stroke-width="2"/>`).join('')}</svg>`;
}
