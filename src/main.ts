import './styles.css';
import * as actions from './actions';
import { client, loading, setDataset } from './actions';
import { store } from './store';
import { mountBackprop } from './ui/backprop';
import { mountBuilder } from './ui/builder';
import { mountCharts } from './ui/charts';
import { mountControls } from './ui/controls';
import { mountDataPanel } from './ui/dataPanel';
import { $, int } from './ui/dom';
import { mountDrawpad } from './ui/drawpad';
import { mountInspector } from './ui/inspector';
import { mountNetworkView } from './ui/networkView';
import { analysis } from './analysis/client';
import { mountAttribution } from './ui/attributionView';
import { mountDistributions } from './ui/distView';
import { mountEmbedding } from './ui/embeddingView';
import { mountUnits } from './ui/unitsView';

mountControls();
mountBuilder();
mountNetworkView();
mountDrawpad();
mountInspector();
mountCharts();
mountBackprop();
mountDataPanel();
mountDistributions();
mountUnits();
mountAttribution();
mountEmbedding();

const engine = () => {
  $('fact-engine').textContent =
    client.mode === 'worker' ? 'In-browser, Web Worker' : client.mode === 'main-thread' ? 'In-browser, main thread' : 'Starting…';
};
engine();
const engineTimer = setInterval(() => {
  engine();
  if (client.mode !== 'starting') clearInterval(engineTimer);
}, 250);

const dataFact = () => {
  const info = store.info;
  const d = store.data;
  const el = $('fact-data');
  if (loading) el.textContent = `${info.name}, loading ${loading.done}/${loading.total}…`;
  else if (!d) el.textContent = `${info.name}, loading…`;
  else el.textContent = `${info.name} · ${int(d.trainY.length)} train · ${int(d.testY.length)} test`;
  const input = $('fact-input');
  const s = store.input;
  input.textContent = info.kind === 'image' ? `${s.h} × ${s.w} px, ${s.c === 1 ? 'grey' : 'colour'}` : `${s.c} feature${s.c === 1 ? '' : 's'} from ${info.dims}-D points`;
};
store.on('dataset', dataFact);
// Section 03 is about trying the model on your own input; what that means depends on the data.
const tryTitle = () => {
  const info = store.info;
  $('h-draw').textContent = info.kind === 'points' ? 'Decision boundary' : info.image!.shape.c === 3 ? 'Try a photo' : 'Draw';
  $('drawpad').hidden = info.kind === 'points';
  $('boundary').hidden = info.kind !== 'points';
};
store.on('dataset', tryTitle);
tryTitle();
store.on('data', dataFact);

setDataset('mnist').catch((err: unknown) => {
  $('fact-data').textContent = `${store.info.name} failed to load: ${err instanceof Error ? err.message : String(err)}. Reload to retry.`;
});

// Exposed for the browser tests and for poking around in the console.
(window as unknown as { raster: unknown }).raster = { store, client, analysis, actions };
