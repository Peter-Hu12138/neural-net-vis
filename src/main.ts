import './styles.css';
import { client, rebuild, setProbe } from './actions';
import { TEST_COUNT, TRAIN_COUNT, loadMnist, sampleToFloat } from './data/mnist';
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

rebuild();

const engine = () => {
  $('fact-engine').textContent =
    client.mode === 'worker' ? 'In-browser, Web Worker' : client.mode === 'main-thread' ? 'In-browser, main thread' : 'Starting…';
};
engine();
const engineTimer = setInterval(() => {
  engine();
  if (client.mode !== 'starting') clearInterval(engineTimer);
}, 250);

loadMnist((done, total) => {
  $('fact-data').textContent = `MNIST, loading ${done}/${total}…`;
})
  .then((data) => {
    store.data = data;
    client.post({ type: 'data', data: { trainX: data.trainX, trainY: data.trainY, testX: data.testX, testY: data.testY } });
    analysis.setData(data.testX, data.testY);
    $('fact-data').textContent = `MNIST · ${int(TRAIN_COUNT)} train · ${int(TEST_COUNT)} test`;
    const i = 0;
    setProbe({ x: sampleToFloat(data.testX, i), label: data.testY[i], caption: `Test digit #${i} · label ${data.testY[i]}`, key: `test:${i}` });
    store.emit('data');
  })
  .catch((err: unknown) => {
    $('fact-data').textContent = `MNIST failed to load: ${err instanceof Error ? err.message : String(err)}. Reload to retry.`;
  });

// Exposed for the browser tests and for poking around in the console.
(window as unknown as { raster: unknown }).raster = { store, client, analysis };
