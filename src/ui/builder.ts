import './builder.css';
import { select, setFrozen, setKeepWeights, setSpec } from '../actions';
import { featureDefs } from '../data/features';
import { describe, type LayerInfo } from '../nn/network';
import { ACTIVATIONS, fmtShape, size, type Act, type Arch, type ConvSpec, type DenseSpec, type LayerSpec } from '../nn/types';
import { presetsFor, store } from '../store';
import { $, clear, h, int, selectField } from './dom';

const MAX_CONV = 4;
const MAX_DENSE = 3;
const FILTERS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64];
const UNITS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256];
/** Above these the builder warns that training in a browser tab will be slow. */
const BIG_PARAMS = 250_000;
const BIG_MACS = 4_000_000;
const actOptions = ACTIVATIONS.map((a) => ({ value: a.id, label: a.label }));

const LOCK_CLOSED =
  '<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><path d="M4.5 7V5a3.5 3.5 0 0 1 7 0v2" fill="none" stroke="currentColor" stroke-width="1.8"/><rect x="2.5" y="7" width="11" height="8" fill="currentColor"/></svg>';
const LOCK_OPEN =
  '<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><path d="M4.5 7V5a3.5 3.5 0 0 1 6.8-1.2" fill="none" stroke="currentColor" stroke-width="1.8"/><rect x="2.5" y="7" width="11" height="8" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>';

export function layerName(spec: LayerSpec | null, index: number): string {
  if (!spec) return 'Output';
  return `${spec.kind === 'conv' ? 'Conv' : 'Dense'} ${index + 1}`;
}

export function layerDetail(spec: LayerSpec | null): string {
  if (!spec) return `${store.classes} · softmax`;
  const act = ACTIVATIONS.find((a) => a.id === spec.act)!.label;
  if (spec.kind === 'conv') return `${spec.filters} × ${spec.kernel}×${spec.kernel} · ${act}${spec.pool ? ' · pool' : ''}`;
  return `${spec.units} · ${act}`;
}

/** The network's input in words: "28×28×1 grey image" or "5 features: x₁, x₂, x₁², x₂², x₁x₂". */
export function inputWords(): { shape: string; rest: string } {
  const info = store.info;
  const s = store.input;
  if (info.kind === 'image') return { shape: fmtShape(s), rest: s.c === 1 ? ' grey image' : s.c === 3 ? ' colour image' : ` image, ${s.c} channels` };
  let labels: string[];
  try {
    labels = featureDefs(info.dims!, store.features).map((f) => f.label);
  } catch {
    labels = store.features.slice();
  }
  return { shape: `${s.c} feature${s.c === 1 ? '' : 's'}`, rest: `: ${labels.join(', ')}` };
}

/** Multiply-adds for one forward pass, a rough measure of training cost per sample. */
function multiplyAdds(info: LayerInfo[]): number {
  let n = 0;
  for (const l of info) {
    if (l.spec?.kind === 'conv') n += l.spec.filters * l.inShape.c * l.spec.kernel * l.spec.kernel * l.inShape.h * l.inShape.w;
    else n += size(l.inShape) * l.outShape.c;
  }
  return n;
}

/** Why a convolution cannot be added here, or null when it can. */
function convBlocked(arch: Arch): string | null {
  const probe: Arch = { ...arch, layers: [{ kind: 'conv', filters: 8, kernel: 3, act: 'relu', pool: false }] };
  return describe(probe)[0].error ?? null;
}

const optionsWith = (values: number[], current: number) =>
  (values.includes(current) ? values : [...values, current].sort((a, b) => a - b)).map((v) => ({ value: v, label: String(v) }));

export function mountBuilder(): void {
  const root = $('builder');

  const freezeButton = (block: number, name: string) => {
    const on = store.isFrozen(block);
    const b = h('button', {
      type: 'button',
      id: `freeze-${block}`,
      class: 'btn btn-sm freeze',
      'aria-pressed': String(on),
      'aria-label': `Freeze ${name}`,
      title: on ? 'Frozen: training leaves these weights alone. Click to let them learn again.' : 'Freeze: hold these weights fixed while the rest of the network trains.',
      onclick: () => setFrozen(block, !store.isFrozen(block)),
    });
    b.innerHTML = on ? LOCK_CLOSED : LOCK_OPEN;
    b.append(h('span', null, on ? 'Frozen' : 'Freeze'));
    return b;
  };

  const render = () => {
    // Re-rendering replaces every control; keep keyboard focus on the one that was in use.
    const focused = document.activeElement instanceof HTMLElement && root.contains(document.activeElement) ? document.activeElement.id : '';
    clear(root);
    const spec = store.spec;
    const arch = store.arch;
    const info = describe(arch);
    const nConv = spec.filter((l) => l.kind === 'conv').length;
    const nDense = spec.length - nConv;
    const pointsData = store.info.kind === 'points';

    const update = (i: number, patch: Partial<ConvSpec> | Partial<DenseSpec>) => {
      const next = spec.map((l, j) => (j === i ? ({ ...l, ...patch } as LayerSpec) : l));
      setSpec(next);
    };
    const remove = (i: number) => setSpec(spec.filter((_, j) => j !== i));

    const presets = h('div', { class: 'presets', role: 'group', 'aria-label': 'Presets' });
    for (const p of presetsFor(store.info)) {
      presets.append(h('button', { type: 'button', class: 'btn btn-sm', onclick: () => setSpec(structuredClone(p.spec)) }, p.name));
    }

    const input = inputWords();
    const list = h('ol', { class: 'layers' });
    list.append(
      h(
        'li',
        { class: 'layer is-fixed' },
        h('span', { class: 'layer-idx' }, 'IN'),
        h('div', { class: 'layer-main' }, h('span', { class: 'layer-title' }, 'Input'), h('span', { class: 'layer-shape builder-input' }, h('b', null, input.shape), input.rest)),
        h('span'),
      ),
    );

    spec.forEach((l, i) => {
      const li = info[i];
      const fields = h('div', { class: 'layer-fields' });
      const id = (n: string) => `l${i}-${n}`;
      if (l.kind === 'conv') {
        fields.append(
          selectField(id('filters'), 'Filters', optionsWith(FILTERS, l.filters), l.filters, (filters) => update(i, { filters })),
          selectField(id('kernel'), 'Kernel', [{ value: 3, label: '3×3' }, { value: 5, label: '5×5' }], l.kernel, (kernel) => update(i, { kernel: kernel as 3 | 5 })),
          selectField(id('act'), 'Activation', actOptions, l.act, (act) => update(i, { act: act as Act })),
        );
        const pool = h('input', { type: 'checkbox', id: id('pool'), checked: l.pool }) as HTMLInputElement;
        pool.addEventListener('change', () => update(i, { pool: pool.checked }));
        fields.append(h('label', { class: 'check', for: id('pool'), style: { alignSelf: 'end', paddingBottom: '4px' } }, pool, 'Max-pool 2×2'));
      } else {
        fields.append(
          selectField(id('units'), 'Units', optionsWith(UNITS, l.units), l.units, (units) => update(i, { units })),
          selectField(id('act'), 'Activation', actOptions, l.act, (act) => update(i, { act: act as Act })),
        );
      }
      const name = layerName(l, i);
      const title = h('span', { class: 'layer-title', role: 'button', tabindex: '0', title: 'Inspect weights' }, name);
      title.addEventListener('click', () => select(i));
      title.addEventListener('keydown', (e) => {
        if ((e as KeyboardEvent).key === 'Enter') select(i);
      });
      const frozen = store.isFrozen(i);
      list.append(
        h(
          'li',
          { class: `layer${store.selected === i ? ' is-selected' : ''}${frozen ? ' is-frozen' : ''}`, 'data-block': String(i) },
          h('span', { class: 'layer-idx' }, `L${i + 1}`),
          h(
            'div',
            { class: 'layer-main' },
            title,
            fields,
            h(
              'div',
              { class: 'layer-foot' },
              h('span', { class: 'layer-shape' }, '→ ', h('b', null, fmtShape(li.outShape)), ` · ${int(li.params)} params`),
              li.error ? null : freezeButton(i, name),
            ),
            li.error ? h('span', { class: 'layer-error' }, li.error) : null,
          ),
          h('button', { type: 'button', id: `remove-${i}`, class: 'btn btn-sm btn-icon', 'aria-label': `Remove ${name}`, title: 'Remove layer', onclick: () => remove(i) }, '×'),
        ),
      );
    });

    const out = info[info.length - 1];
    const outIdx = spec.length;
    const outTitle = h('span', { class: 'layer-title', role: 'button', tabindex: '0', title: 'Inspect weights' }, 'Output');
    outTitle.addEventListener('click', () => select(outIdx));
    outTitle.addEventListener('keydown', (e) => {
      if ((e as KeyboardEvent).key === 'Enter') select(outIdx);
    });
    const outFrozen = store.isFrozen(outIdx);
    list.append(
      h(
        'li',
        { class: `layer is-fixed${store.selected === outIdx ? ' is-selected' : ''}${outFrozen ? ' is-frozen' : ''}`, 'data-block': String(outIdx) },
        h('span', { class: 'layer-idx' }, 'OUT'),
        h(
          'div',
          { class: 'layer-main' },
          outTitle,
          h(
            'div',
            { class: 'layer-foot' },
            h('span', { class: 'layer-shape' }, 'Dense ', h('b', null, String(store.classes)), ` · softmax · ${int(out.params)} params`),
            store.valid ? freezeButton(outIdx, 'Output') : null,
          ),
        ),
        h('span'),
      ),
    );

    const blocked = convBlocked(arch);
    const addConv = h(
      'button',
      {
        type: 'button',
        class: 'btn btn-sm',
        disabled: nConv >= MAX_CONV || !!blocked,
        'aria-describedby': blocked ? 'builder-conv-why' : undefined,
        onclick: () => {
          const next = spec.slice();
          next.splice(nConv, 0, { kind: 'conv', filters: 8, kernel: 3, act: 'relu', pool: true });
          setSpec(next);
        },
      },
      '+ Conv layer',
    );
    const addDense = h(
      'button',
      {
        type: 'button',
        class: 'btn btn-sm',
        disabled: nDense >= MAX_DENSE,
        onclick: () => setSpec([...spec, { kind: 'dense', units: pointsData ? 8 : 32, act: pointsData ? 'tanh' : 'relu' }]),
      },
      '+ Dense layer',
    );

    const total = info.reduce((s, l) => s + l.params, 0);
    const frozenParams = info.reduce((s, l, i) => s + (store.isFrozen(i) ? l.params : 0), 0);
    const macs = multiplyAdds(info);
    const warnings: string[] = [];
    if (store.valid && total > BIG_PARAMS) {
      warnings.push(`${int(total)} parameters is a lot to train in a browser tab: each step is slow, and a saved model file would be about ${((total * 4 * 4) / 3 / 1e6).toFixed(1)} MB.`);
    }
    if (store.valid && macs > BIG_MACS) {
      warnings.push(`About ${(macs / 1e6).toFixed(1)} million multiply-adds per ${pointsData ? 'point' : 'image'}: expect slow training. Pooling earlier, or fewer filters in the first layers, makes it cheaper.`);
    }

    const keep = h('input', { type: 'checkbox', id: 'keep-weights', checked: store.keepWeights }) as HTMLInputElement;
    keep.addEventListener('change', () => setKeepWeights(keep.checked));

    root.append(
      presets,
      list,
      h(
        'div',
        { class: 'builder-actions' },
        addConv,
        addDense,
        blocked ? h('p', { class: 'hint builder-why', id: 'builder-conv-why' }, blocked) : null,
      ),
      h(
        'div',
        { class: 'builder-foot' },
        h('span', { class: 'label' }, 'Parameters'),
        h('span', { class: 'builder-total' }, frozenParams ? h('span', { class: 'builder-frozen' }, `${int(frozenParams)} frozen · `) : null, h('b', null, int(total))),
      ),
      ...warnings.map((w) => h('p', { class: 'notice builder-warn' }, w)),
      h(
        'div',
        { class: 'builder-keep' },
        h('label', { class: 'check', for: 'keep-weights' }, keep, 'Keep trained weights when editing'),
        h(
          'p',
          { class: 'hint' },
          store.keepWeights
            ? 'On: when you edit the layers or switch to a dataset with the same input, every layer whose shape is unchanged keeps its weights (and its freeze setting). Changed layers start from random weights.'
            : 'Off: every edit starts the whole network from random weights. Turn this on to keep the layers you did not change.',
        ),
      ),
      h(
        'p',
        { class: 'hint builder-note' },
        'Convolutions use 3×3 or 5×5 kernels, stride 1 and same padding. Editing the architecture resets the training history. A frozen layer keeps its weights while the others train.',
      ),
    );

    if (focused) document.getElementById(focused)?.focus();
  };

  render();
  store.on('model', render);
  store.on('dataset', render);
  store.on('frozen', render);
  store.on('hyper', render);
  store.on('select', () => {
    root.querySelectorAll<HTMLElement>('.layer[data-block]').forEach((li) => li.classList.toggle('is-selected', Number(li.dataset.block) === store.selected));
  });
}
