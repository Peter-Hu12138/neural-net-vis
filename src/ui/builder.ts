import { select, setSpec } from '../actions';
import { describe } from '../nn/network';
import { ACTIVATIONS, fmtShape, type Act, type ConvSpec, type DenseSpec, type LayerSpec } from '../nn/types';
import { PRESETS, store } from '../store';
import { $, clear, h, int, selectField } from './dom';

const MAX_CONV = 4;
const MAX_DENSE = 3;
const FILTERS = [1, 2, 4, 6, 8, 12, 16];
const UNITS = [4, 8, 16, 32, 64, 128];
const actOptions = ACTIVATIONS.map((a) => ({ value: a.id, label: a.label }));

export function layerName(spec: LayerSpec | null, index: number): string {
  if (!spec) return 'Output';
  return `${spec.kind === 'conv' ? 'Conv' : 'Dense'} ${index + 1}`;
}

export function layerDetail(spec: LayerSpec | null): string {
  if (!spec) return '10 · softmax';
  const act = ACTIVATIONS.find((a) => a.id === spec.act)!.label;
  if (spec.kind === 'conv') return `${spec.filters} × ${spec.kernel}×${spec.kernel} · ${act}${spec.pool ? ' · pool' : ''}`;
  return `${spec.units} · ${act}`;
}

export function mountBuilder(): void {
  const root = $('builder');

  const render = () => {
    clear(root);
    const spec = store.spec;
    const info = describe(store.arch);
    const nConv = spec.filter((l) => l.kind === 'conv').length;
    const nDense = spec.length - nConv;

    const update = (i: number, patch: Partial<ConvSpec> | Partial<DenseSpec>) => {
      const next = spec.map((l, j) => (j === i ? ({ ...l, ...patch } as LayerSpec) : l));
      setSpec(next);
    };
    const remove = (i: number) => setSpec(spec.filter((_, j) => j !== i));

    const presets = h('div', { class: 'presets', role: 'group', 'aria-label': 'Presets' });
    for (const p of PRESETS) {
      presets.append(h('button', { type: 'button', class: 'btn btn-sm', onclick: () => setSpec(structuredClone(p.spec)) }, p.name));
    }

    const list = h('ol', { class: 'layers' });
    list.append(
      h(
        'li',
        { class: 'layer is-fixed' },
        h('span', { class: 'layer-idx' }, 'IN'),
        h('div', { class: 'layer-main' }, h('span', { class: 'layer-title' }, 'Input'), h('span', { class: 'layer-shape' }, 'Grey image ', h('b', null, '28×28×1'))),
        h('span'),
      ),
    );

    spec.forEach((l, i) => {
      const li = info[i];
      const fields = h('div', { class: 'layer-fields' });
      const id = (n: string) => `l${i}-${n}`;
      if (l.kind === 'conv') {
        fields.append(
          selectField(id('filters'), 'Filters', FILTERS.map((v) => ({ value: v, label: String(v) })), l.filters, (filters) => update(i, { filters })),
          selectField(id('kernel'), 'Kernel', [{ value: 3, label: '3×3' }, { value: 5, label: '5×5' }], l.kernel, (kernel) => update(i, { kernel: kernel as 3 | 5 })),
          selectField(id('act'), 'Activation', actOptions, l.act, (act) => update(i, { act: act as Act })),
        );
        const pool = h('input', { type: 'checkbox', id: id('pool'), checked: l.pool }) as HTMLInputElement;
        pool.addEventListener('change', () => update(i, { pool: pool.checked }));
        fields.append(h('label', { class: 'check', for: id('pool'), style: { alignSelf: 'end', paddingBottom: '4px' } }, pool, 'Max-pool 2×2'));
      } else {
        fields.append(
          selectField(id('units'), 'Units', UNITS.map((v) => ({ value: v, label: String(v) })), l.units, (units) => update(i, { units })),
          selectField(id('act'), 'Activation', actOptions, l.act, (act) => update(i, { act: act as Act })),
        );
      }
      const title = h('span', { class: 'layer-title', role: 'button', tabindex: '0', title: 'Inspect weights' }, layerName(l, i));
      title.addEventListener('click', () => select(i));
      title.addEventListener('keydown', (e) => {
        if ((e as KeyboardEvent).key === 'Enter') select(i);
      });
      list.append(
        h(
          'li',
          { class: `layer${store.selected === i ? ' is-selected' : ''}`, 'data-block': String(i) },
          h('span', { class: 'layer-idx' }, `L${i + 1}`),
          h(
            'div',
            { class: 'layer-main' },
            title,
            fields,
            h('span', { class: 'layer-shape' }, '→ ', h('b', null, fmtShape(li.outShape)), ` · ${int(li.params)} params`),
            li.error ? h('span', { class: 'layer-error' }, li.error) : null,
          ),
          h('button', { type: 'button', class: 'btn btn-sm btn-icon', 'aria-label': `Remove ${layerName(l, i)}`, title: 'Remove layer', onclick: () => remove(i) }, '×'),
        ),
      );
    });

    const out = info[info.length - 1];
    const outTitle = h('span', { class: 'layer-title', role: 'button', tabindex: '0', title: 'Inspect weights' }, 'Output');
    outTitle.addEventListener('click', () => select(spec.length));
    list.append(
      h(
        'li',
        { class: `layer is-fixed${store.selected === spec.length ? ' is-selected' : ''}`, 'data-block': String(spec.length) },
        h('span', { class: 'layer-idx' }, 'OUT'),
        h(
          'div',
          { class: 'layer-main' },
          outTitle,
          h('span', { class: 'layer-shape' }, 'Dense ', h('b', null, '10'), ` · softmax · ${int(out.params)} params`),
        ),
        h('span'),
      ),
    );

    const addConv = h(
      'button',
      {
        type: 'button',
        class: 'btn btn-sm',
        disabled: nConv >= MAX_CONV,
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
      { type: 'button', class: 'btn btn-sm', disabled: nDense >= MAX_DENSE, onclick: () => setSpec([...spec, { kind: 'dense', units: 32, act: 'relu' }]) },
      '+ Dense layer',
    );

    const total = info.reduce((s, l) => s + l.params, 0);
    root.append(
      presets,
      list,
      h('div', { class: 'builder-actions' }, addConv, addDense),
      h('div', { class: 'builder-foot' }, h('span', { class: 'label' }, 'Parameters'), h('b', null, int(total))),
      h('p', { class: 'hint', style: { marginTop: '10px' } }, 'Convolutions use 3×3 or 5×5 kernels, stride 1 and same padding. Changing the architecture resets training.'),
    );
  };

  render();
  store.on('model', render);
  store.on('select', () => {
    root.querySelectorAll<HTMLElement>('.layer[data-block]').forEach((li) => li.classList.toggle('is-selected', Number(li.dataset.block) === store.selected));
  });
}
