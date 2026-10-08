type Attrs = Record<string, unknown> & { class?: string; style?: Partial<CSSStyleDeclaration> };
type Child = Node | string | number | null | undefined | false;

/** Tiny element factory: h('button', { class: 'btn', onclick }, 'Label'). */
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs | null = null, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === 'class') el.className = String(v);
      else if (k === 'style') Object.assign(el.style, v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v as EventListener);
      else if (k in el && typeof v !== 'string') (el as unknown as Record<string, unknown>)[k] = v;
      else el.setAttribute(k, v === true ? '' : String(v));
    }
  }
  append(el, children);
  return el;
}

export function append(el: Element, children: Child[]): void {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : String(c));
  }
}

export function clear(el: Element): void {
  while (el.firstChild) el.removeChild(el.firstChild);
}

export const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

export function selectField<T extends string | number>(
  id: string,
  label: string,
  options: { value: T; label: string }[],
  value: T,
  onChange: (v: T) => void,
): HTMLElement {
  const sel = h('select', { id }) as HTMLSelectElement;
  for (const o of options) {
    const opt = h('option', { value: String(o.value) }, o.label) as HTMLOptionElement;
    if (o.value === value) opt.selected = true;
    sel.append(opt);
  }
  sel.addEventListener('change', () => {
    const o = options[sel.selectedIndex];
    onChange(o.value);
  });
  return h('label', { class: 'field', for: id }, h('span', { class: 'label' }, label), h('span', { class: 'select' }, sel));
}

export function segmented<T extends string>(
  options: { value: T; label: string }[],
  value: T,
  onChange: (v: T) => void,
  ariaLabel: string,
): HTMLElement {
  const wrap = h('div', { class: 'seg', role: 'group', 'aria-label': ariaLabel });
  for (const o of options) {
    const b = h('button', { type: 'button', 'aria-pressed': String(o.value === value) }, o.label);
    b.addEventListener('click', () => {
      for (const x of Array.from(wrap.children)) x.setAttribute('aria-pressed', 'false');
      b.setAttribute('aria-pressed', 'true');
      onChange(o.value);
    });
    wrap.append(b);
  }
  return wrap;
}

export function digitChips(selected: number | null, onPick: (d: number) => void, label = 'Label'): HTMLElement {
  const wrap = h('div', { class: 'chips', role: 'group', 'aria-label': label });
  for (let d = 0; d < 10; d++) {
    const b = h('button', { type: 'button', class: 'chip', 'aria-pressed': String(selected === d) }, String(d));
    b.addEventListener('click', () => {
      for (const x of Array.from(wrap.children)) x.setAttribute('aria-pressed', 'false');
      b.setAttribute('aria-pressed', 'true');
      onPick(d);
    });
    wrap.append(b);
  }
  return wrap;
}

export { fmt, int, pct } from './format';
