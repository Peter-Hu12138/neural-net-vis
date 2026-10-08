let el: HTMLElement | null = null;

export function showTip(text: string, clientX: number, clientY: number): void {
  el ??= document.getElementById('tip');
  if (!el) return;
  el.textContent = text;
  el.hidden = false;
  const r = el.getBoundingClientRect();
  let x = clientX + 14;
  let y = clientY + 14;
  if (x + r.width > window.innerWidth - 8) x = clientX - r.width - 14;
  if (y + r.height > window.innerHeight - 8) y = clientY - r.height - 14;
  el.style.left = `${Math.max(8, x)}px`;
  el.style.top = `${Math.max(8, y)}px`;
}

export function hideTip(): void {
  if (el) el.hidden = true;
}
