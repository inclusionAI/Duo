export const FPS = 24;
export const FRAMES = 17;
export const CHUNK_SECONDS = FRAMES / FPS;
export const $ = id => document.getElementById(id);
export const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
export const code = index => `C${String(index).padStart(3, '0')}`;
export const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
export const percent = (done, total) => total ? Math.floor(100 * done / total) : 0;
export const seconds = value => `${Number(value || 0).toFixed(3)}s`;
export const time = value => {
  const total = Math.max(0, Math.floor(Number.isFinite(value) ? value : 0));
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
};
export const fill = (id, value) => {
  const progress = clamp(value, 0, 100);
  const element = $(id);
  element.style.width = `${progress}%`;
  element.parentElement.classList.toggle('done', progress === 100);
};
export const icon = name => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
export const stageNames = {pending: 'Pending', running: 'Running', ready: 'Ready', consumed: 'Consumed', failed: 'Failed'};
export function modal(id) { const dialog = $(id); if (!dialog.open) dialog.showModal(); }
export function close(id) { if ($(id).open) $(id).close(); }
let toastTimer;
export function toast(message) {
  clearTimeout(toastTimer);
  $('toast').textContent = message;
  $('toast').hidden = false;
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 4500);
}
