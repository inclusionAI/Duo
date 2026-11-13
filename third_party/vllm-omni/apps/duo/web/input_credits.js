'use strict';

const INPUT_WINDOW = 2;
const OUTPUT_WINDOW = 4;

function releasedInputs(s) {
  return Math.max(s.received, s.creditThrough);
}

function canSubmitInput(s) {
  return !s.resetRequested && s.sent - releasedInputs(s) < INPUT_WINDOW
    && s.sent - s.received < OUTPUT_WINDOW;
}

function connectInputCredits(s, key) {
  const url = new URL('./input-credits/' + encodeURIComponent(key), location.href);
  const ws = s.creditSocket = new WebSocket(url.href.replace(/^http/, 'ws'));
  ws.onmessage = event => {
    if (current !== s) return;
    const {through, reset} = JSON.parse(event.data);
    s.creditThrough = Math.max(s.creditThrough, Math.min(s.sent, through));
    if (reset === s.pendingReset) acceptMediaReset(s);
  };
}
