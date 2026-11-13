// Server-owned identity, FIFO turns and recordings. Playback remains local.
export const IDLE_MS = 5 * 60 * 1000;
export const IDLE_WARNING_MS = 30 * 1000;
const KEY = 'duo.video.member.v2';
const uid = () => crypto.randomUUID();
const endpoint = path => new URL(`..${path}`, location.href);

function adaptState(state, schema = {}) {
  const stages = {intent: 'intent', realization: 'realization', codec: 'codec', ...schema.stages};
  const fields = {
    realizationSteps: 'realizationSteps', intentRows: 'intentRows', conditionSource: 'conditionSource',
    historyWindow: 'historyWindow', ...schema.fields,
  };
  const values = {baseCondition: 'base', ...schema.values};
  const chunks = (state.session?.chunks || []).map(chunk => ({
    ...chunk,
    intent: chunk[stages.intent],
    realization: chunk[stages.realization],
    codec: chunk[stages.codec],
    realizationSteps: chunk[fields.realizationSteps],
    intentRows: chunk[fields.intentRows],
    conditionSource: chunk[fields.conditionSource] === values.baseCondition ? 'base' : 'adapted',
    historyWindow: chunk[fields.historyWindow],
  }));
  const compatibilityMode = Object.entries(stages).some(([name, source]) => name !== source);
  const requests = compatibilityMode ? (state.session?.requests || []).map(request => ({
    ...request,
    error: request.error ? 'Request processing failed. Check the backend logs.' : null,
    notice: request.notice ? 'The backend reported a request update.' : null,
  })) : state.session?.requests || [];
  return {
    ...state,
    error: compatibilityMode && state.error ? 'Inference service error' : state.error,
    detail: compatibilityMode ? state.error ? 'Inference service error' : state.ready ? 'Ready' : 'Loading model' : state.detail,
    removals: compatibilityMode ? (state.removals || []).map(item => ({...item, reason: 'Session ended'})) : state.removals,
    session: {...state.session, chunks, requests},
  };
}

export class ServerClient extends EventTarget {
  constructor(schema = {}) {
    super();
    this.schema = schema;
    let saved;
    try { saved = JSON.parse(sessionStorage.getItem(KEY) || 'null'); } catch { /* Old tab data. */ }
    this.memberId = saved?.memberId || null;
    this.token = saved?.token || '';
    this.clientId = uid();
    this.connectionId = uid();
    this.connected = false;
    this.detached = false;
    this.lastTouch = 0;
    this.clockOffset = 0;
    this.mediaVersion = 0;
    this.state = {revision: 0, members: [], removals: [], activeId: null, turn: null, ready: false,
      config: {modelId: 'duo', caseId: 'case', descriptions: {}, originalPrompt: false,
        waitReuse: false, waitThreshold: 3, waitKVWindow: false},
      session: {id: uid(), requests: [], chunks: [], capture: null, media: null, demo: null, phase: 'idle'}};
    this.channel = typeof BroadcastChannel === 'function' ? new BroadcastChannel('duo-video-members-v2') : null;
    this.channel?.addEventListener('message', ({data}) => {
      if (data.kind === 'probe' && !this.detached && data.memberId === this.memberId && data.clientId !== this.clientId) {
        this.channel.postMessage({kind: 'taken', clientId: data.clientId});
      } else if (data.kind === 'taken' && this.checkingTab && data.clientId === this.clientId) {
        this.clearIdentity();
      }
    });
  }

  headers() { return {'Authorization': `Bearer ${this.token}`, 'X-Connection-ID': this.connectionId}; }
  member(state = this.state) { return state.members.find(member => member.id === this.memberId); }
  canOperate() { return Boolean(this.connected && !this.detached && this.state.ready && this.member() && this.state.activeId === this.memberId && this.state.turn?.acceptedAt); }
  canAdmin() { return this.canOperate() && this.member()?.role === 'admin'; }
  canModerate() { return this.connected && !this.detached && this.member()?.role === 'admin'; }
  now() { return Date.now() + this.clockOffset; }
  clearIdentity() { this.token = ''; this.memberId = null; sessionStorage.removeItem(KEY); }

  publish(state) {
    state = adaptState(state, this.schema);
    if (state.revision < this.state.revision) return;
    const member = state.members.find(member => member.id === this.memberId);
    if (member && member.connectionId !== this.connectionId) {
      this.controller?.abort();
      this.connected = false;
      this.clearIdentity();
    }
    this.clockOffset = state.serverTime - Date.now();
    this.state = state;
    this.dispatchEvent(new Event('change'));
  }

  async decode(response) {
    const value = await response.json();
    if (!response.ok) {
      const error = new Error(value.error || 'The service is temporarily unavailable. Try again.');
      error.status = response.status;
      throw error;
    }
    return value;
  }

  async request(path, data = {}) {
    const response = await fetch(endpoint(path), {method: 'POST', headers: {...this.headers(), 'Content-Type': 'application/json'},
      body: JSON.stringify({sessionId: this.state.session.id, ...data})});
    const value = await this.decode(response);
    if (value.state) this.publish(value.state);
    return value.result;
  }

  async refresh() {
    const url = endpoint('/api/video/state');
    url.searchParams.set('version', this.mediaVersion);
    const state = await this.decode(await fetch(url, {headers: this.headers(), cache: 'no-store'}));
    this.connected = Boolean(this.token && this.member(state));
    this.publish(state);
  }

  async connect() {
    if (this.memberId && this.channel) {
      this.checkingTab = true;
      this.channel.postMessage({kind: 'probe', memberId: this.memberId, clientId: this.clientId});
      await new Promise(resolve => setTimeout(resolve, 180));
      this.checkingTab = false;
    }
    if (this.token) await this.resume();
    else await this.refresh();
    this.poll = setInterval(() => {
      if (!this.detached) this.refresh().catch(() => {});
    }, 5000);
    return this;
  }

  async stream() {
    if (this.detached || !this.token || this.controller) return;
    const controller = new AbortController();
    this.controller = controller;
    try {
      const response = await fetch(endpoint('/api/video/events'), {headers: this.headers(), signal: controller.signal, cache: 'no-store'});
      if (!response.ok) await this.decode(response);
      this.connected = true;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = '';
      while (!this.detached) {
        const {value, done} = await reader.read();
        if (done) break;
        pending += decoder.decode(value, {stream: true});
        let end;
        while ((end = pending.indexOf('\n\n')) >= 0) {
          const message = pending.slice(0, end);
          pending = pending.slice(end + 2);
          if (message.startsWith('data: ')) this.publish(JSON.parse(message.slice(6)));
        }
        if (!this.member()) break;
      }
      await reader.cancel();
    } catch (error) {
      if (error.status === 401 || error.status === 409) await this.refresh().catch(() => {});
    } finally {
      if (this.controller === controller) {
        this.controller = null;
        if (!this.detached && this.member()) this.reconnect = setTimeout(() => this.stream(), 1000);
      }
    }
  }

  async join(name, role, password) {
    const response = await fetch(endpoint('/api/video/join'), {method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Connection-ID': this.connectionId},
      body: JSON.stringify({name, role, password})});
    const value = await this.decode(response);
    this.memberId = value.result.memberId;
    this.token = value.result.token;
    sessionStorage.setItem(KEY, JSON.stringify(value.result));
    this.publish(value.state);
    this.stream();
  }

  async leave() {
    await this.request('/api/video/leave');
    this.controller?.abort();
    this.clearIdentity();
    this.dispatchEvent(new Event('change'));
  }

  removeMember(memberId) { return this.request('/api/video/kick', {memberId}); }
  acceptTurn() { return this.request('/api/video/accept', {turnId: this.state.turn?.id}); }
  heartbeat() { return this.refresh(); }
  touch(force = false) {
    if (!this.canOperate() || (!force && Date.now() - this.lastTouch < 8000)) return Promise.resolve();
    this.lastTouch = Date.now();
    return this.request('/api/video/touch');
  }

  suspend() {
    if (this.detached) return;
    this.detached = true;
    clearTimeout(this.reconnect);
    this.controller?.abort();
    if (this.token) navigator.sendBeacon(endpoint('/api/video/depart'),
      new Blob([JSON.stringify({token: this.token, connectionId: this.connectionId})], {type: 'application/json'}));
  }

  async resume() {
    this.detached = false;
    clearTimeout(this.reconnect);
    this.controller?.abort();
    this.controller = null;
    this.connectionId = uid();
    if (!this.token) { await this.refresh(); return; }
    try {
      await this.request('/api/video/resume', {connectionId: this.connectionId});
      this.stream();
    } catch (error) {
      if (error.status !== 401) throw error;
      await this.refresh();
      this.clearIdentity();
    }
  }

  setOption(key, value) {
    if (key === 'originalPrompt') return this.request('/api/global-prompt', {enabled: value});
    if (key === 'waitKVWindow') return this.request('/api/wait-kv-window', {enabled: value});
    if (key === 'waitReuse' || key === 'waitThreshold') return this.request('/api/wait-reuse', {
      enabled: key === 'waitReuse' ? value : this.state.config.waitReuse,
      threshold: key === 'waitThreshold' ? value : this.state.config.waitThreshold});
    throw new Error('Invalid setting.');
  }
  applySettings(patch) {
    return patch.modelId ? this.request('/api/model', {model_id: patch.modelId}) :
      this.request('/api/case', {case_id: patch.caseId, description: patch.description});
  }
  reset(all) { return this.request('/api/reset', {mode: all ? 'all' : 'cache'}); }
  beginCapture() { return this.request('/api/video/capture'); }
  cancelCapture(capture) {
    if (!capture || !this.member()) return Promise.resolve();
    return this.request('/api/video/cancel-capture', {captureId: capture.id, sessionId: capture.sessionId});
  }
  async saveRecording(blob, metadata, capture) {
    const url = endpoint('/api/video/upload');
    url.searchParams.set('capture', capture.id);
    url.searchParams.set('session', capture.sessionId);
    const response = await fetch(url, {method: 'POST', body: blob, headers: {...this.headers(),
      'Content-Type': blob.type || metadata.mimeType, 'X-Recording-Metadata': JSON.stringify(metadata)}});
    const value = await this.decode(response);
    this.publish(value.state);
    return `R${String(value.result.id).padStart(3, '0')}`;
  }
  async readRecording(id) {
    const request = this.state.session.requests.find(request => request.recordingId === id);
    if (!request?.recordingURL) return null;
    const response = await fetch(request.recordingURL, {headers: this.headers()});
    if (!response.ok) throw new Error('This recording is no longer available.');
    return response.blob();
  }
}
