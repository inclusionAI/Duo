// Browser-only transport for layout and interaction review. It never runs a model.
export const IDLE_MS = 5 * 60 * 1000;
export const IDLE_WARNING_MS = 30 * 1000;

const BLOCK_SECONDS = 17 / 24;
const uid = () => crypto.randomUUID();
const copy = value => structuredClone(value);

export class PreviewClient extends EventTarget {
  constructor(catalog) {
    super();
    this.catalog = catalog;
    this.memberId = null;
    this.connected = true;
    this.detached = false;
    this.mediaVersion = 0;
    this.recordings = new Map();
    this.state = this.initialState();
  }

  initialState() {
    return {
      revision: 0,
      serverTime: Date.now(),
      members: [],
      removals: [],
      activeId: null,
      turn: null,
      ready: true,
      detail: 'Frontend preview · no model inference',
      error: null,
      gpus: [],
      context: {fixedPrefix: 'Preview intent context', sharedVariants: {}},
      config: {
        modelId: this.catalog.models[0].id,
        caseId: this.catalog.cases[0].id,
        descriptions: {},
        originalPrompt: false,
        waitReuse: false,
        waitThreshold: 3,
        waitKVWindow: false,
      },
      session: this.emptySession(),
    };
  }

  emptySession() {
    return {id: uid(), requests: [], chunks: [], capture: null, media: null, phase: 'idle', cacheResets: 0};
  }

  publish() {
    this.state.revision += 1;
    this.state.serverTime = Date.now();
    this.dispatchEvent(new Event('change'));
  }

  member() { return this.state.members.find(member => member.id === this.memberId); }
  canOperate() { return Boolean(this.member() && this.state.activeId === this.memberId && this.state.turn?.acceptedAt); }
  canAdmin() { return this.canOperate() && this.member()?.role === 'admin'; }
  canModerate() { return this.canAdmin(); }
  now() { return Date.now(); }
  connect() { this.publish(); return Promise.resolve(this); }
  resume() { this.detached = false; this.connected = true; this.publish(); return Promise.resolve(); }
  suspend() { this.detached = true; this.connected = false; }
  touch() {
    if (this.state.turn) this.state.turn.lastActivityAt = Date.now();
    this.publish();
    return Promise.resolve();
  }

  join(name, role) {
    name = name.trim();
    if (!name || name.length > 24) return Promise.reject(new Error('Enter a display name with 1–24 characters.'));
    const member = {id: uid(), name, role, connectionId: uid()};
    this.memberId = member.id;
    this.state.members = [member];
    this.state.activeId = member.id;
    this.state.turn = {id: uid(), offeredAt: Date.now(), acceptedAt: null, lastActivityAt: Date.now()};
    this.publish();
    return Promise.resolve();
  }

  acceptTurn() {
    this.state.turn.acceptedAt = Date.now();
    this.state.turn.lastActivityAt = Date.now();
    this.publish();
    return Promise.resolve();
  }

  leave() {
    this.state.members = [];
    this.state.activeId = null;
    this.state.turn = null;
    this.state.session = this.emptySession();
    this.memberId = null;
    this.publish();
    return Promise.resolve();
  }

  removeMember() { return Promise.resolve(null); }
  heartbeat() { this.publish(); return Promise.resolve(); }

  setOption(key, value) {
    this.state.config[key] = value;
    this.publish();
    return Promise.resolve();
  }

  applySettings(patch) {
    if (patch.modelId) this.state.config.modelId = patch.modelId;
    if (patch.caseId) this.state.config.caseId = patch.caseId;
    if (patch.description !== undefined) this.state.config.descriptions[patch.caseId] = patch.description;
    this.state.session = this.emptySession();
    this.publish();
    return Promise.resolve();
  }

  reset(all) {
    if (all) this.state.session = this.emptySession();
    else this.state.session.cacheResets += 1;
    this.publish();
    return Promise.resolve();
  }

  beginCapture() {
    const capture = {id: uid(), sessionId: this.state.session.id, ownerId: this.memberId, startedAt: Date.now()};
    this.state.session.capture = capture;
    this.state.session.phase = 'recording';
    this.publish();
    return Promise.resolve(copy(capture));
  }

  cancelCapture(capture) {
    if (this.state.session.capture?.id === capture?.id) {
      this.state.session.capture = null;
      this.state.session.phase = this.state.session.requests.at(-1)?.state || 'idle';
      this.publish();
    }
    return Promise.resolve();
  }

  saveRecording(blob, metadata, capture) {
    if (this.state.session.capture?.id !== capture.id) return Promise.reject(new Error('This recording is no longer active.'));
    const id = `R${String(this.state.session.requests.length + 1).padStart(3, '0')}`;
    const request = {
      id,
      recordingId: capture.id,
      author: this.member().name,
      createdAt: Date.now(),
      state: 'queued',
      recording: {...metadata, size: blob.size},
      transcript: '',
      asr: {progress: 0, segments: []},
      build: 0,
      chunks: [],
      sequenceURL: null,
    };
    this.recordings.set(capture.id, blob);
    this.state.session.capture = null;
    this.state.session.requests.push(request);
    this.state.session.phase = 'queued';
    this.publish();
    this.simulate(request);
    return Promise.resolve(id);
  }

  readRecording(id) { return Promise.resolve(this.recordings.get(id) || null); }

  simulate(request) {
    setTimeout(() => {
      request.state = 'asr';
      request.asr.progress = 55;
      this.state.session.phase = 'asr';
      this.publish();
    }, 300);
    setTimeout(() => {
      request.asr = {progress: 100, segments: [{start: 0, end: request.recording.duration, text: '<user-video>'}]};
      request.state = 'building';
      request.build = 45;
      this.state.session.phase = 'building';
      this.publish();
    }, 900);
    setTimeout(() => {
      const count = Math.max(1, Math.min(12, Math.ceil(request.recording.duration / BLOCK_SECONDS)));
      const start = this.state.session.chunks.length;
      for (let local = 0; local < count; local += 1) {
        const global = start + local;
        const sourceStart = Math.min(local * BLOCK_SECONDS, Math.max(0, request.recording.duration - BLOCK_SECONDS));
        const chunk = {
          global, local, requestId: request.id, text: local < count - 1 ? '<wait>' : 'Preview only',
          timelineStart: global * BLOCK_SECONDS, timelineEnd: (global + 1) * BLOCK_SECONDS,
          intent: 'ready', realization: local === 0 ? 'running' : 'pending', codec: 'pending',
          step: 0, realizationSteps: 4, intentRows: [], conditionSource: 'adapted', historyWindow: 14,
          user: {recordingId: request.recordingId, kind: 'real', sourceChunk: global, text: '<user-video>',
            sourceStart, sourceEnd: Math.min(request.recording.duration, sourceStart + BLOCK_SECONDS),
            playbackStart: sourceStart, playbackEnd: Math.min(request.recording.duration, sourceStart + BLOCK_SECONDS),
            frameStart: Math.round(sourceStart * 24), frameEnd: Math.round((sourceStart + BLOCK_SECONDS) * 24),
            fps: 24, hasAudio: true},
        };
        this.state.session.chunks.push(chunk);
        request.chunks.push(global);
      }
      request.build = 100;
      request.inputChunks = count;
      request.state = 'running';
      this.state.session.phase = 'running';
      this.publish();
      this.advance(request, 0);
    }, 1500);
  }

  advance(request, local) {
    const chunks = this.state.session.chunks.filter(chunk => chunk.requestId === request.id);
    if (local >= chunks.length) {
      request.state = 'completed';
      this.state.session.phase = 'completed';
      this.publish();
      return;
    }
    const chunk = chunks[local];
    chunk.realization = 'running';
    chunk.step = 2;
    this.publish();
    setTimeout(() => {
      chunk.step = chunk.realizationSteps;
      chunk.realization = 'ready';
      chunk.codec = 'ready';
      this.publish();
      this.advance(request, local + 1);
    }, 450);
  }
}
