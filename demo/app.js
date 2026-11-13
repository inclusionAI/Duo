import {ServerClient, IDLE_MS, IDLE_WARNING_MS} from './client.js';
import {PreviewClient} from './preview.js';
import {CameraCapture} from './capture.js';
import {DualVideoPlayer} from './player.js';
import {renderPeople, renderHeader, renderTracks, renderProgress, currentRequest} from './render.js';
import {$, FPS, FRAMES, CHUNK_SECONDS, close, code, esc, modal, seconds, stageNames, time, toast} from './ui.js';

let client, catalog, camera, player, view;
let selectedRequest = null, selectedChunk = null, follow = true;
let lastSessionId = null, lastViewId = null, lastRendered = '', lastTracks = '', previousMember = null;
let activeTurnId = null, noticeShown = false, leaving = false, loginBusy = false, captureBusy = false;
let uploadBusy = false;
let detailSpec = null, settingsKind = null, settingsDraft = null, idleExpiring = false;
let recordingLookup = 0;
let mediaGeneration = 0;
const blobURLs = new Map();
const sequences = new Map();
let sequenceLoading = false;

function fail(error) { toast(error.message || String(error)); }
async function action(operation) {
  try { return await operation(); } catch (error) { fail(error); return undefined; }
}

function revokeMedia() {
  ++mediaGeneration;
  for (const url of blobURLs.values()) URL.revokeObjectURL(url);
  blobURLs.clear();
  sequences.clear();
  closeRecording();
}

async function recordingSource(id) {
  if (blobURLs.has(id)) return blobURLs.get(id);
  const generation = mediaGeneration;
  const blob = await client.readRecording(id);
  if (!blob || generation !== mediaGeneration) return null;
  if (blobURLs.has(id)) return blobURLs.get(id);
  const url = URL.createObjectURL(blob);
  blobURLs.set(id, url);
  return url;
}

function setFollow(value) {
  follow = value;
  $('follow-stream').setAttribute('aria-pressed', String(value));
  $('follow-stream').classList.toggle('active', value);
  player?.setFollow(value);
  if (value) $('timeline-scroll').scrollLeft = $('timeline-scroll').scrollWidth;
}

function positionChanged(index, at) {
  document.querySelectorAll('[data-cell]').forEach(element => {
    const active = Number(element.dataset.cell) === index;
    element.classList.toggle('playing', active);
    element.classList.toggle('selected', Number(element.dataset.cell) === selectedChunk);
    if (active) element.style.setProperty('--play-progress', `${100 * ((at % CHUNK_SECONDS) / CHUNK_SECONDS)}%`);
  });
}

function permissions() {
  const allowed = client.canOperate();
  document.querySelectorAll('[data-operate]').forEach(button => { button.disabled = !allowed; });
  document.querySelectorAll('[data-remove-member]').forEach(button => { button.disabled = !client.canModerate(); });
  updateSettingsApply();
  const description = $('case-description');
  if (description) description.readOnly = !client.canAdmin();
  const restore = $('restore-description');
  if (restore) restore.disabled = !client.canAdmin();
  if (settingsKind) $('settings-access').textContent = client.canAdmin() ? 'Administrators can edit these settings. Changes begin a new media stream.' : client.member()?.role === 'user' ? 'The active model and case are available in read-only mode.' : 'You can inspect settings while observing and edit them when your turn begins.';
  renderCamera();
}

function updateSettingsApply() {
  const config = client.state.config;
  let changed = false;
  if (settingsKind === 'model' && settingsDraft) changed = settingsDraft.modelId !== config.modelId;
  if (settingsKind === 'case' && settingsDraft) {
    const item = catalog.cases.find(item => item.id === settingsDraft.caseId);
    const description = $('case-description')?.value.trim() || '';
    changed = Boolean(description) && (settingsDraft.caseId !== config.caseId || description !== (config.descriptions[item.id] || item.description));
  }
  $('settings-apply').disabled = !client.canAdmin() || !changed;
}

function renderCamera() {
  if (!camera || !client) return;
  const canRecord = client.canOperate();
  const recording = camera.recording;
  $('camera-placeholder').hidden = camera.ready;
  $('camera-message').textContent = camera.error || (camera.pending ? 'Allow camera and microphone access…' : 'Enable the camera to begin');
  $('camera-retry').disabled = camera.pending;
  $('camera-retry').textContent = camera.pending ? 'Waiting for permission…' : 'Enable camera';
  $('camera-badge').textContent = recording ? 'Recording' : 'Local preview';
  $('camera-status').textContent = camera.error && camera.ready ? 'Recording is unavailable in this browser' : camera.ready ? 'Camera and microphone ready' : camera.pending ? 'Waiting for device permission' : 'No device connected';
  const settings = camera.settings;
  $('camera-format').textContent = camera.ready ? `${settings.width || '—'}×${settings.height || '—'}` : '—';
  $('recording-clock').hidden = !recording;
  $('recording-clock').querySelector('span').textContent = time(camera.duration);
  $('record-toggle').classList.toggle('is-recording', recording);
  $('record-toggle').disabled = !recording && (!canRecord || captureBusy || camera.saving || !camera.ready || typeof MediaRecorder !== 'function' || Boolean(camera.recorder));
  $('upload-video').disabled = !canRecord || captureBusy || uploadBusy || camera.saving || recording;
  $('upload-title').textContent = uploadBusy ? 'Uploading…' : '↑ Upload local video';
  $('record-label').textContent = recording ? 'Stop recording' : camera.saving ? 'Uploading…' : captureBusy ? 'Preparing…' : 'Start recording';
  $('record-hint').textContent = !client.connected && client.member() ? 'Reconnecting…' : client.state.activeId !== client.memberId ? 'Read-only observer · recording unlocks on your turn' : !client.state.ready ? client.state.detail || 'Preparing the session…' : camera.error ? camera.error : recording ? 'Recording video and audio · stop to submit this segment' : 'Stop to upload and align the video while keeping the live preview active';
  const latest = client.state.session.requests.at(-1);
  $('recording-summary').hidden = !latest;
  $('recording-result').textContent = latest ? `${latest.id} · ${time(latest.recording.duration)} · uploaded` : '';
}

function localVideo(file) {
  const extension = file.name.split('.').pop()?.toLowerCase();
  const mimeType = file.type === 'video/mp4' || file.type === 'video/webm' ? file.type :
    extension === 'mp4' ? 'video/mp4' : extension === 'webm' ? 'video/webm' : '';
  if (!mimeType) throw new Error('Only MP4 and WebM videos are supported.');
  if (!file.size || file.size > 512 * 1024 * 1024) throw new Error('Video size must be between 1 byte and 512 MiB.');
  const blob = file.type === mimeType ? file : file.slice(0, file.size, mimeType);
  // The server probes the file, including MediaRecorder WebM without a duration header.
  if (client instanceof ServerClient) return Promise.resolve({blob, metadata: {mimeType}});
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    const url = URL.createObjectURL(blob);
    const done = () => { URL.revokeObjectURL(url); video.removeAttribute('src'); video.load(); };
    video.preload = 'metadata';
    video.onloadedmetadata = () => {
      const duration = Number.isFinite(video.duration) ? video.duration : 0;
      const metadata = {duration, mimeType, width: video.videoWidth || 0, height: video.videoHeight || 0, fps: 0};
      done();
      if (!duration) reject(new Error('The video duration could not be read. Try another file.'));
      else resolve({blob, metadata});
    };
    video.onerror = () => { done(); reject(new Error('The video could not be read. Check that the file is complete.')); };
    video.src = url;
  });
}

function closeSessionDialogs() {
  for (const id of ['notice-dialog', 'turn-dialog', 'idle-dialog', 'settings-dialog', 'detail-dialog']) close(id);
  closeRecording();
  settingsKind = null;
  detailSpec = null;
}

function loggedOut(reason) {
  previousMember = null;
  noticeShown = false;
  activeTurnId = null;
  camera.close();
  player.clear();
  revokeMedia();
  closeSessionDialogs();
  $('dashboard').inert = true;
  $('login-reason').textContent = reason || '';
  $('login-reason').hidden = !reason;
  $('admin-password').value = '';
  modal('login-dialog');
  $('login-name').focus();
}

function sessionChanged() {
  if (!catalog || !client) return;
  const state = client.state;
  if (state.config.cases?.length) catalog.cases = state.config.cases;
  const {me, active, position} = renderPeople(state, client.memberId);
  if (lastSessionId !== state.session.id) {
    lastSessionId = state.session.id;
    lastViewId = null;
    lastRendered = '';
    lastTracks = '';
    selectedRequest = null;
    selectedChunk = null;
    camera.abortRecording();
    player.clear();
    revokeMedia();
    closeSessionDialogs();
    setFollow(true);
  }
  if (!me) {
    const removal = state.removals?.find(item => item.memberId === client.memberId);
    if (previousMember) loggedOut(leaving ? 'You left the session.' : removal?.reason || 'The session expired. Please join again.');
    else if (!loginBusy) { $('dashboard').inert = true; modal('login-dialog'); }
    renderView();
    return;
  }
  if (!previousMember) { previousMember = me.id; noticeShown = false; }
  close('login-dialog');
  $('dashboard').inert = false;
  if (state.activeId === me.id) {
    close('notice-dialog');
    noticeShown = true;
    if (!state.turn.acceptedAt) {
      if (activeTurnId !== state.turn.id) {
        close('settings-dialog'); close('detail-dialog'); closeRecording();
        $('turn-copy').textContent = `${me.name}, it is your turn. ${me.role === 'user' ? 'The configured Duo model and case are ready.' : 'You can choose a model and case before starting.'}`;
        activeTurnId = state.turn.id;
      }
      modal('turn-dialog');
    } else close('turn-dialog');
  } else {
    close('turn-dialog'); close('idle-dialog');
    if (!noticeShown) {
      $('notice-copy').textContent = `${active?.name || 'The current user'} is active. You are number ${position} in the queue.`;
      modal('notice-dialog');
      noticeShown = true;
    }
  }
  renderView();
  permissions();
}

function renderView() {
  if (!catalog || !client) return;
  const state = client.state;
  view = {...state.session, example: false, now: client.now()};
  if (lastViewId !== view.id) {
    lastViewId = view.id;
    lastTracks = '';
    lastRendered = '';
    selectedRequest = null;
    selectedChunk = null;
    close('detail-dialog');
    detailSpec = null;
  }
  renderHeader(state, catalog, view, client.canOperate());
  const signature = JSON.stringify([view.requests, view.chunks, view.capture, view.phase]);
  const description = state.config.descriptions[state.config.caseId] || catalog.cases.find(item => item.id === state.config.caseId)?.description || '';
  const trackSignature = JSON.stringify([view.chunks, view.phase, description]);
  if (lastTracks !== trackSignature) { renderTracks(view, follow, description); lastTracks = trackSignature; }
  if (lastRendered !== signature || view.capture) {
    renderProgress(view, selectedRequest);
    lastRendered = signature;
  }
  player.update(view);
  client.mediaVersion = player.loaded?.version || 0;
  if (detailSpec && $('detail-dialog').open) updateDetails();
  renderCamera();
  $('live-detail').textContent = state.error || (!client.connected && client.member() ? 'Reconnecting…' : state.detail || 'Waiting for service');
}

function systemText() {
  return client.state.context?.fixedPrefix || 'Preparing the fixed intent context…';
}

function detailContent(spec) {
  const config = client.state.config;
  const item = catalog.cases.find(item => item.id === config.caseId);
  const request = spec.requestId ? view.requests.find(item => item.id === spec.requestId) : currentRequest(view, selectedRequest);
  const note = request?.error || request?.notice || '';
  if (spec.kind === 'system') return {title: `Intent context · ${item.label}`, text: systemText(),
    note: 'Fixed intent context. User evidence is added from the aligned recording.', filename: 'intent-context.txt'};
  if (spec.kind === 'shared') {
    const source = config.originalPrompt ? 'original' : 'finetuned';
    const shared = client.state.context?.sharedVariants?.[source];
    return {title: 'Shared intent condition · P', text: `Condition source: ${config.originalPrompt ? 'base intent' : 'adapted intent'}\nState rows: ${shared?.rows ?? 'pending'}\n\n${shared?.groups?.map(group => `${group.section} · ${group.rows} rows\n${group.text}`).join('\n\n') || 'Waiting for backend data'}\n\nLater chunks receive intent states selected at each boundary.`,
      note: 'The shared condition remains resident while chunk conditions enter the generation queue.', filename: 'shared-condition.txt'};
  }
  if (spec.kind === 'user-inputs') return {title: 'User video track · input and reuse',
    text: view.chunks.length ? view.chunks.map(chunk => `${code(chunk.global)} · ${chunk.user.recordingId}\n${chunk.user.kind === 'reused' ? 'Reused from ' + code(chunk.user.sourceChunk) : 'Recorded input'} · ${seconds(chunk.user.sourceStart)} – ${seconds(chunk.user.sourceEnd)}\nUser: ${chunk.user.text || '<blank>'}`).join('\n\n') : 'Recording → speech alignment → chunk alignment → user video track',
    note: 'A reused visual chunk does not replay its original speech audio.', filename: 'user-video-track.txt'};
  if (spec.kind === 'asr') return {title: `${request?.id || ''} · Speech alignment`,
    text: request ? `Recording: ${seconds(request.recording.duration)}\nTranscript: ${request.transcript || 'pending'}\n\nSpeech segments\n${request.asr.segments.map((segment, index) => `${index + 1}. ${seconds(segment.start)} – ${seconds(segment.end)}\n${segment.text}`).join('\n\n') || request.asr.error || 'Waiting for recognition'}` : 'Waiting for input', note, filename: 'speech-segments.txt'};
  if (spec.kind === 'sequence') return {title: `${request?.id || ''} · intent_sequence.txt`,
    text: sequences.get(request?.id)?.text || (request?.sequenceURL ? 'Reading the intent sequence…' : 'Waiting for speech and sequence alignment.'),
    note: 'Decoded tokens sent to the intent and updated during causal execution.', filename: 'intent_sequence.txt'};
  if (spec.kind === 'recording') return {title: `${request?.id || ''} · Recording details`,
    text: request ? `Submitted by: ${request.author}\nRecording ID: ${request.recordingId}\nDuration: ${seconds(request.recording.duration)}\nResolution: ${request.recording.width || '—'} × ${request.recording.height || '—'}\nFrame rate: ${Number(request.recording.fps || 0).toFixed(2)}\nFormat: ${request.recording.mimeType}\nSize: ${(request.recording.size / 1024 / 1024).toFixed(2)} MiB\nStorage: temporary backend storage` : 'Waiting for input', note, filename: 'recording-info.txt'};
  const chunk = view.chunks.find(item => item.global === spec.index);
  if (!chunk) return {title: 'Chunk details', text: 'This chunk has not arrived yet.', note, filename: 'chunk.txt'};
  const mapping = chunk.user;
  const source = mapping.kind === 'reused' ? `Full segment reused from ${code(mapping.sourceChunk)}` : 'Recorded input';
  const title = `${code(chunk.global)} · ${spec.track === 'user' ? 'User video' : spec.track === 'assistant' ? 'Assistant video' : ({intent: 'Intent states', realization: 'Realization', codec: 'Media decode'}[spec.track] || 'Processing details')}`;
  const text = `Global chunk: ${code(chunk.global)}\nRequest: ${chunk.requestId}\nMedia time: ${seconds(chunk.timelineStart)} – ${seconds(chunk.timelineEnd)}\nGenerated frames: [${chunk.global * FRAMES}, ${(chunk.global + 1) * FRAMES}) · ${FPS} FPS\n\nUser\nText: ${mapping.text || '<blank>'}\nVisual source: ${source}\nRecording: ${mapping.recordingId}\nSource interval: [${seconds(mapping.sourceStart)}, ${seconds(mapping.sourceEnd)})\nSource frames: [${mapping.frameStart}, ${mapping.frameEnd}) · ${mapping.fps} FPS\nPlayback interval: [${seconds(mapping.playbackStart)}, ${seconds(mapping.playbackEnd)})\n\nAssistant\nContent: ${chunk.text || 'Waiting for intent execution'}\nToken IDs: ${JSON.stringify(chunk.token_ids || [])}\nIntent rows: ${JSON.stringify(chunk.intentRows || [])}\nIntent: ${stageNames[chunk.intent]}\nRealization: ${stageNames[chunk.realization]} · ${chunk.step} / ${chunk.realizationSteps} steps\nCodec: ${stageNames[chunk.codec]}\n\nRealization conditions\nShared condition: ${chunk.conditionSource === 'base' ? 'base intent' : 'adapted intent'}\nChunk condition: intent execution\nWAIT reuse: ${chunk.waitReuseSource != null ? code(chunk.waitReuseSource) : 'not reused'}\nCommitted history window: ${chunk.historyWindow || 14} chunks`;
  const gate = chunk.userSpeechGate;
  const gateText = gate?.policy ? `\n\nTurn-taking gate\nPolicy: ${gate.policy}\nFirst utterance ends at: ${gate.firstUserEndBlock == null ? '—' : code(gate.firstUserEndBlock)}\nGeneration releases at: ${gate.releaseBlock == null ? '—' : code(gate.releaseBlock)}\nCurrent chunk: ${gate.applied ? 'forced <wait> using the executed intent state' : 'normal intent execution'}\nDecision source: ${gate.decisionSource || 'pending'}` : '';
  return {title, text: text + gateText, note, filename: `${code(chunk.global)}-${spec.track || 'chunk'}.txt`};
}

async function refreshSequence(request) {
  if (!request?.sequenceURL || sequenceLoading || (sequences.has(request.id) && sequences.get(request.id).version === request.sequenceVersion)) return;
  sequenceLoading = true;
  const sessionId = view.id;
  try {
    const response = await fetch(request.sequenceURL, {cache: 'no-store'});
    if (!response.ok) throw new Error('The intent sequence is not available yet. Try again shortly.');
    const text = await response.text();
    if (view.id === sessionId) sequences.set(request.id, {text, version: request.sequenceVersion});
  } catch (error) {
    if (view.id === sessionId) sequences.set(request.id, {text: error.message, version: request.sequenceVersion});
  }
  finally { sequenceLoading = false; }
}

function updateDetails() {
  if (!detailSpec) return;
  if (detailSpec.kind === 'sequence') refreshSequence(view.requests.find(request => request.id === detailSpec.requestId));
  const data = detailContent(detailSpec);
  $('dialog-title').textContent = data.title;
  if ($('dialog-content').textContent !== data.text) $('dialog-content').textContent = data.text;
  $('dialog-note').textContent = data.note;
  $('dialog-eyebrow').textContent = 'Session details';
}

function showDetails(spec) {
  detailSpec = spec;
  updateDetails();
  modal('detail-dialog');
}

function saveText(text, filename) {
  const url = URL.createObjectURL(new Blob([text], {type: 'text/plain;charset=utf-8'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function closeRecording() {
  ++recordingLookup;
  $('recording-video').pause();
  $('recording-video').removeAttribute('src');
  $('recording-video').load();
  close('recording-dialog');
}

async function openRecording(request) {
  if (!request) return;
  if (request.example) { showDetails({kind: 'recording', requestId: request.id}); return; }
  const lookup = ++recordingLookup;
  const url = await recordingSource(request.recordingId);
  if (lookup !== recordingLookup) return;
  if (!url) { toast('This recording was cleared when the session reset.'); return; }
  player.pause();
  $('recording-title').textContent = `${request.id} · ${time(request.recording.duration)}`;
  $('recording-video').src = url;
  $('recording-video').load();
  const download = $('download-recording');
  download.href = request.recordingDownloadURL || url;
  const extension = request.recordingDownloadURL || request.recording.mimeType.includes('mp4') ? 'mp4' : 'webm';
  download.download = `${request.id}-${request.recordingId}.${extension}`;
  download.textContent = request.recordingDownloadURL ? 'Download MP4' : 'Download recording';
  $('recording-note').textContent = 'Original backend recording with camera and microphone input.';
  modal('recording-dialog');
}

function openSettings(kind) {
  settingsKind = kind;
  settingsDraft = {modelId: client.state.config.modelId, caseId: client.state.config.caseId,
    descriptions: {...client.state.config.descriptions}};
  renderSettings();
  modal('settings-dialog');
}

function renderSettings() {
  if (!settingsKind) return;
  const canEdit = client.canAdmin();
  $('settings-title').textContent = settingsKind === 'model' ? 'Model checkpoint' : 'Case and fixed conditions';
  $('settings-eyebrow').textContent = canEdit ? 'Administrator settings' : 'Read-only';
  $('settings-message').textContent = 'Applying settings starts a new media stream';
  $('settings-apply').hidden = !canEdit;
  if (settingsKind === 'model') {
    $('settings-body').innerHTML = `<div class="setting-options">${catalog.models.map((model, index) => `<button class="setting-option ${settingsDraft.modelId === model.id ? 'selected' : ''}" type="button" data-model="${model.id}" aria-pressed="${settingsDraft.modelId === model.id}"><span class="model-number">${index + 1}</span><span><strong>${esc(model.label)}</strong><small>${esc(model.detail || 'Checkpoint supplied by the backend')}</small><small>${esc(model.sampler || '')}</small></span></button>`).join('')}</div><p class="settings-note">Model names and checkpoint parameters are supplied by the backend at launch time.</p>`;
  } else {
    const item = catalog.cases.find(item => item.id === settingsDraft.caseId);
    const description = settingsDraft.descriptions[item.id] ?? item.description;
    const create = canEdit && client instanceof ServerClient ? `<fieldset class="case-description">
      <legend>Save as new Case</legend>
      <label for="new-case-label">Case name</label>
      <input id="new-case-label" maxlength="60" type="text">
      <label for="new-case-reference">New assistant reference (JPEG, PNG, WebP · up to 5 MiB)</label>
      <input id="new-case-reference" type="file" accept="image/jpeg,image/png,image/webp">
      <button id="new-case-save" class="button" type="button">Save and switch</button>
    </fieldset>` : '';
    $('settings-body').innerHTML = `<div class="setting-options">${catalog.cases.map(item => `<button class="setting-option ${settingsDraft.caseId === item.id ? 'selected' : ''}" type="button" data-case="${item.id}" aria-pressed="${settingsDraft.caseId === item.id}"><img src="${esc(item.assistant_image)}" alt=""><span><strong>${esc(item.label)}</strong>${item.subtitle ? `<small>${esc(item.subtitle)}</small>` : ''}</span></button>`).join('')}</div><div class="case-conditions"><div class="case-reference"><img src="${esc(item.assistant_image)}" alt="${esc(item.label)} assistant reference"><small>Fixed assistant reference</small></div><div class="case-description"><label for="case-description">Assistant and scene description</label><textarea id="case-description" maxlength="6000" ${canEdit ? '' : 'readonly'}>${esc(description)}</textarea><button id="restore-description" class="text-button restore-description" type="button" ${canEdit ? '' : 'disabled'}>Restore default description</button></div></div><p class="settings-note">User text and video come from the current recording. This panel configures only the fixed assistant conditions.</p>${create}`;
  }
  permissions();
}

async function leave() {
  if (leaving) return;
  leaving = true;
  camera.abortRecording();
  try { await client.leave(); loggedOut('You left and passed control to the next user.'); }
  catch (error) { fail(error); }
  finally { leaving = false; }
}

function idleTick() {
  if (!client?.member()) return;
  const state = client.state;
  if (state.activeId !== client.memberId || !state.turn) return;
  if (camera.recording || camera.saving) { client.touch().catch(fail); return; }
  const remaining = IDLE_MS - (client.now() - state.turn.lastActivityAt);
  if (!state.turn.acceptedAt) {
    $('turn-expiry').textContent = `Enter within ${time(remaining)} or the turn will pass automatically.`;
  } else if (remaining <= IDLE_WARNING_MS && remaining > 0) {
    $('idle-countdown').textContent = `${Math.ceil(remaining / 1000)} seconds`;
    modal('idle-dialog');
  } else close('idle-dialog');
  if (remaining <= 0 && !idleExpiring) {
    idleExpiring = true;
    client.heartbeat().catch(fail).finally(() => { idleExpiring = false; });
  }
}

function bind() {
  $('login-dialog').addEventListener('cancel', event => event.preventDefault());
  $('turn-dialog').addEventListener('cancel', event => event.preventDefault());
  $('idle-dialog').addEventListener('cancel', event => event.preventDefault());
  $('notice-dialog').addEventListener('cancel', () => { noticeShown = true; });
  $('login-form').addEventListener('change', () => {
    const admin = new FormData($('login-form')).get('role') === 'admin';
    $('admin-password-field').hidden = !admin;
    $('admin-password').disabled = !admin;
    $('admin-password').required = admin;
    $('login-error').hidden = true;
  });
  $('login-form').addEventListener('submit', async event => {
    event.preventDefault();
    if (loginBusy) return;
    const data = new FormData($('login-form'));
    loginBusy = true;
    $('login-error').hidden = true;
    $('login-submit').disabled = true;
    try {
      await client.join(String(data.get('name') || ''), String(data.get('role')), String(data.get('password') || ''));
      $('admin-password').value = '';
      camera.request();
    } catch (error) {
      $('login-error').textContent = error.message;
      $('login-error').hidden = false;
    } finally { loginBusy = false; $('login-submit').disabled = false; }
  });
  $('notice-done').onclick = () => close('notice-dialog');
  $('enter-turn').onclick = () => action(async () => { await client.acceptTurn(); camera.request(); });
  $('skip-turn').onclick = $('leave-session').onclick = $('idle-leave').onclick = leave;
  document.querySelector('.people-panel').addEventListener('click', event => {
    const button = event.target.closest('[data-remove-member]');
    if (!button || button.disabled) return;
    button.disabled = true;
    action(async () => {
      const removed = await client.removeMember(button.dataset.removeMember);
      toast(removed ? `Removed ${removed.name}` : 'The user already left');
    }).finally(() => { if (button.isConnected) button.disabled = false; });
  });
  $('keep-session').onclick = () => action(async () => { await client.touch(true); close('idle-dialog'); });
  $('camera-retry').onclick = () => camera.request();
  $('record-toggle').onclick = () => action(async () => {
    if (camera.recording) { camera.stop(); return; }
    if (!client.canOperate() || captureBusy) return;
    captureBusy = true;
    renderCamera();
    let capture;
    try {
      capture = await client.beginCapture();
      camera.start(capture);
      selectedRequest = null;
      renderView();
    } catch (error) {
      if (capture) await client.cancelCapture(capture);
      throw error;
    } finally { captureBusy = false; renderCamera(); }
  });
  $('upload-video').onclick = () => {
    if (!client.canOperate() || captureBusy || uploadBusy || camera.recording) return;
    $('video-file').value = '';
    $('video-file').click();
  };
  $('video-file').onchange = () => action(async () => {
    const file = $('video-file').files?.[0];
    if (!file || !client.canOperate() || captureBusy || uploadBusy || camera.recording) return;
    uploadBusy = true;
    renderCamera();
    let capture;
    try {
      const {blob, metadata} = await localVideo(file);
      capture = await client.beginCapture();
      selectedRequest = await client.saveRecording(blob, {...metadata, startedAt: capture.startedAt}, capture);
      renderView();
      renderProgress(view, selectedRequest);
      toast('Local video uploaded. Starting speech and chunk alignment.');
    } catch (error) {
      if (capture) await client.cancelCapture(capture).catch(() => {});
      throw error;
    } finally {
      uploadBusy = false;
      $('video-file').value = '';
      renderCamera();
    }
  });
  $('latest-recording').onclick = () => action(() => openRecording(client.state.session.requests.at(-1)));
  $('recording-close').onclick = $('recording-done').onclick = closeRecording;
  $('recording-dialog').addEventListener('close', () => { $('recording-video').pause(); });
  $('recording-dialog').addEventListener('cancel', () => { $('recording-video').pause(); });
  for (const [id, key] of [['global-prompt-toggle', 'originalPrompt'], ['wait-reuse-toggle', 'waitReuse'], ['wait-kv-window-toggle', 'waitKVWindow']]) {
    $(id).onclick = () => action(() => client.setOption(key, !client.state.config[key]));
  }
  $('wait-reuse-threshold').addEventListener('change', () => action(async () => {
    const value = Number($('wait-reuse-threshold').value);
    try { await client.setOption('waitThreshold', value); }
    finally { $('wait-reuse-threshold').value = client.state.config.waitThreshold; }
  }));
  $('reset-cache').onclick = () => action(async () => { await client.reset(false); toast('History reset requested for the next published chunk.'); });
  $('reset-all').onclick = () => action(async () => { await client.reset(true); toast('Full reset requested.'); });
  $('model-picker').onclick = () => openSettings('model');
  $('case-picker').onclick = () => openSettings('case');
  $('settings-close').onclick = $('settings-cancel').onclick = () => { close('settings-dialog'); settingsKind = null; };
  $('settings-dialog').addEventListener('close', () => { settingsKind = null; });
  $('settings-body').addEventListener('click', event => {
    const model = event.target.closest('[data-model]');
    const item = event.target.closest('[data-case]');
    if (model) { settingsDraft.modelId = model.dataset.model; renderSettings(); }
    if (item) {
      if (client.canAdmin()) settingsDraft.descriptions[settingsDraft.caseId] = $('case-description').value;
      settingsDraft.caseId = item.dataset.case;
      renderSettings();
    }
    if (event.target.closest('#restore-description') && client.canAdmin()) {
      settingsDraft.descriptions[settingsDraft.caseId] = catalog.cases.find(item => item.id === settingsDraft.caseId).description;
      renderSettings();
    }
  });
  $('settings-body').addEventListener('click', event => {
    if (!event.target.closest('#new-case-save')) return;
    action(async () => {
      if (!client.canAdmin()) return;
      const label = $('new-case-label').value.trim();
      const description = $('case-description').value.trim();
      const file = $('new-case-reference').files[0];
      if (!label || !description || !file || file.size > 5 * 1024 * 1024) {
        throw new Error('Enter a Case name and description, and choose a reference image up to 5 MiB.');
      }
      const reference = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.onerror = () => reject(new Error('The reference image could not be read.'));
        reader.readAsDataURL(file);
      });
      await client.request('/api/cases', {label, description, reference});
      close('settings-dialog');
      settingsKind = null;
      toast('Case saved. Preparing its reference image.');
    });
  });
  $('settings-body').addEventListener('input', updateSettingsApply);
  $('settings-apply').onclick = () => action(async () => {
    if (!client.canAdmin()) return;
    const patch = settingsKind === 'model' ? {modelId: settingsDraft.modelId} : {caseId: settingsDraft.caseId, description: $('case-description').value};
    await client.applySettings(patch);
    close('settings-dialog'); settingsKind = null;
    toast('Switch requested. Waiting for the current chunk to publish.');
  });
  $('timeline-scroll').addEventListener('click', event => {
    const context = event.target.closest('[data-context]');
    if (context) { showDetails({kind: context.dataset.context}); return; }
    const info = event.target.closest('[data-info-chunk]');
    if (info) { showDetails({kind: 'chunk', index: Number(info.dataset.infoChunk), track: info.dataset.infoTrack}); return; }
    const button = event.target.closest('[data-chunk]');
    if (!button) return;
    const index = Number(button.dataset.chunk);
    const chunk = view.chunks.find(item => item.global === index);
    selectedChunk = index;
    if (chunk?.codec === 'consumed') { setFollow(false); player.seekChunk(index); }
    else showDetails({kind: 'chunk', index, track: button.dataset.track});
  });
  $('timeline-scroll').addEventListener('dblclick', event => {
    const button = event.target.closest('[data-chunk]');
    if (button) showDetails({kind: 'chunk', index: Number(button.dataset.chunk), track: button.dataset.track});
  });
  $('timeline-scroll').addEventListener('wheel', () => setFollow(false), {passive: true});
  $('timeline-scroll').addEventListener('touchmove', () => setFollow(false), {passive: true});
  $('timeline-scroll').addEventListener('pointerdown', event => { if (event.target === $('timeline-scroll')) setFollow(false); });
  $('timeline-scroll').addEventListener('keydown', event => { if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) setFollow(false); });
  $('go-start').onclick = () => { setFollow(false); $('timeline-scroll').scrollLeft = 0; if (view.media) player.seekChunk(0); };
  $('go-latest').onclick = () => { setFollow(true); if (view.media) player.seekChunk(view.media.end); };
  $('follow-stream').onclick = () => setFollow(!follow);
  $('request-picker').addEventListener('change', () => {
    selectedRequest = $('request-picker').value === '__capture' ? null : $('request-picker').value;
    renderProgress(view, selectedRequest);
  });
  document.querySelector('.stages').addEventListener('click', event => {
    const button = event.target.closest('[data-info-chunk]');
    if (button) showDetails({kind: 'chunk', index: Number(button.dataset.infoChunk), track: button.dataset.infoTrack});
  });
  $('shared-features').onclick = () => showDetails({kind: 'shared'});
  $('show-asr').onclick = () => showDetails({kind: 'asr', requestId: currentRequest(view, selectedRequest)?.id});
  $('show-sequence').onclick = () => showDetails({kind: 'sequence', requestId: currentRequest(view, selectedRequest)?.id});
  $('recording-detail').onclick = () => action(() => openRecording(currentRequest(view, selectedRequest)));
  $('close-dialog').onclick = $('dialog-done').onclick = () => close('detail-dialog');
  $('detail-dialog').addEventListener('close', () => { detailSpec = null; });
  $('copy-detail').onclick = () => action(async () => {
    if (navigator.clipboard?.writeText) await navigator.clipboard.writeText($('dialog-content').textContent);
    else {
      const range = document.createRange(); range.selectNodeContents($('dialog-content'));
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      if (!document.execCommand('copy')) throw new Error('Select the text and copy it manually.');
    }
    toast('Details copied');
  });
  $('download-detail').onclick = () => { if (detailSpec) { const data = detailContent(detailSpec); saveText(data.text, data.filename); } };
  for (const eventName of ['pointerdown', 'keydown', 'input', 'wheel']) {
    document.addEventListener(eventName, event => {
      if (event.isTrusted && !$('idle-dialog').open && !$('turn-dialog').open) client.touch().catch(fail);
    }, {passive: true});
  }
  window.addEventListener('pagehide', () => { client.suspend(); camera.close(); player.pause(); revokeMedia(); });
  window.addEventListener('pageshow', event => {
    if (!event.persisted) return;
    action(async () => {
      await client.resume();
      if (client.member()) {
        if (client.state.session.capture?.ownerId === client.memberId) await client.cancelCapture(client.state.session.capture);
        camera.request();
      }
    });
  });
}

async function boot() {
  modal('login-dialog');
  const override = new URLSearchParams(location.search).get('catalog');
  for (const catalogURL of override ? [override] : ['../api/frontend-catalog', './catalog.json']) {
    const response = await fetch(catalogURL, {cache: 'no-store'}).catch(() => null);
    if (response?.ok) { catalog = await response.json(); break; }
  }
  if (!catalog) throw new Error('Frontend configuration failed to load. Refresh the page.');
  if (!catalog.models?.length || !catalog.cases?.length) throw new Error('The frontend catalog must define at least one model and one case.');
  client = new URLSearchParams(location.search).has('preview') ? new PreviewClient(catalog) : new ServerClient(catalog.schema);
  camera = new CameraCapture($('camera-preview'), {
    onChange: renderCamera,
    onComplete: async (blob, metadata, capture) => {
      try {
        selectedRequest = await client.saveRecording(blob, metadata, capture);
        renderView();
        renderProgress(view, selectedRequest);
        toast('Video uploaded. Starting speech and chunk alignment.');
      } catch (error) { await client.cancelCapture(capture).catch(() => {}); throw error; }
    },
    onError: message => toast(message),
  });
  player = new DualVideoPlayer({resolveRecording: recordingSource, onPosition: positionChanged, onError: toast, onSeek: () => setFollow(false)});
  bind();
  client.addEventListener('change', sessionChanged);
  client.addEventListener('failure', event => fail(event.detail));
  await client.connect();
  $('login-submit').disabled = false;
  $('login-submit').innerHTML = 'Enter and join the queue <svg><use href="#i-arrow"/></svg>';
  if (client.member()) {
    if (client.state.session.capture?.ownerId === client.memberId) await client.cancelCapture(client.state.session.capture);
    camera.request();
  }
  sessionChanged();
  setInterval(() => { renderView(); idleTick(); }, 400);
}

boot().catch(error => {
  $('login-error').textContent = error.message || 'Could not connect to the service. Refresh and try again.';
  $('login-error').hidden = false;
  $('login-submit').textContent = 'Refresh to retry';
  $('login-submit').disabled = true;
});
