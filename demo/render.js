import {$, CHUNK_SECONDS, FRAMES, clamp, code, esc, fill, percent, seconds, stageNames, time} from './ui.js';

export const requestNames = {idle: 'Waiting for input', recording: 'Recording', preparing: 'Preparing',
  uploading: 'Uploading', queued: 'Queued', asr: 'Speech detection', building: 'Aligning chunks', running: 'Generating',
  completed: 'Completed', failed: 'Failed', no_speech: 'No speech detected', cancelled: 'Cancelled'};

export function currentRequest(view, id) {
  if (id) {
    const request = view.requests.find(item => item.id === id);
    if (request) return request;
  }
  if (view.capture) return {id: '__capture', author: 'Current user', state: 'recording',
    recording: {duration: ((view.now ?? Date.now()) - view.capture.startedAt) / 1000}, asr: {progress: 0, segments: []},
    build: 0, chunks: [], transcript: ''};
  return view.requests.at(-1) || null;
}

export function renderPeople(state, memberId) {
  const active = state.members.find(member => member.id === state.activeId);
  const waiting = state.members.filter(member => member.id !== state.activeId);
  const me = state.members.find(member => member.id === memberId);
  const position = waiting.findIndex(member => member.id === memberId) + 1;
  const role = member => member.role === 'admin' ? 'Administrator' : 'User';
  const removeButton = member => me?.role === 'admin' && member.id !== memberId ? `<button class="queue-remove" type="button" data-remove-member="${esc(member.id)}" aria-label="Remove ${esc(member.name)}" title="Remove ${esc(member.name)}">Remove</button>` : '';
  $('queue-total').textContent = `${waiting.length} waiting`;
  $('operator-card').classList.toggle('is-empty', !active);
  const operatorHTML = active ? `<span class="operator-avatar" aria-hidden="true">${esc([...active.name][0])}</span><div class="operator-info"><div><strong class="operator-name" title="${esc(active.name)}">${esc(active.name)}${active.id === memberId ? ' (me)' : ''}</strong><span class="role-tag">${role(active)}</span></div><span class="operator-status"><span class="dot green"></span>${state.turn?.acceptedAt ? 'Active' : 'Turn offered · waiting to enter'}</span></div>${removeButton(active)}` : 'No active user';
  const peopleHTML = waiting.map((member, index) => `<li class="person-row ${member.id === memberId ? 'is-me' : ''}"><span class="person-position mono">${index + 1}</span><span class="person-avatar" aria-hidden="true">${esc([...member.name][0])}</span><span class="person-name" title="${esc(member.name)}">${esc(member.name)}${member.id === memberId ? ' (me)' : ''}</span><span class="role-tag">${role(member)}</span>${removeButton(member)}</li>`).join('') || '<li class="people-empty">No one else is waiting</li>';
  // Heartbeats must not replace a focused or pressed moderation button.
  for (const [id, html] of [['operator-card', operatorHTML], ['people-list', peopleHTML]]) {
    const element = $(id);
    if (element.dataset.rendered !== html) {
      element.innerHTML = html;
      element.dataset.rendered = html;
    }
  }
  $('viewer-notice').hidden = !me || state.activeId === memberId;
  $('viewer-notice-text').textContent = `Read-only observer · number ${position} in queue`;
  $('my-session-label').textContent = me ? `Me · ${me.name}` : '';
  $('my-session-label').title = me ? `${me.name} · ${role(me)}` : '';
  $('leave-session').textContent = state.activeId === memberId ? 'Leave session' : 'Leave queue';
  $('leave-session').title = state.activeId === memberId ? 'End this session and pass control to the next user' : 'Leave the waiting queue';
  $('entry-queue').textContent = active ? `${active.name} · ${state.turn?.acceptedAt ? 'active' : 'turn offered'} · ${waiting.length} waiting` : 'No active user';
  return {active, waiting, me, position};
}

export function renderHeader(state, catalog, view, canOperate) {
  const config = state.config;
  const model = catalog.models.find(item => item.id === config.modelId) || catalog.models[0];
  const item = catalog.cases.find(item => item.id === config.caseId) || catalog.cases[0];
  $('model-label').textContent = model.label;
  $('model-step').textContent = model.detail || 'Configured model';
  $('case-name').textContent = item.label;
  if ($('case-thumb').getAttribute('src') !== item.assistant_image) $('case-thumb').src = item.assistant_image;
  $('case-thumb').alt = '';
  $('live-status').textContent = state.error ? 'Service error' : !state.ready ? 'Preparing' : canOperate ? (view.phase === 'idle' ? 'Ready' : requestNames[view.phase] || 'Ready') : 'Read-only';
  for (const [id, key, label] of [
    ['global-prompt-toggle', 'originalPrompt', 'Base intent'], ['wait-reuse-toggle', 'waitReuse', 'WAIT reuse'],
    ['wait-kv-window-toggle', 'waitKVWindow', 'Short history'],
  ]) {
    $(id).textContent = `${label}: ${config[key] ? 'On' : 'Off'}`;
    $(id).setAttribute('aria-pressed', String(config[key]));
  }
  if (document.activeElement !== $('wait-reuse-threshold')) $('wait-reuse-threshold').value = config.waitThreshold;
  const consumed = [...view.chunks].reverse().find(chunk => chunk.realization !== 'pending');
  $('wait-reuse-status').textContent = !config.waitReuse ? 'Disabled' : consumed?.waitReuseSource !== null && consumed?.waitReuseSource !== undefined ? `Reusing ${code(consumed.waitReuseSource)}` : consumed ? `Streak ${consumed.waitStreak || 0}` : 'Waiting for chunk';
  $('global-prompt-toggle').title = 'Choose the base or adapted shared intent condition.';
  $('wait-reuse-toggle').title = 'After N consecutive WAIT chunks, reuse the intent condition from chunk N for later WAIT chunks.';
  $('wait-kv-window-toggle').title = consumed ? `Recent media history window: ${consumed.historyWindow || 0} chunks` : 'Use a shorter committed media history during WAIT chunks';
  document.querySelectorAll('.gpu-row').forEach((row, index) => {
    const gpu = state.gpus?.find(gpu => gpu.index === index);
    const metrics = row.querySelectorAll('.gpu-metrics b');
    metrics[0].textContent = gpu ? `${gpu.utilization}%` : '—';
    metrics[1].textContent = gpu ? `${(gpu.used_mib / 1024).toFixed(1)} / ${(gpu.total_mib / 1024).toFixed(0)} GiB` : '—';
    row.querySelector('.gpu-meter').innerHTML = `<i style="width:${gpu?.utilization || 0}%"></i>`;
  });
}

function cell(chunk, track) {
  const user = track === 'user';
  const mapping = chunk.user;
  const published = chunk.codec === 'consumed';
  const generated = ['ready', 'consumed'].includes(chunk.realization);
  const reused = user && mapping?.kind === 'reused';
  const status = user ? reused ? 'reused' : mapping ? 'published' : 'unknown' : published ? 'published' : generated ? 'generated' : chunk.realization === 'running' ? 'working' : 'pending';
  const text = user ? reused ? 'Reused' : mapping?.text || 'No speech text' : chunk.text || 'Waiting for intent';
  const mark = user ? reused ? `↻${code(mapping.sourceChunk)}` : mapping ? '✓' : '' : chunk.realization === 'running' ? `${Math.floor(chunk.step)}/${chunk.realizationSteps}` : published ? '✓' : '';
  const progress = user ? mapping ? 100 : 0 : published || generated ? 100 : chunk.realization === 'running' ? percent(chunk.step, chunk.realizationSteps) : 0;
  return `<button class="timeline-cell ${status}" type="button" data-cell="${chunk.global}" data-chunk="${chunk.global}" data-track="${track}" aria-label="${code(chunk.global)} ${user ? 'User' : 'Assistant'}, ${published ? 'seek playback' : 'view details; assistant media is not published'}: ${esc(text)}"><span class="condition" data-info-chunk="${chunk.global}" data-info-track="${track}" title="${esc(text)}"><span class="condition-text">${esc(text)}</span></span><div class="segment"><i class="segment-fill" style="width:${clamp(progress, 0, 100)}%"></i><span class="segment-mark">${mark}</span></div><span class="timeline-label"><b>${code(chunk.global)}</b><span>${(chunk.timelineStart ?? chunk.global * CHUNK_SECONDS).toFixed(2)}s</span></span></button>`;
}

export function renderTracks(view, follow, description = '') {
  const chunks = view.chunks;
  const published = chunks.filter(chunk => chunk.codec === 'consumed').length;
  const real = chunks.filter(chunk => chunk.user?.kind === 'real').length;
  const reused = chunks.filter(chunk => chunk.user?.kind === 'reused').length;
  const assistantContext = `<button class="context-cell" type="button" data-context="system" aria-label="View fixed intent context"><span class="context-title">ASSISTANT · INTENT ↗</span><span class="context-preview">${esc(description || 'Fixed intent context')}</span><span class="context-bar"><span>context</span><span>expand</span></span></button>`;
  const userContext = `<button class="context-cell" type="button" data-context="user-inputs" aria-label="View user video sources"><span class="context-title">USER · INPUT VIDEO ↗</span><span class="context-preview">${chunks.length ? `${real} input chunks · ${reused} reused chunks` : 'Waiting for speech and chunk alignment'}</span><span class="context-bar"><span>video · text</span><span>expand</span></span></button>`;
  const trailing = view.phase !== 'completed' ? Array.from({length: 10}, (_, index) => `<div class="timeline-cell unknown"><span class="condition">${index === 0 ? 'Waiting' : ''}</span><div class="segment"></div><span class="timeline-label"><b>${code(chunks.length + index)}</b><span>—</span></span></div>`).join('') : '';
  $('assistant-track').innerHTML = assistantContext + chunks.map(chunk => cell(chunk, 'assistant')).join('') + trailing;
  $('user-track').innerHTML = userContext + chunks.map(chunk => cell(chunk, 'user')).join('') + trailing;
  $('stream-id').textContent = `S-${view.id.slice(0, 6)}`;
  $('stream-state').textContent = requestNames[view.phase] || 'Waiting for input';
  $('stream-summary').textContent = `Assistant ${published} / ${chunks.length} chunks · ${(published * CHUNK_SECONDS).toFixed(2)}s`;
  if (follow && chunks.length) {
    const current = chunks.find(chunk => chunk.realization === 'running') || chunks.at(-1);
    const track = $('assistant-track');
    const target = track.querySelector(`[data-chunk="${current.global}"]`);
    const scroll = $('timeline-scroll');
    const left = target.offsetLeft - track.offsetLeft;
    if (left < scroll.scrollLeft || left + target.offsetWidth > scroll.scrollLeft + scroll.clientWidth) {
      scroll.scrollLeft = Math.max(0, left - scroll.clientWidth * 0.6);
    }
  }
}

export function renderProgress(view, selectedRequest) {
  const request = currentRequest(view, selectedRequest);
  const r = request || {id: '', state: 'idle', chunks: [], asr: {progress: 0, segments: []}, build: 0};
  const chunks = view.chunks.filter(chunk => chunk.requestId === r.id);
  const n = chunks.length;
  const options = view.requests.map(item => `<option value="${esc(item.id)}">${esc(item.id)} · ${requestNames[item.state] || item.state}</option>`);
  if (view.capture) options.push('<option value="__capture">Recording</option>');
  const optionHTML = options.join('') || '<option value="">Waiting for input</option>';
  if ($('request-picker').innerHTML !== optionHTML) $('request-picker').innerHTML = optionHTML;
  $('request-picker').value = r.id;
  $('detail-status').textContent = `${view.example ? 'Example · ' : ''}${requestNames[r.state] || r.state}`;
  $('detail-status').className = `status-label ${r.state === 'completed' ? 'completed' : ['failed', 'no_speech'].includes(r.state) ? 'failed' : ['asr', 'building', 'running', 'recording'].includes(r.state) ? 'running' : ''}`;
  $('detail-text').textContent = r.error || r.notice || r.transcript || ({idle: 'Record or upload a video. The system will locate speech and align it into chunks.',
    recording: 'Recording camera and microphone input. Select “Stop recording” to submit.',
    asr: 'Detecting speech in the input video…', building: 'Aligning user video and text into intent chunks…',
    no_speech: 'No usable speech was detected. Record another segment.'}[r.state] || 'Waiting to process video');
  $('detail-meta').textContent = r.recording ? `${r.author || ''} · ${time(r.recording.duration)} recording${r.example ? ' · example media' : ''}` : '';
  $('detail-chunks').textContent = n ? `${n} chunks · ${(n * CHUNK_SECONDS).toFixed(2)}s` : 'Not aligned';
  $('recording-detail').disabled = !request || r.state === 'recording';
  $('recording-detail').textContent = r.example ? 'Video source' : 'Recording details';
  fill('asr-fill', r.asr?.progress || 0);
  $('asr-percent').textContent = `${r.asr?.progress || 0}%`;
  $('show-asr').disabled = !r.asr?.segments?.length && !r.asr?.error;
  $('asr-detail').textContent = r.asr?.segments?.length ? `Speech ${seconds(r.asr.segments[0].start)}–${seconds(r.asr.segments.at(-1).end)} · usable video located` : r.state === 'asr' ? 'Detecting speech boundaries…' : r.asr?.error || 'Locate speech and trim the usable video segment';
  fill('build-fill', r.build || 0);
  $('build-percent').textContent = `${r.build || 0}%`;
  $('show-sequence').disabled = !r.sequenceURL;
  $('build-detail').textContent = r.build >= 100 ? `${r.inputChunks || chunks.filter(chunk => chunk.user?.kind === 'real').length} user chunks ready · later chunks continue with intent execution` : r.state === 'building' ? 'Aligning text and video to build intent_sequence.txt…' : 'Waiting for speech detection · align user video and text automatically';
  for (const name of ['intent', 'realization', 'codec']) {
    const finished = chunks.filter(chunk => ['ready', 'consumed'].includes(chunk[name])).length;
    const ready = chunks.filter(chunk => chunk[name] === 'ready').length;
    const active = chunks.find(chunk => chunk[name] === 'running');
    const progress = name === 'realization' ? active ? percent(active.step, active.realizationSteps) : n && finished === n ? 100 : 0 : percent(finished, n);
    $(name + '-grid').innerHTML = chunks.map(chunk => `<button class="chunk-tile ${chunk[name]}" type="button" data-info-chunk="${chunk.global}" data-info-track="${name}" aria-label="${code(chunk.global)} ${name.toUpperCase()} ${stageNames[chunk[name]]}">${String(chunk.global).padStart(3, '0')}</button>`).join('') || '<span class="empty-grid">Waiting for intent sequence</span>';
    $(name + '-count').textContent = `${finished} / ${n}`;
    $(name + '-percent').textContent = `${progress}%`;
    fill(name + '-fill', progress);
    $(name + '-queue').textContent = ready;
    $(name + '-detail').textContent = name === 'intent' ? `${finished} / ${n} chunks ready` : name === 'codec' ? `${finished * FRAMES} / ${n * FRAMES} frames decoded` : active ? `${code(active.global)} · step ${active.step} / ${active.realizationSteps}` : n && finished === n ? `${n} chunks generated` : 'Waiting for intent states';
  }
}
