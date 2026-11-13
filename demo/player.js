import {$, CHUNK_SECONDS, clamp, code, icon, time} from './ui.js';

export class DualVideoPlayer {
  constructor({resolveRecording, onPosition, onError, onSeek}) {
    this.assistant = $('video');
    this.user = $('user-video');
    this.resolveRecording = resolveRecording;
    this.onPosition = onPosition;
    this.onError = onError;
    this.onSeek = onSeek;
    this.chunks = [];
    this.latest = null;
    this.loaded = null;
    this.loading = false;
    this.applying = false;
    this.userSource = null;
    this.userSourceKey = null;
    this.userGeneration = 0;
    this.volume = {assistant: 0.8, user: 0.8};
    this.muted = {assistant: false, user: false};
    this.bigUser = false;
    this.frame = null;
    this.autoplay = false;
    this.follow = true;
    this.wantsPlayback = false;
    this.waitingForMore = false;
    this.followPosition = 0;
    this.sourceGeneration = 0;
    this.lastPaint = 0;
    this.bind();
    this.updateVolume();
  }

  get ceiling() {
    if (!this.loaded) return 0;
    const published = this.loaded.duration ?? ((this.loaded.end + 1) * CHUNK_SECONDS);
    return Number.isFinite(this.assistant.duration) ? Math.max(0, Math.min(published, this.assistant.duration)) : Math.max(0, published);
  }

  bind() {
    this.assistant.addEventListener('loadedmetadata', () => {
      if (!this.loaded) return;
      if (this.assistant.videoWidth && this.assistant.videoHeight) $('video-canvas').style.aspectRatio = `${this.assistant.videoWidth} / ${this.assistant.videoHeight}`;
      this.assistant.currentTime = clamp(this.seekAfterLoad || 0, 0, Math.max(0, this.ceiling - 0.025));
      this.syncUser(true);
      this.render();
    });
    this.assistant.addEventListener('loadeddata', () => {
      if (!this.loaded) return;
      this.loading = false;
      $('video-loading').hidden = true;
      if (this.autoplay && this.wantsPlayback) { this.autoplay = false; this.play(); }
      else this.updateMedia();
      this.render();
    });
    this.assistant.addEventListener('play', () => {
      if (this.assistant.paused) return;
      this.wantsPlayback = true;
      this.waitingForMore = false;
      this.syncUser(true); this.tick(); this.render();
    });
    this.assistant.addEventListener('pause', () => {
      // A pause queued by source switching can arrive after playback resumes.
      if (!this.assistant.paused) return;
      this.user.pause();
      cancelAnimationFrame(this.frame);
      if (!this.applying && !this.loading) {
        if (this.wantsPlayback && !this.waitingForMore && this.assistant.currentTime >= this.ceiling - 0.012) this.finishPlayback();
        else if (!this.waitingForMore) {
          this.wantsPlayback = false;
          this.autoplay = false;
          this.updateMedia();
        }
      }
      this.render();
    });
    this.assistant.addEventListener('seeking', () => { this.syncUser(true); this.render(); });
    this.assistant.addEventListener('timeupdate', () => { this.enforceCeiling(); this.syncUser(); this.render(); });
    this.assistant.addEventListener('ended', () => {
      if (!this.loading && this.assistant.currentTime >= this.ceiling - 0.012) this.finishPlayback();
    });
    this.assistant.addEventListener('waiting', () => this.user.pause());
    this.assistant.addEventListener('playing', () => this.syncUser(true));
    this.assistant.addEventListener('ratechange', () => { this.user.playbackRate = this.assistant.playbackRate; });
    this.assistant.addEventListener('error', () => {
      if (!this.loaded) return;
      this.loading = false;
      this.autoplay = false;
      this.wantsPlayback = false;
      this.waitingForMore = false;
      $('video-loading').hidden = true;
      this.user.pause();
      this.onError('The assistant video could not load. Select “Update and seek” to retry.');
      this.render();
    });
    this.user.addEventListener('loadedmetadata', () => this.syncUser(true));
    this.user.addEventListener('loadeddata', () => { $('user-placeholder').hidden = true; this.syncUser(true); });
    this.user.addEventListener('error', () => {
      if (!this.userSource) return;
      $('user-placeholder').hidden = false;
      $('user-placeholder').querySelector('small').textContent = 'This user video is unavailable';
    });
    $('play-toggle').onclick = $('play-overlay').onclick = () => this.togglePlay();
    $('seek').addEventListener('input', () => {
      if (!this.loaded || this.assistant.readyState < 1) return;
      const at = Number($('seek').value);
      this.onSeek?.();
      this.assistant.currentTime = clamp(at, 0, Math.max(0, this.ceiling - 0.025));
      this.syncUser(true);
      this.render();
    });
    $('refresh-video').onclick = () => this.apply(this.latest, this.latest.end * CHUNK_SECONDS, false, Boolean(this.assistant.error));
    $('swap-videos').onclick = () => this.swap();
    document.querySelectorAll('[data-swap]').forEach(button => { button.onclick = () => this.swap(); });
    $('fullscreen').onclick = () => {
      const action = document.fullscreenElement ? document.exitFullscreen?.() : $('video-canvas').requestFullscreen?.();
      if (!action) this.onError('This browser does not support fullscreen dual-video playback.');
      else action.catch(() => this.onError('The browser did not allow fullscreen mode.'));
    };
    for (const role of ['assistant', 'user']) {
      $(`${role}-volume`).addEventListener('input', () => {
        this.volume[role] = Number($(`${role}-volume`).value);
        this.muted[role] = false;
        this.updateVolume();
      });
      $(`${role}-mute`).onclick = () => { this.muted[role] = !this.muted[role]; this.updateVolume(); };
    }
  }

  update(view) {
    if (this.streamId !== view.id) { this.clear(); this.streamId = view.id; }
    this.chunks = view.chunks;
    this.latest = view.media;
    this.updateMedia();
    if (!this.latest && this.loaded) this.clear();
    // User chunks may be ready before the first Assistant publication; display a still only.
    if (!this.loaded && this.chunks.length) this.syncUser(true);
    this.render();
  }

  setFollow(value) {
    this.follow = value;
    if (!value && this.waitingForMore) {
      this.waitingForMore = false;
      this.wantsPlayback = false;
      this.autoplay = false;
    }
    this.updateMedia();
    this.render();
  }

  updateMedia() {
    if (!this.latest || this.loading) return;
    if (!this.loaded) this.apply(this.latest, this.latest.end * CHUNK_SECONDS);
    else if (this.latest.version > this.loaded.version && this.assistant.paused) {
      if (this.follow && this.wantsPlayback && this.waitingForMore) this.apply(this.latest, this.followPosition, true);
      else if (this.follow && !this.wantsPlayback) this.apply(this.latest, this.latest.end * CHUNK_SECONDS);
    }
  }

  finishPlayback() {
    if (!this.loaded || this.loading) return;
    this.waitingForMore = this.follow && this.wantsPlayback;
    this.followPosition = this.ceiling;
    if (!this.waitingForMore) this.wantsPlayback = false;
    this.assistant.pause();
    this.user.pause();
    cancelAnimationFrame(this.frame);
    this.updateMedia();
    this.render();
  }

  apply(media, at, autoplay = false, reload = false) {
    if (!media) return;
    this.applying = true;
    ++this.sourceGeneration;
    this.wantsPlayback = autoplay;
    this.waitingForMore = false;
    this.assistant.pause();
    this.user.pause();
    this.loaded = {...media};
    this.seekAfterLoad = at;
    this.autoplay = autoplay;
    const sameSource = this.assistant.getAttribute('src') === media.url;
    if (!reload && sameSource && this.assistant.readyState >= 2) {
      this.loading = false;
      this.assistant.currentTime = clamp(at, 0, Math.max(0, this.ceiling - 0.025));
      this.autoplay = false;
      this.syncUser(true);
    } else {
      this.loading = true;
      this.assistant.src = media.url;
      this.assistant.load();
    }
    this.applying = false;
    if (!this.loading && autoplay) this.play();
    this.render();
  }

  async play() {
    if (!this.loaded) return;
    this.wantsPlayback = true;
    if (this.loading) { this.autoplay = true; this.render(); return; }
    if (this.assistant.currentTime >= this.ceiling - 0.03) {
      if (this.follow && this.latest?.version > this.loaded.version) {
        this.waitingForMore = true;
        this.followPosition = this.ceiling;
        this.updateMedia();
        return;
      }
      this.assistant.currentTime = 0;
    }
    this.waitingForMore = false;
    const generation = this.sourceGeneration;
    this.syncUser(true);
    try { await this.assistant.play(); }
    catch {
      if (generation !== this.sourceGeneration || !this.wantsPlayback) return;
      this.wantsPlayback = false;
      this.autoplay = false;
      this.onError('The video is still loading or playback was blocked. Select play again.');
      this.render();
    }
  }

  togglePlay() {
    if (!this.loaded) return;
    if (this.wantsPlayback || !this.assistant.paused) this.pause();
    else this.play();
  }

  seekChunk(index) {
    const chunk = this.chunks.find(item => item.global === index);
    if (!chunk || chunk.codec !== 'consumed' || !this.latest || index > this.latest.end) {
      this.onError(`Assistant media for ${code(index)} is not published yet. Details are available, but playback cannot seek there.`);
      return false;
    }
    const at = chunk.timelineStart ?? index * CHUNK_SECONDS;
    if (!this.loaded || index > this.loaded.end || this.assistant.error || this.assistant.readyState < 1) this.apply(this.latest, at, true, Boolean(this.assistant.error));
    else {
      this.assistant.currentTime = clamp(at, 0, Math.max(0, this.ceiling - 0.025));
      this.syncUser(true);
      this.play();
    }
    this.render();
    return true;
  }

  enforceCeiling() {
    if (!this.loaded) return;
    if (this.assistant.currentTime >= this.ceiling - 0.012 && !this.assistant.paused) {
      this.finishPlayback();
    }
    if (this.assistant.currentTime > this.ceiling && this.assistant.readyState >= 1) this.assistant.currentTime = this.ceiling;
  }

  tick() {
    cancelAnimationFrame(this.frame);
    if (this.assistant.paused) return;
    this.enforceCeiling();
    this.syncUser();
    if (performance.now() - this.lastPaint > 90) { this.render(); this.lastPaint = performance.now(); }
    if (!this.assistant.paused) this.frame = requestAnimationFrame(() => this.tick());
  }

  currentChunk() {
    const at = Math.min(this.assistant.currentTime || 0, Math.max(0, this.ceiling - 0.001));
    return this.chunks.find(chunk => at >= chunk.timelineStart && at < chunk.timelineEnd) || this.chunks[0];
  }

  continuousUserPair(previous, next) {
    const left = previous?.user;
    const right = next?.user;
    if (!left || !right || left.kind !== 'real' || right.kind !== 'real') return false;
    const leftEnd = left.playbackEnd ?? left.sourceEnd;
    const rightStart = right.playbackStart ?? right.sourceStart;
    return left.recordingId === right.recordingId
      && (left.url || '') === (right.url || '')
      && Number.isFinite(leftEnd) && Number.isFinite(rightStart)
      && Math.abs(leftEnd - rightStart) < 0.04
      && Math.abs(previous.timelineEnd - next.timelineStart) < 0.04;
  }

  async loadUserSource(mapping, key) {
    const generation = ++this.userGeneration;
    this.userSourceKey = key;
    this.userSource = null;
    this.user.pause();
    this.user.removeAttribute('src');
    this.user.load();
    $('user-placeholder').hidden = false;
    $('user-placeholder').querySelector('small').textContent = 'Loading user video…';
    try {
      const url = mapping.url || await this.resolveRecording(mapping.recordingId);
      if (generation !== this.userGeneration) return;
      if (!url) { $('user-placeholder').querySelector('small').textContent = 'Waiting for user video'; return; }
      this.userSource = url;
      this.user.src = url;
      this.user.load();
    } catch {
      if (generation === this.userGeneration) $('user-placeholder').querySelector('small').textContent = 'User video failed to load';
    }
  }

  syncUser(force = false) {
    const chunk = this.currentChunk();
    const mapping = chunk?.user;
    if (!mapping) { this.user.pause(); this.updateVolume(); return; }
    const key = `${mapping.recordingId}:${mapping.url || ''}`;
    if (this.userSourceKey !== key) { this.loadUserSource(mapping, key); return; }
    $('user-source-label').textContent = `User · ${mapping.kind === 'reused' ? 'reused from ' + code(mapping.sourceChunk) : 'recorded input'}`;
    this.updateVolume();
    if (!this.userSource || this.user.readyState < 1) return;
    const playbackStart = mapping.playbackStart ?? mapping.sourceStart;
    const playbackEnd = mapping.playbackEnd ?? mapping.sourceEnd;
    const sourceDuration = Math.max(0.001, playbackEnd - playbackStart);
    const localTime = clamp((this.assistant.currentTime || 0) - chunk.timelineStart, 0, chunk.timelineEnd - chunk.timelineStart);
    const sourceEnd = Number.isFinite(this.user.duration) ? Math.min(playbackEnd, this.user.duration) : playbackEnd;
    const target = Math.min(playbackStart + Math.min(localTime, sourceDuration - 0.01), Math.max(0, sourceEnd - 0.015));
    // Every reused chunk restarts the entire final speaking segment. It is never a frozen final frame.
    const boundary = this.syncedChunk !== chunk.global;
    const previous = boundary ? this.chunks.find(item => item.global === this.syncedChunk) : null;
    const continuousBoundary = boundary && this.continuousUserPair(previous, chunk);
    this.syncedChunk = chunk.global;
    const drift = this.user.currentTime - target;
    // Real input is one continuous media file. Seeking it at every 17-frame
    // boundary restarts the browser's audio decoder and creates a periodic
    // click/dropout. Keep contiguous chunks running and reserve hard seeks for
    // explicit positioning, source/reuse boundaries, or severe clock drift.
    const driftLimit = mapping.kind === 'real' ? 0.50 : 0.10;
    const hardSync = force || (boundary && !continuousBoundary) || Math.abs(drift) > driftLimit;
    if (hardSync) {
      try { this.user.currentTime = Math.max(0, target); } catch { return; }
    }
    const baseRate = this.assistant.playbackRate;
    const correction = mapping.kind === 'real' && !hardSync && Math.abs(drift) > 0.04
      ? clamp(1 - drift * 0.25, 0.97, 1.03) : 1;
    this.user.playbackRate = baseRate * correction;
    const next = this.chunks.find(item => item.global === chunk.global + 1);
    const continues = this.continuousUserPair(chunk, next);
    const shouldPlay = this.loaded && !this.assistant.paused && this.assistant.readyState >= 3
      && (continues || localTime < sourceDuration - 0.01);
    if (shouldPlay && this.user.paused && !this.userPlayPending) {
      this.userPlayPending = true;
      this.user.play().catch(() => {
        if (!this.userPlayWarned) { this.userPlayWarned = true; this.onError('The browser paused the user video. Select play to resynchronize.'); }
      }).finally(() => { this.userPlayPending = false; });
    } else if (!shouldPlay) this.user.pause();
  }

  updateVolume() {
    this.assistant.volume = this.volume.assistant;
    this.assistant.muted = this.muted.assistant;
    this.user.volume = this.volume.user;
    const mapping = this.currentChunk()?.user;
    this.user.muted = this.muted.user || !mapping?.hasAudio || mapping?.kind === 'reused';
    for (const role of ['assistant', 'user']) {
      const muted = this.muted[role] || this.volume[role] === 0;
      $(`${role}-mute`).setAttribute('aria-pressed', String(muted));
      $(`${role}-mute`).setAttribute('aria-label', `${muted ? 'Unmute' : 'Mute'} ${role}`);
      $(`${role}-volume-value`).textContent = muted ? 'Muted' : `${Math.round(100 * this.volume[role])}%`;
    }
    $('user-volume').title = mapping?.kind === 'reused' ? 'Reused visuals do not replay their original speech audio' : 'Original user recording volume';
  }

  swap() {
    this.bigUser = !this.bigUser;
    $('assistant-layer').className = `video-layer ${this.bigUser ? 'is-pip' : 'is-main'}`;
    $('user-layer').className = `video-layer ${this.bigUser ? 'is-main' : 'is-pip'}`;
  }

  pause() {
    this.wantsPlayback = false;
    this.waitingForMore = false;
    this.autoplay = false;
    this.assistant.pause();
    this.user.pause();
    if (!this.applying) this.render();
  }

  clear() {
    this.applying = true;
    ++this.sourceGeneration;
    this.latest = null;
    this.loaded = null;
    this.chunks = [];
    this.pause();
    cancelAnimationFrame(this.frame);
    ++this.userGeneration;
    this.userSource = null;
    this.userSourceKey = null;
    this.syncedChunk = null;
    this.loading = false;
    this.autoplay = false;
    this.followPosition = 0;
    this.userPlayWarned = false;
    for (const video of [this.assistant, this.user]) { video.removeAttribute('src'); video.load(); }
    this.bigUser = true;
    this.swap();
    this.applying = false;
    $('user-placeholder').hidden = false;
    $('user-placeholder').querySelector('small').textContent = 'Waiting for aligned video';
    $('user-source-label').textContent = 'User · recorded video';
    this.render();
  }

  render() {
    const available = Boolean(this.loaded);
    const paused = this.assistant.paused;
    const waiting = this.waitingForMore && this.wantsPlayback;
    const activePlayback = !paused || waiting || this.autoplay;
    const pending = this.latest && this.loaded && this.latest.version > this.loaded.version;
    const chunk = this.currentChunk();
    $('assistant-placeholder').hidden = available;
    $('play-overlay').hidden = !available || !paused || this.loading || waiting;
    $('video-loading').hidden = !this.loading;
    $('play-toggle').disabled = !available;
    $('seek').disabled = !available || this.assistant.readyState < 1;
    $('play-toggle').innerHTML = icon(activePlayback ? 'pause' : 'play');
    $('play-toggle').setAttribute('aria-label', activePlayback ? 'Pause both videos' : 'Play both videos');
    $('play-clock').textContent = `${time(this.assistant.currentTime)} / ${time(this.ceiling)}`;
    $('seek').max = String(this.ceiling);
    $('seek').value = String(clamp(this.assistant.currentTime || 0, 0, this.ceiling));
    $('seek').setAttribute('aria-valuetext', `${time(this.assistant.currentTime)}, ${this.ceiling.toFixed(2)} seconds loaded`);
    $('player-version').textContent = available ? `v${String(this.loaded.version).padStart(2, '0')}` : '—';
    $('loaded-duration').textContent = available ? `${this.ceiling.toFixed(2)}s` : '—';
    $('selected-chunk').textContent = available && chunk ? code(chunk.global) : '—';
    $('refresh-video').hidden = !available || (!pending && !this.assistant.error);
    $('video-update').classList.toggle('has-update', Boolean(pending));
    $('video-status-dot').className = `dot ${pending ? 'blue' : available ? 'green' : ''}`;
    $('video-status').textContent = !available ? 'Waiting for the first assistant chunk' : this.assistant.error ? 'Video failed to load · select retry' : waiting ? `Played through ${code(this.loaded.end)} · waiting for more media` : pending ? `Update available through ${code(this.latest.end)} · current playback is uninterrupted` : `Loaded through ${code(this.loaded.end)} · ${paused ? 'select play' : 'videos synchronized'}`;
    $('download-video').hidden = !available || (this.loaded.example && !this.loaded.complete);
    if (available) { $('download-video').href = this.loaded.url; $('download-video').download = this.loaded.example ? 'assistant-example.mp4' : `assistant-${this.loaded.chunks}-chunks.mp4`; }
    this.onPosition(available ? chunk?.global ?? null : null, this.assistant.currentTime || 0);
  }
}
