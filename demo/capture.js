const CAMERA_ERRORS = {
  NotAllowedError: 'Camera or microphone access was denied. Allow access in the site settings and try again.',
  NotFoundError: 'No camera or microphone was found. Connect a device and try again.',
  NotReadableError: 'Another application is using the device. Release it and try again.',
  OverconstrainedError: 'The device does not support the requested recording settings. Try another camera.',
};

export class CameraCapture {
  constructor(video, {onChange, onComplete, onError}) {
    this.video = video;
    this.onChange = onChange;
    this.onComplete = onComplete;
    this.onError = onError;
    this.stream = null;
    this.recorder = null;
    this.error = '';
    this.pending = false;
    this.saving = false;
    this.generation = 0;
  }

  get ready() { return Boolean(this.stream?.getVideoTracks().some(track => track.readyState === 'live') && this.stream?.getAudioTracks().some(track => track.readyState === 'live')); }
  get recording() { return this.recorder?.state === 'recording'; }
  get duration() { return this.startedAt ? (performance.now() - this.startedAt) / 1000 : 0; }
  get settings() { return this.stream?.getVideoTracks()[0]?.getSettings() || {}; }

  async request() {
    if (this.ready || this.pending) return;
    this.error = '';
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      this.error = 'Open this page over HTTPS in a browser with camera support.';
      this.onChange();
      return;
    }
    const generation = ++this.generation;
    this.pending = true;
    this.onChange();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640}, height: {ideal: 480}, frameRate: {ideal: 24}}, audio: true});
      if (generation !== this.generation) { stream.getTracks().forEach(track => track.stop()); return; }
      this.stream?.getTracks().forEach(track => track.stop());
      this.stream = stream;
      this.video.srcObject = stream;
      this.video.muted = true;
      await this.video.play().catch(() => {});
      stream.getTracks().forEach(track => track.addEventListener('ended', () => {
        if (this.stream !== stream) return;
        this.error = 'The device disconnected and recording stopped. Enable the camera again.';
        if (this.recording) this.stop();
        this.onChange();
      }));
      if (typeof MediaRecorder !== 'function') this.error = 'This browser can preview the camera but cannot record video. Use a recent Chrome, Edge, or Safari.';
    } catch (error) {
      if (generation === this.generation) this.error = CAMERA_ERRORS[error.name] || `Could not open the device: ${error.message}`;
    } finally {
      if (generation === this.generation) { this.pending = false; this.onChange(); }
    }
  }

  start(capture) {
    if (!this.ready) throw new Error('Enable the camera and microphone first.');
    if (typeof MediaRecorder !== 'function') throw new Error('This browser does not support video recording.');
    if (this.recorder || this.saving) throw new Error('The previous recording has not finished saving.');
    const mimeType = ['video/webm;codecs=vp8,opus', 'video/webm', 'video/mp4'].find(type => MediaRecorder.isTypeSupported(type));
    const recorder = new MediaRecorder(this.stream, {...(mimeType ? {mimeType} : {}), videoBitsPerSecond: 1800000, audioBitsPerSecond: 96000});
    const pieces = [];
    const run = {capture, discarded: false, endedAt: null};
    this.run = run;
    this.recorder = recorder;
    this.startedAt = performance.now();
    this.error = '';
    const startedAt = this.startedAt;
    const settings = this.settings;
    recorder.ondataavailable = event => { if (event.data.size && !run.discarded) pieces.push(event.data); };
    recorder.onerror = event => {
      this.onError(event.error?.message || 'Recording was interrupted. Try again.');
      if (recorder.state === 'recording') this.stop();
    };
    recorder.onstop = async () => {
      if (this.recorder === recorder) this.recorder = null;
      if (run.discarded) { this.onChange(); return; }
      this.saving = true;
      this.onChange();
      try {
        const blob = new Blob(pieces, {type: recorder.mimeType || mimeType || 'video/webm'});
        const duration = Math.max(0, ((run.endedAt || performance.now()) - startedAt) / 1000);
        const metadata = {duration, mimeType: blob.type, width: settings.width, height: settings.height,
          fps: settings.frameRate, startedAt: capture.startedAt, hasAudio: true};
        await this.onComplete(blob, metadata, capture);
      } catch (error) {
        this.onError(error.message || 'The recording could not be saved. Try again.');
      } finally {
        this.saving = false;
        this.startedAt = null;
        this.onChange();
      }
    };
    try {
      recorder.start(1000);
    } catch (error) {
      this.recorder = null;
      this.run = null;
      this.startedAt = null;
      throw error;
    }
    this.onChange();
  }

  stop() {
    if (!this.recorder || this.recorder.state !== 'recording') return;
    this.run.endedAt = performance.now();
    this.recorder.stop();
    this.onChange();
  }

  abortRecording() {
    if (this.run) this.run.discarded = true;
    if (this.recorder?.state === 'recording') this.recorder.stop();
    this.recorder = null;
    this.run = null;
    this.startedAt = null;
    this.onChange();
  }

  close() {
    ++this.generation;
    this.abortRecording();
    this.stream?.getTracks().forEach(track => track.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.pending = false;
    this.error = '';
    this.onChange();
  }
}
