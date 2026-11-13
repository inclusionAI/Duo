'use strict';
const el=id=>document.getElementById(id);
const delay=ms=>new Promise(ok=>setTimeout(ok,ms));
const period=17/24;
let current=null,available=false,mediaResetAvailable=false,sampleAvailable=false;
let presets={},portraitData='';
const fields=['appearance','environment','initial_pose'];
function fullSystemPrompt(){return `<chunk_708> You are the assistant in a real-time full-duplex video call.
The assistant's visual appearance and scene are described as follows:
"${fields.map(f=>el(f).value.trim()).join(' ')}"
Begin every speaking chunk with <speak>. Output <wait> if the assistant is waiting or continuing an unfinished utterance.
In this chunk, the assistant should say:`}
function selectPreset(){const k=el('preset').value;fields.forEach(f=>el(f).value=presets[k][f]||'');portraitData='';el('photo').value='';el('portrait').src='./portraits/'+k+'.jpg';el('systemPrompt').value=presets[k].system_prompt||fullSystemPrompt();promptMode()}
async function loadPresets(){presets=await(await fetch('./presets')).json();for(const [k,p] of Object.entries(presets))el('preset').add(new Option(p.label,k));selectPreset();controls()}
el('preset').onchange=selectPreset;
el('photo').onchange=async()=>{const file=el('photo').files[0];if(!file){portraitData='';el('portrait').src='./portraits/'+el('preset').value+'.jpg';return}
try{const bmp=await createImageBitmap(file),c=document.createElement('canvas'),scale=Math.min(1,1536/Math.max(bmp.width,bmp.height));c.width=Math.round(bmp.width*scale);c.height=Math.round(bmp.height*scale);c.getContext('2d').drawImage(bmp,0,0,c.width,c.height);bmp.close();el('portrait').src=c.toDataURL('image/jpeg',.9);portraitData=el('portrait').src.split(',')[1]}catch{el('status').textContent='无法读取参考图，请重新选择'}};
function promptMode(){const direct=el('promptMode').value==='system';document.querySelectorAll('.structured-setting').forEach(n=>n.hidden=direct);document.querySelectorAll('.system-setting').forEach(n=>n.hidden=!direct);if(direct&&!el('systemPrompt').value.trim())el('systemPrompt').value=fullSystemPrompt()}
el('promptMode').onchange=promptMode;
el('import').onchange=async()=>{try{const data=JSON.parse(await el('import').files[0].text());if(typeof data.system_prompt==='string'){el('systemPrompt').value=data.system_prompt;el('promptMode').value='system'}else{if(!fields.every(f=>typeof data[f]==='string'&&data[f].trim()&&data[f].length<=800))throw Error();fields.forEach(f=>el(f).value=data[f]);el('promptMode').value='fields'}promptMode()}catch{el('status').textContent='设定 JSON 无效，请提供 system_prompt 或三个文本字段'}};
el('randomSeed').onclick=()=>{el('generationSeed').value=crypto.getRandomValues(new Uint32Array(1))[0]};
function personaPacket(){const p={preset:el('preset').value,image:portraitData,orientation:el('orientation').value,prompt_mode:el('promptMode').value};if(p.prompt_mode==='system'){p.system_prompt=el('systemPrompt').value.trim();if(!p.system_prompt)throw Error('请填写 System prompt')}else{for(const f of fields){p[f]=el(f).value.trim();if(!p[f])throw Error('请填写人物、环境和初始姿态')}}
const field=el('generationSeed');
if(!field.validity.valid)throw Error('生成种子必须是 0 到 4294967295 的整数');
if(field.value!=='')p.seed=Number(field.value);
return p}
function controls(){const busy=!!current||!!fileJob;
el('editor').disabled=busy;el('inputVideo').disabled=el('tailSeconds').disabled=busy;
el('live').disabled=busy||!available||!Object.keys(presets).length;
el('sample').disabled=busy||!available||!sampleAvailable||!Object.keys(presets).length;
el('inferFile').disabled=busy||!available||!Object.keys(presets).length||!el('inputVideo').files.length;
el('stop').disabled=!busy;
el('resetMedia').disabled=!mediaResetAvailable||(current?(!current.received||!!current.resetInProgress):!(fileJob&&fileJob.phase==='generating'&&fileJob.completed>0&&(!fileJob.resetTarget||fileJob.resetApplied>=fileJob.resetTarget)));}
async function health(){if(current||fileJob)return;try{const s=await(await fetch('./status',{cache:'no-store'})).json();if(current||fileJob)return;mediaResetAvailable=!!s.media_reset_supported;sampleAvailable=!!s.sample_available;available=s.ready&&!s.busy;el('status').textContent=s.busy?'已有会话进行中':s.ready?'服务就绪':'模型准备中…'}catch{available=false;if(!current)el('status').textContent='服务连接中…'}controls()}
function send(s,obj){if(current===s&&s.ws?.readyState===1)s.ws.send(JSON.stringify(obj))}
function end(s,message){if(current!==s)return;send(s,{type:'session.close'});current=null;clearInterval(s.firstFrameTimer);if(s.decoder&&s.decoder.state!=='closed')s.decoder.close();for(const frame of s.frames)frame.img.close?.();s.ws?.close();s.creditSocket?.close();s.stream?.getTracks().forEach(t=>t.stop());s.audio?.close().catch(()=>{});s.inputAudio?.close().catch(()=>{});const v=el('input');v.classList.remove('live-mirror');v.pause();v.srcObject=null;v.removeAttribute('src');v.load();el('inHint').hidden=false;el('status').textContent=message;controls()}
function encode(arr){const bytes=new Uint8Array(arr.buffer,arr.byteOffset,arr.byteLength);let data='';for(let i=0;i<bytes.length;i+=8192)data+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(data)}
function drawContained(ctx,image,sourceWidth,sourceHeight,width,height){const scale=Math.min(width/sourceWidth,height/sourceHeight),w=sourceWidth*scale,h=sourceHeight*scale;ctx.fillStyle='#000';ctx.fillRect(0,0,width,height);ctx.drawImage(image,(width-w)/2,(height-h)/2,w,h)}
async function cameraBlock(s,n,uploaded=false){const capturedAt=performance.now(),frames=[],target=Math.ceil((n+1)*17*16000/24),start=Math.ceil(n*17*16000/24);
  for(let i=0;i<17;i++){await delay(Math.max(0,s.tick-performance.now()));if(current!==s)return;s.tick+=1000/24;
    const v=el('input');drawContained(s.ctx,v,v.videoWidth,v.videoHeight,s.capture.width,s.capture.height);frames.push(s.capture.toDataURL('image/jpeg',.76).split(',')[1]);
    if(uploaded&&(i%2===1||i===16))send(s,{type:'input_audio_buffer.append',audio:'',format:'pcm_f32le',sample_rate_hz:16000,is_speech:false,video_frames:frames.slice(i===16?16:i-1,i+1)});}
  while(current===s && s.mic.length<target-start)await delay(4);
  if(current!==s)return;
  const pcm=Float32Array.from(s.mic.splice(0,target-start));
  const rms=Math.sqrt(pcm.reduce((a,x)=>a+x*x,0)/pcm.length),speech=rms>.007;
  return {frames,samples:pcm,speech,capturedAt,uploaded}}
async function captureLive(s){
  for(let n=0;current===s;n++){
    if(n===1){
      s.capturing=false;
      while(current===s&&!s.firstFrameShown)await delay(15);
      if(current!==s)return;
      s.mic.length=0;s.tick=performance.now();s.capturing=true;
    }
    const block=await cameraBlock(s,n,s.queue.length===0&&canSubmitInput(s));
    if(!block||current!==s)return;
    s.queue.push(block);
    while(s.queue.length>2){
      const stale=s.queue.findIndex((item,i)=>i<s.queue.length-1&&!item.speech&&!item.uploaded);
      if(stale<0)break;
      s.queue.splice(stale,1);s.droppedInputs=(s.droppedInputs||0)+1;
    }
    if(s.queue.length>=16){
      s.captureStopped=true;s.capturing=false;s.stream.getTracks().forEach(t=>t.stop());
      el('status').textContent='推理落后，采集已暂停，正在播完已录内容';return;
    }
    el('inputlabel').textContent=(block.speech?'检测到语音':'当前安静')+' · 待提交 '+s.queue.length+' 块';
  }
}
async function produce(s){let start=performance.now();let previewStarted=false;
  if(s.mode==='live'){s.queue=[];s.capturing=true;captureLive(s).catch(err=>end(s,err.message))}
  for(let n=0;current===s&&(s.mode==='live'||n<s.count);n++){
    if(n===1){
      el('inputlabel').textContent='等待首个画面，暂停输入';
      if(s.mode==='sample')el('input').pause();
      while(current===s&&!s.firstFrameShown)await delay(15);
      if(current!==s)return;
      el('inputlabel').textContent=s.mode==='live'?'摄像头与麦克风':'内置对话样片';
      if(s.mode==='sample'){await el('input').play();start=performance.now()-period*1000}
    }
    while(current===s&&!canSubmitInput(s))await delay(15);
    if(current!==s)return;
    let block;
    if(s.mode==='sample'){
      if(!previewStarted){await el('input').play();previewStarted=true}

      await delay(Math.max(0,start+(n+1)*period*1000-performance.now()));
      if(current!==s)return;
      while(current===s&&!canSubmitInput(s))await delay(15);if(current!==s)return;
      s.captureTimes.push(performance.now()-period*1000);s.submitted.push(performance.now());s.sent++;send(s,{type:'demo.sample_block',index:n});continue;
    }else{
      while(current===s&&!(block=nextInputBlock(s.queue,el('catchup').checked,s.audio,s.captureStopped))){
        if(s.captureStopped&&!s.queue.length){s.captureCount=s.sent;return}await delay(8)
      }
      if(block?.accelerated)s.inputCatchups=(s.inputCatchups||0)+1;
    }
    if(!block||current!==s)return;
    while(current===s&&!canSubmitInput(s))await delay(15);if(current!==s)return;
    if(!block.uploaded)for(let i=0;i<17;i+=2)send(s,{type:'input_audio_buffer.append',audio:'',format:'pcm_f32le',sample_rate_hz:16000,is_speech:false,video_frames:block.frames.slice(i,i+2)});
    if(block.speech)send(s,{type:'input_audio_buffer.append',audio:encode(block.samples),format:'pcm_f32le',sample_rate_hz:16000,is_speech:true,video_frames:[]});
    else send(s,{type:'demo.record_audio',audio:encode(block.samples)});
    s.captureTimes.push(block.capturedAt);s.submitted.push(performance.now());s.sent++;send(s,{type:'input_audio_buffer.commit',final:true})
  }}
function unpackPacket(buffer){
  const headerLength=new DataView(buffer).getUint32(0,true);
  const packet=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,headerLength)));
  let offset=4+headerLength;
  packet.pcmBytes=new Uint8Array(buffer,offset,packet.audio_byte_length);offset+=packet.audio_byte_length;
  for(const frame of packet.video.frames){frame.encodedBytes=new Uint8Array(buffer,offset,frame.byte_length);offset+=frame.byte_length}
  return packet;
}
async function playback(s,p){const bytes=p.pcmBytes||Uint8Array.from(atob(p.delta),c=>c.charCodeAt(0)),pcm=new Int16Array(bytes.slice().buffer);
  const ctx=s.audio;let buf=ctx.createBuffer(2,pcm.length/2,32000);
  for(let ch=0;ch<2;ch++){const data=buf.getChannelData(ch);for(let j=0;j<data.length;j++)data[j]=pcm[j*2+ch]/32768}
  const images=await decodeVideoFrames(s,p.video);
  if(current!==s)return;
  const backlog=Math.max(0,s.playEnd-ctx.currentTime);
  const rate=el('catchup').checked&&backlog>.10?1+Math.min(.12,(backlog-.08)*.25):1;
  const originalLength=buf.length;buf=catchupAudio(buf,rate,ctx);
  const effectiveRate=originalLength/buf.length;
  const t=Math.max(ctx.currentTime+.045,s.playEnd);s.playEnd=t+buf.duration;s.playWait[p.video.chunk_index]=Math.max(0,t-ctx.currentTime)*1000;
  el('buffer').textContent='播放前等待 '+Math.round(Math.max(0,t-ctx.currentTime)*1000)+' ms · 含当前块缓冲 '+Math.round(Math.max(0,s.playEnd-ctx.currentTime)*1000)+' ms'+(effectiveRate>1.001?' · 追赶 '+effectiveRate.toFixed(2)+'×':'');

  const src=ctx.createBufferSource();src.buffer=buf;src.connect(ctx.destination);src.start(t);
  images.forEach((img,i)=>s.frames.push({t:t+i/(24*effectiveRate),img,capturedAt:s.captureTimes[p.video.chunk_index]+i*1000/24,block:p.video.chunk_index}));
  const block=p.video.chunk_index;
  src.onended=()=>{if(current!==s)return;send(s,{type:'playback.ack',played_ms:Math.floor((block+1)*period*1000),committed_ms:Math.floor((block+1)*period*1000),response_id:p.response_id});if(s.mode==='sample'&&block+1===s.count)end(s,'示例播放完毕');if(s.captureCount&&block+1===s.captureCount)end(s,'已播完全部采集内容')};
}
function paint(s){if(current!==s)return;let frame;while(s.frames.length&&s.frames[0].t<=s.audio.currentTime){frame?.img.close?.();frame=s.frames.shift();}if(frame){const canvas=el('output'),width=frame.img.displayWidth??frame.img.naturalWidth??frame.img.width,height=frame.img.displayHeight??frame.img.naturalHeight??frame.img.height;if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height}drawContained(canvas.getContext('2d'),frame.img,width,height,width,height);frame.img.close?.();el('outHint').hidden=true;s.firstFrameShown=true;
    if(s.displayedBlock!==frame.block){
      s.displayedBlock=frame.block;
      const latency=Math.round(performance.now()-frame.capturedAt),buffered=Math.round(Math.max(0,s.playEnd-s.audio.currentTime)*1000),queued=s.queue?.length||0,inflight=s.sent-s.received;
      el('latency').textContent='画面延迟 '+latency+' ms · 待提交 '+queued+' 块 · 处理中 '+inflight+' 块';
      send(s,{type:'demo.client_stats',block:frame.block,display_latency_ms:latency,playback_buffer_ms:buffered,input_queue_blocks:queued,input_inflight_blocks:inflight,submit_to_receive_ms:Math.round(s.receivedAt[frame.block]-s.submitted[frame.block]),capture_to_submit_ms:Math.round(s.submitted[frame.block]-s.captureTimes[frame.block]),receive_to_display_ms:Math.round(performance.now()-s.receivedAt[frame.block]),scheduled_play_wait_ms:Math.round(s.playWait[frame.block]||0),input_catchup_blocks:s.inputCatchups||0,dropped_silent_inputs:s.droppedInputs||0});
    }
  }requestAnimationFrame(()=>paint(s))}
async function start(mode){if(current||fileJob)return;el('streams').hidden=false;el('fileResultBox').hidden=true;el('fileResult').pause();el('comparisonResult').pause();const s={mode,sent:0,received:0,creditThrough:0,resetGeneration:0,submitted:[],receivedAt:[],playWait:[],captureTimes:[],playEnd:0,mic:[],frames:[],chain:Promise.resolve()};current=s;controls();el('outHint').hidden=false;el('status').textContent='正在建立会话…';el('stats').textContent='等待首个音视频块';el('output').getContext('2d').clearRect(0,0,el('output').width,el('output').height);
  try{const persona=personaPacket();s.audio=new AudioContext();await s.audio.resume();s.h264=await h264Supported();const v=el('input');v.controls=false;
    if(mode==='live'){
      v.classList.add('live-mirror');
      const videoConstraints={facingMode:{ideal:'user'},frameRate:{ideal:24,max:30}};
      if(navigator.mediaDevices.getSupportedConstraints?.().resizeMode)videoConstraints.resizeMode={ideal:'none'};
      s.stream=await navigator.mediaDevices.getUserMedia({video:videoConstraints,audio:{echoCancellation:true,noiseSuppression:true,channelCount:1}});
      if(current!==s){s.stream.getTracks().forEach(t=>t.stop());return}
      v.muted=true;v.srcObject=s.stream;await v.play();
      s.capture=document.createElement('canvas');const scale=Math.min(1,320/v.videoWidth,384/v.videoHeight);s.capture.width=Math.max(2,Math.round(v.videoWidth*scale/2)*2);s.capture.height=Math.max(2,Math.round(v.videoHeight*scale/2)*2);s.ctx=s.capture.getContext('2d');
      s.inputAudio=new AudioContext();await s.inputAudio.audioWorklet.addModule('./capture.js');await s.inputAudio.resume();
      const source=s.inputAudio.createMediaStreamSource(s.stream),mic=new AudioWorkletNode(s.inputAudio,'mic-capture'),mute=s.inputAudio.createGain();mute.gain.value=0;source.connect(mic).connect(mute).connect(s.inputAudio.destination);
      mic.port.onmessage=e=>{if(s.capturing)s.mic.push(...e.data)};s.inputNodes=[source,mic,mute]
    }else{v.classList.remove('live-mirror');v.srcObject=null;v.src='./example.mp4';v.muted=false;v.volume=.45;v.load()}
    el('inputlabel').textContent=mode==='live'?'摄像头与麦克风':'内置对话样片';el('inHint').hidden=true;
    s.count=(await(await fetch('./status')).json()).blocks;if(current!==s)return;
    const ws=s.ws=new WebSocket(new URL('./ws?setup=1&binary=1&credits=1'+(s.h264?'&video=h264':''),location.href).href.replace(/^http/,'ws'));
    ws.binaryType='arraybuffer';ws.onopen=()=>{send(s,persona);el('status').textContent='正在准备人物和模型…'};
    ws.onmessage=e=>{if(current!==s)return;const p=typeof e.data==='string'?JSON.parse(e.data):unpackPacket(e.data);
      if(Number.isSafeInteger(p.server_event_seq))send(s,{type:'session.event_ack',server_event_seq:p.server_event_seq});
      if(p.type==='demo.recording_warning')el('recordingStatus').textContent='录像保存异常：'+p.message;
      if(p.type==='demo.recording'){s.recordingId=p.id;sessionStorage.setItem('duo-call-recording:'+location.pathname,p.id);watchRecording(p.id);if(p.input_credit_key)connectInputCredits(s,p.input_credit_key)}
      if(p.type==='session.updated'&&(!s.creditSocket||p.session?.metadata?.media_reset_generation===String(s.pendingReset)))acceptMediaReset(s);
      if(p.type==='demo.preparing'&&!s.sessionReady)el('status').textContent=p.message+'… '+p.elapsed_s+' 秒';
      if(p.type==='session.created'){s.sessionReady=true;s.firstFrameStarted=performance.now();el('status').textContent='正在准备首个画面…';s.firstFrameTimer=setInterval(()=>{if(current===s&&!s.received)el('status').textContent='正在准备首个画面… '+Math.floor((performance.now()-s.firstFrameStarted)/1000)+' 秒'},1000);s.tick=performance.now();s.mic.length=0;paint(s);produce(s).catch(err=>end(s,err.message))}
      if(p.type==='response.output_audio.delta'&&p.video){clearInterval(s.firstFrameTimer);el('status').textContent='会话进行中';const n=s.received++;s.receivedAt[n]=performance.now();
      if((p.video.timings?.realization_clean||p.video.timings?.realization_latent)?.media_cache_reset){s.resetInProgress=false;el('resetStatus').textContent='已在第 '+(p.video.chunk_index+1)+' 块前重置 Realization / Codec，保留 Intent 对话历史'}controls();el('stats').textContent='已返回 '+s.received+' 块 · 最近一块 '+Math.round(performance.now()-s.submitted[n])+' ms';s.chain=s.chain.then(()=>playback(s,p)).catch(err=>end(s,'播放失败：'+err.message))}
      if(p.type==='error')end(s,p.error?.message||'服务返回错误')};
    ws.onerror=()=>end(s,'连接失败，请确认服务就绪后重试');ws.onclose=()=>end(s,'会话已关闭')
  }catch(err){end(s,'无法开始：'+err.message)}}
el('fileResult').onplay=()=>el('comparisonResult').pause();el('comparisonResult').onplay=()=>el('fileResult').pause();el('live').onclick=()=>start('live');el('sample').onclick=()=>start('sample');el('stop').onclick=()=>fileJob?cancelFileInference():current&&end(current,'会话已结束');el('resetMedia').onclick=resetMediaCache;el('inputVideo').onchange=selectInputVideo;el('inferFile').onclick=startFileInference;window.addEventListener('pagehide',()=>current&&end(current,'已关闭'));controls();loadPresets().catch(()=>el('status').textContent='预设加载失败');resumeFileInference();health();setInterval(health,5000);

async function watchRecording(id){
  el('recordingBox').hidden=false;['recordingInputDownload','recordingOutputDownload','recordingDownload'].forEach(id=>el(id).hidden=true);
  for(;;){
    try{
      const r=await fetch('./recordings/'+encodeURIComponent(id),{cache:'no-store'});
      if(!r.ok)throw Error('HTTP '+r.status);
      const data=await r.json();
      if(sessionStorage.getItem('duo-call-recording:'+location.pathname)!==id)return;
      if(data.error){el('recordingStatus').textContent='录像保存失败：'+data.error;return}
      if(data.status==='done'){
        el('recordingStatus').textContent=(data.partial?'部分通话录像已保存':'通话录像已保存')+' · 输入 '+(data.input_blocks||0)+' 块，生成 '+(data.output_blocks||0)+' 块';
        const base='./recordings/'+encodeURIComponent(id);
        el('recordingInputDownload').href=base+'/input.mp4';el('recordingInputDownload').hidden=!data.input_blocks;
        el('recordingOutputDownload').href=base+'/output.mp4';el('recordingOutputDownload').hidden=!data.output_blocks;
        el('recordingDownload').href=base+'/comparison.mp4';el('recordingDownload').hidden=!data.output_blocks;return;
      }
      if(data.status==='failed'){el('recordingStatus').textContent='录像保存失败：'+data.error;return}
      el('recordingStatus').textContent={recording:'通话录制中',queued:'等待录像后台处理',encoding:'正在生成通话录像',archiving:'正在归档通话录像',recovering:'正在恢复通话录像'}[data.status]||'录像处理中';
    }catch(error){el('recordingStatus').textContent='正在重新连接录像状态…'}
    await delay(2000);
  }
}
const previousRecording=sessionStorage.getItem('duo-call-recording:'+location.pathname);
if(previousRecording)watchRecording(previousRecording);
