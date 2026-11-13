'use strict';
let fileJob=null,filePreviewUrl=null;
const fileStorageKey='duo-video-job:'+location.pathname;
const filePhases={waiting_upload:'等待上传',uploading:'上传视频',preparing:'解码与准备输入',aligning:'整段语音识别与对齐',generating:'生成音视频',encoding:'封装 MP4',cancelling:'正在取消',cancelled:'已取消',done:'生成完成',failed:'生成失败'};
async function fileJson(response){
  if(!response.ok){
    const message=response.status>=500?'服务连接暂时异常（HTTP '+response.status+'）':await response.text();
    const error=Error(message||'视频请求失败');error.status=response.status;throw error;
  }
  return response.json();
}
async function readFileStatus(identity){
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),10000);
  try{return await fileJson(await fetch('./video-jobs/'+encodeURIComponent(identity),{cache:'no-store',signal:controller.signal}))}
  finally{clearTimeout(timer)}
}
async function watchFileJob(state){
  while(fileJob===state&&!state.cancelled){
    let progress;
    try{progress=await readFileStatus(state.id)}
    catch(error){
      if(error.status&&error.status<500&&![408,429].includes(error.status))throw error;
      el('fileProgress').textContent='状态连接暂时中断，服务器任务继续；正在重新查询…';
      console.warn('视频状态查询重试',state.id,error.message);
      await delay(1000);continue;
    }
    if(state.cancelled)return;
    state.phase=progress.status;state.completed=progress.completed_blocks||0;state.resetApplied=progress.reset_applied||0;controls();
    if(state.resetTarget&&state.resetApplied>=state.resetTarget)el('resetStatus').textContent='已在发布边界重置 Realization / Codec，保留 Intent 对话历史';
    el('fileProgress').textContent=filePhases[progress.status]+(progress.total_blocks?' · '+progress.completed_blocks+' / '+progress.total_blocks+' 块':'')+' · '+progress.elapsed_s+' 秒';
    if(progress.status==='done'){
      const base='./video-jobs/'+encodeURIComponent(state.id);
      el('downloadInputVideo').href=base+'/input.media';
      el('fileResult').src=base+'/result.mp4';el('downloadVideo').href=base+'/result.mp4?download=1';
      el('comparisonResult').src=base+'/comparison.mp4';el('downloadComparison').href=base+'/comparison.mp4?download=1';
      el('fileResultBox').hidden=false;
      el('fileProgress').textContent='生成完成 · 输出 '+progress.output_seconds.toFixed(1)+' 秒'+(progress.reply_limit_reached?' · 已到回复上限':'');
      return;
    }
    if(progress.status==='waiting_upload'){
      await fetch('./video-jobs/'+state.id,{method:'DELETE'});
      throw Error('视频上传尚未完成，请重新提交');
    }
    if(progress.status==='failed')throw Error(progress.error||'生成失败');
    if(progress.status==='cancelled'){el('fileProgress').textContent='已取消';return}
    await delay(1000);
  }
}
function uploadFile(state,file){
  return new Promise((resolve,reject)=>{
    const xhr=state.upload=new XMLHttpRequest();
    xhr.open('PUT','./video-jobs/'+state.id+'/input');
    xhr.upload.onprogress=e=>{if(e.lengthComputable&&fileJob===state)el('fileProgress').textContent='上传视频 '+Math.round(e.loaded/e.total*100)+'%'};
    xhr.onload=()=>{state.upload=null;if(xhr.status>=200&&xhr.status<300)resolve();else reject(Error('上传连接返回 HTTP '+xhr.status))};
    xhr.onerror=()=>reject(Error('视频上传连接中断'));
    xhr.onabort=()=>reject(Error('已取消'));
    xhr.send(file);
  });
}
async function startFileInference(){
  if(current||fileJob)return;
  const file=el('inputVideo').files[0];if(!file)return;
  if(file.size>1024**3){el('fileProgress').textContent='视频文件请控制在 1 GB 以内';return}
  selectInputVideo();
  const state={id:null,cancelled:false,upload:null};
  fileJob=state;controls();el('streams').hidden=true;el('input').pause();el('fileResult').pause();el('comparisonResult').pause();el('fileResultBox').hidden=true;el('fileProgress').textContent='正在建立视频任务…';
  try{
    const setup={...personaPacket(),tail_seconds:Number(el('tailSeconds').value)};
    const created=await fileJson(await fetch('./video-jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(setup)}));
    state.id=created.id;sessionStorage.setItem(fileStorageKey,state.id);
    if(state.cancelled){await waitFileCancellation(state);return}
    try{await uploadFile(state,file)}
    catch(error){
      if(state.cancelled)return;
      el('fileProgress').textContent=error.message+'，正在核对服务器任务状态…';
    }
    await watchFileJob(state);
  }catch(error){if(!state.cancelled)el('fileProgress').textContent=error.message}
  finally{if(fileJob===state&&(!state.cancelled||!state.id)){fileJob=null;controls();health()}}
}
async function waitFileCancellation(state,message='已取消'){
  try{
    await fileJson(await fetch('./video-jobs/'+state.id,{method:'DELETE'}));
    for(;;){
      let result;
      try{result=await readFileStatus(state.id)}
      catch(error){
        if(error.status&&error.status<500&&![408,429].includes(error.status))throw error;
        await delay(1000);continue;
      }
      if(['done','failed','cancelled'].includes(result.status))break;
      await delay(250);
    }
    el('fileProgress').textContent=message;
  }catch(error){el('fileProgress').textContent='取消状态未确认，可刷新恢复任务：'+error.message}
  finally{if(fileJob===state){fileJob=null;controls();health()}}
}
async function cancelFileInference(){
  const state=fileJob;if(!state||state.cancelled)return;
  state.cancelled=true;el('fileProgress').textContent='正在取消…';
  state.upload?.abort();
  if(state.id)await waitFileCancellation(state);
}
async function resumeFileInference(){
  if(current||fileJob)return;
  const identity=new URLSearchParams(location.search).get('video_job')||sessionStorage.getItem(fileStorageKey);
  if(!identity)return;
  const state={id:identity,cancelled:false,upload:null};fileJob=state;controls();
  sessionStorage.setItem(fileStorageKey,identity);
  el('streams').hidden=true;el('fileProgress').textContent='正在恢复视频任务…';
  try{await watchFileJob(state)}
  catch(error){el('fileProgress').textContent='无法恢复任务：'+error.message}
  finally{if(fileJob===state&&!state.cancelled){fileJob=null;controls();health()}}
}
function selectInputVideo(){
  if(filePreviewUrl)URL.revokeObjectURL(filePreviewUrl);
  const file=el('inputVideo').files[0];
  if(file&&!current&&!fileJob){
    el('streams').hidden=false;filePreviewUrl=URL.createObjectURL(file);
    const input=el('input');input.classList.remove('live-mirror');input.srcObject=null;input.src=filePreviewUrl;input.controls=true;input.muted=true;input.load();
    el('inHint').hidden=true;el('inputlabel').textContent=file.name;
  }
  controls();
}

async function resetMediaCache(){
  if(fileJob){
    const state=fileJob;state.resetTarget=Infinity;controls();
    try{
      const result=await fileJson(await fetch('./video-jobs/'+state.id+'/reset-media',{method:'POST'}));
      state.resetTarget=result.reset_requested;
      el('resetStatus').textContent='已请求重置，等待已提交的视频块发布完成';
    }catch(error){state.resetTarget=0;el('resetStatus').textContent=error.message}
    controls();return;
  }
  const s=current;if(!s||s.resetInProgress)return;
  s.resetRequested=s.resetInProgress=true;controls();
  el('resetStatus').textContent='等待已提交的视频块发布完成…';
  while(current===s&&s.sent>releasedInputs(s))await delay(15);
  if(current!==s)return;
  s.pendingReset=++s.resetGeneration;
  send(s,{type:'demo.reset_media'});
}

function acceptMediaReset(s){
  if(!s.pendingReset)return;
  s.pendingReset=null;
  s.resetRequested=false;
  el('resetStatus').textContent='已到发布边界，将在下一块开始前重置';
  controls();
}
