# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HTTP lifecycle for one file-inference task at a time."""

import asyncio
import json
import shutil

from aiohttp import web

UPLOAD_START_SECONDS = 30
UPLOAD_IDLE_SECONDS = 30
UPLOAD_TOTAL_SECONDS = 600
UPLOAD_MAX_BYTES = 1 << 30


def install_file_routes(app, jobs, claim, release):
    def discard_upload(job):
        (job.folder / "input.media").unlink(missing_ok=True)
        shutil.rmtree(job.work, ignore_errors=True)
        (job.folder / "job.json").write_text(json.dumps(job.public(), ensure_ascii=False))

    async def end_upload(job, status, error):
        job.status, job.error = status, error
        job.upload_task = None
        release(job.id)
        try:
            await asyncio.to_thread(discard_upload, job)
        except OSError:
            app.logger.exception("Unable to clean up video upload %s", job.id)

    async def expire_upload(job):
        await asyncio.sleep(UPLOAD_START_SECONDS)
        await end_upload(job, "failed", "等待视频上传超时")

    async def create(request):
        try:
            setup = await asyncio.wait_for(request.json(), UPLOAD_IDLE_SECONDS)
        except TimeoutError as exc:
            raise web.HTTPRequestTimeout(text="读取视频任务设置超时") from exc
        claim(None, request)
        creation = asyncio.create_task(asyncio.to_thread(jobs.create, setup))
        try:
            job = await asyncio.shield(creation)
        except asyncio.CancelledError:
            release(None)
            (job,) = await asyncio.gather(creation, return_exceptions=True)
            if not isinstance(job, BaseException):
                await end_upload(job, "cancelled", "视频上传已取消")
            raise
        except Exception:
            release(None)
            raise
        claim(job.id)
        job.upload_task = asyncio.create_task(expire_upload(job))
        return web.json_response(job.public(), status=201)

    def finished(job, task):
        if task.cancelled():
            job.status = "cancelled"
            asyncio.create_task(asyncio.to_thread(shutil.rmtree, job.work, ignore_errors=True))
        release(job.id)

    async def upload(request):
        job = jobs.get(request.match_info["identity"])
        if job.status != "waiting_upload":
            raise web.HTTPConflict(text="该任务不接受上传")
        job.upload_task.cancel()
        job.upload_task = asyncio.current_task()
        job.status = "uploading"
        try:
            async with asyncio.timeout(UPLOAD_TOTAL_SECONDS):
                output = (job.folder / "input.media").open("wb")
                try:
                    while chunk := await asyncio.wait_for(request.content.read(256 << 10), UPLOAD_IDLE_SECONDS):
                        job.upload_bytes += len(chunk)
                        if job.upload_bytes > UPLOAD_MAX_BYTES:
                            raise web.HTTPRequestEntityTooLarge(max_size=UPLOAD_MAX_BYTES, actual_size=job.upload_bytes)
                        write = asyncio.create_task(asyncio.to_thread(output.write, chunk))
                        try:
                            await asyncio.shield(write)
                        except asyncio.CancelledError:
                            release(job.id)
                            await asyncio.gather(write, return_exceptions=True)
                            raise
                finally:
                    await asyncio.to_thread(output.close)
            job.check_cancel()
            if not job.upload_bytes:
                raise web.HTTPBadRequest(text="视频文件为空")
            job.status = "preparing"
            job.upload_task = None
            job.task = asyncio.create_task(jobs.run(job))
            job.task.add_done_callback(lambda task: finished(job, task))
            return web.json_response(job.public(), status=202)
        except TimeoutError as exc:
            message = "视频上传超时，请重新提交"
            await end_upload(job, "failed", message)
            raise web.HTTPRequestTimeout(text=message) from exc
        except asyncio.CancelledError:
            await end_upload(job, "cancelled", "视频上传已取消")
            raise
        except Exception as exc:
            await end_upload(job, "cancelled" if job.cancelled.is_set() else "failed", str(exc))
            raise

    async def status(request):
        return web.json_response(jobs.get(request.match_info["identity"]).public())

    async def result(request):
        job = jobs.get(request.match_info["identity"])
        if job.status != "done":
            raise web.HTTPConflict(text="视频尚未生成完成")
        comparison = request.match_info["kind"] == "comparison"
        name = "comparison" if comparison else "output"
        headers = (
            {"Content-Disposition": f'attachment; filename="duo-{name}-{job.id[:8]}.mp4"'}
            if request.query.get("download")
            else {}
        )
        return web.FileResponse(job.folder / (name + ".mp4"), headers=headers)

    async def input_video(request):
        job = jobs.get(request.match_info["identity"])
        path = job.folder / "input.media"
        if not await asyncio.to_thread(path.is_file):
            raise web.HTTPNotFound(text="原始输入不存在")
        return web.FileResponse(
            path, headers={"Content-Disposition": f'attachment; filename="duo-input-{job.id[:8]}.media"'}
        )

    async def reset_media(request):
        job = jobs.get(request.match_info["identity"])
        if job.status != "generating":
            raise web.HTTPConflict(text="累积误差重置仅适用于正在生成的视频")
        job.reset_requested += 1
        return web.json_response(job.public(), status=202)

    async def cancel(request):
        job = jobs.get(request.match_info["identity"])
        if job.status in ("done", "failed", "cancelled", "cancelling"):
            return web.json_response(job.public())
        job.cancelled.set()
        if job.task is not None:
            job.status = "cancelling"
            job.task.cancel()
        else:
            uploading = job.status == "uploading"
            job.upload_task.cancel()
            if uploading:
                job.status = "cancelling"
            else:
                await end_upload(job, "cancelled", "视频上传已取消")
        return web.json_response(job.public())

    async def cleanup(app):
        tasks = []
        for job in jobs.jobs.values():
            job.cancelled.set()
            task = job.task or job.upload_task
            if task and not task.done():
                task.cancel()
                tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

    app.on_cleanup.append(cleanup)
    app.router.add_post("/video-jobs", create)
    app.router.add_put("/video-jobs/{identity}/input", upload)
    app.router.add_get("/video-jobs/{identity}", status)
    app.router.add_get("/video-jobs/{identity}/input.media", input_video)
    app.router.add_get("/video-jobs/{identity}/{kind:result|comparison}.mp4", result)
    app.router.add_delete("/video-jobs/{identity}", cancel)
    app.router.add_post("/video-jobs/{identity}/reset-media", reset_media)
