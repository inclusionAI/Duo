# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Report slow portrait setup and cancel it when the client disconnects."""

import asyncio
import time


async def prepare_persona(ws, setup, prepare):
    started = time.monotonic()
    message = "正在编码参考图片，首次需要加载编码器" if setup.get("image") else "正在设置人物提示词"
    await ws.send_json(dict(type="demo.preparing", message=message, elapsed_s=0))
    preparation = asyncio.create_task(prepare(setup))
    # Observe cancellation while a cold encoder is loading instead of keeping
    # an abandoned preparation alive until its GPU work completes.
    incoming = asyncio.create_task(ws.receive())
    try:
        while True:
            done, _ = await asyncio.wait((preparation, incoming), timeout=2, return_when=asyncio.FIRST_COMPLETED)
            if incoming in done:
                raise ConnectionResetError("准备期间通话已关闭")
            if preparation in done:
                return await preparation
            await ws.send_json(dict(type="demo.preparing", message=message, elapsed_s=int(time.monotonic() - started)))
    finally:
        for task in (preparation, incoming):
            if not task.done():
                task.cancel()
        await asyncio.gather(preparation, incoming, return_exceptions=True)
