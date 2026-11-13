# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Receive engine events and acknowledge receipt for bounded replay."""


async def receive_and_ack(engine, *, timeout):
    event = await engine.receive_json(timeout=timeout)
    seq = event.get("server_event_seq")
    if seq is not None:
        await engine.send_json({"type": "session.event_ack", "server_event_seq": seq})
    return event
