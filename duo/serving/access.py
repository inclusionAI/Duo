"""Server-owned FIFO access, independent of the inference request queue.

The State lock also guards authorization and mutations. A handoff revokes the
old turn immediately; the engine then resets at its next published chunk.
"""

import copy
import os
import secrets
import time
import uuid

IDLE_MS = 5 * 60 * 1000
DISCONNECT_GRACE_MS = 10 * 1000
USER_MODEL = "duo"
USER_CASE = "default"


class AccessError(ValueError):
    def __init__(self, message, status=403):
        super().__init__(message)
        self.status = status


def uid():
    return uuid.uuid4().hex


class AccessQueue:
    def __init__(self, state, clock=None):
        self.state = state
        self.clock = clock or (lambda: time.time() * 1000)
        self.members = []
        self.removals = []
        self.active_id = None
        self.turn = None
        self.session_id = uid()
        self.created_at = self.clock()
        self.capture = None
        self.hidden_stream = None
        self.revision = 0

    def changed(self):
        self.revision += 1
        self.state.changed.notify_all()

    def invalidate(self):
        self.session_id = uid()
        self.created_at = self.clock()
        self.capture = None
        self.hidden_stream = self.state.stream_id
        self.changed()

    def promote(self):
        if any(m["id"] == self.active_id for m in self.members):
            return
        next_member = self.members[0] if self.members else None
        if not next_member and self.active_id is None and self.turn is None:
            return
        self.active_id = next_member["id"] if next_member else None
        now = self.clock()
        self.turn = (
            dict(id=uid(), offeredAt=now, acceptedAt=None, lastActivityAt=now)
            if next_member
            else None
        )
        self.invalidate()
        # A newer handoff supersedes an older one, including while weights load.
        self.state.handoff = dict(
            id=uid(),
            session_id=self.session_id,
            model_id=USER_MODEL if next_member and next_member["role"] == "user" else None,
            case_id=USER_CASE if next_member and next_member["role"] == "user" else None,
        )
        self.changed()

    def remove(self, member, reason, by=None):
        self.members.remove(member)
        self.removals.append(dict(memberId=member["id"], reason=reason, by=by, at=self.clock()))
        self.removals = self.removals[-128:]
        self.promote()
        self.changed()

    def sweep(self):
        now = self.clock()
        for member in list(self.members):
            departure = member.get("disconnectedAt")
            if departure is not None and now - max(departure, member["seenAt"]) >= DISCONNECT_GRACE_MS:
                self.remove(member, "Connection closed. Join the queue again.")
            elif (
                member["id"] == self.active_id
                and self.turn
                and now - self.turn["lastActivityAt"] >= IDLE_MS
            ):
                self.remove(member, "The inactive turn was passed to the next person.")

    def authenticate(self, token, connection_id=None):
        self.sweep()
        member = next(
            (m for m in self.members if secrets.compare_digest(m["token"], token or "")), None
        )
        if not member:
            raise AccessError("The session has ended. Join again.", 401)
        if connection_id is not None and member["connectionId"] != connection_id:
            raise AccessError("This page connection expired. Refresh and join again.", 409)
        return member

    def join(self, name, role, password, connection_id):
        self.sweep()
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 24:
            raise ValueError("Enter a name between 1 and 24 characters.")
        if role not in ("user", "admin"):
            raise ValueError("Select an access level.")
        admin_password = os.environ.get("DUO_ADMIN_PASSWORD")
        if role == "admin" and not admin_password:
            raise AccessError("Administrator access is disabled on this server.")
        if role == "admin" and not secrets.compare_digest(str(password), admin_password):
            raise AccessError("The administrator password is incorrect.")
        if not isinstance(connection_id, str) or not 8 <= len(connection_id) <= 80:
            raise ValueError("Invalid page connection.")
        now = self.clock()
        member = dict(
            id=uid(),
            token=secrets.token_urlsafe(32),
            name=name.strip(),
            role=role,
            joinedAt=now,
            connectionId=connection_id,
            seenAt=now,
            connected=False,
            disconnectedAt=now,
        )
        self.members.append(member)
        self.promote()
        self.changed()
        return dict(memberId=member["id"], token=member["token"])

    def resume(self, token, connection_id):
        if not isinstance(connection_id, str) or not 8 <= len(connection_id) <= 80:
            raise ValueError("Invalid page connection.")
        member = self.authenticate(token)
        member.update(
            connectionId=connection_id,
            connected=False,
            seenAt=self.clock(),
            disconnectedAt=self.clock(),
        )
        self.changed()
        return member

    def heartbeat(self, token, connection_id):
        try:
            member = self.authenticate(token, connection_id)
        except AccessError:
            return None
        member["seenAt"] = self.clock()
        return member

    def connected(self, token, connection_id):
        member = self.authenticate(token, connection_id)
        lease = uid()
        member.update(connected=True, seenAt=self.clock(), eventLease=lease)
        member.pop("disconnectedAt", None)
        self.changed()
        return lease

    def disconnect(self, token, connection_id, lease=None):
        member = next(
            (m for m in self.members if secrets.compare_digest(m["token"], token or "")), None
        )
        # A late pagehide / closed SSE from the old document cannot evict a reload.
        if (
            member
            and member["connectionId"] == connection_id
            and (lease is None or member.get("eventLease") == lease)
        ):
            member["connected"] = False
            member.setdefault("disconnectedAt", self.clock())
            self.changed()

    def require_operator(self, member, session_id, *, ready=True):
        if member["id"] != self.active_id or not self.turn or self.turn["acceptedAt"] is None:
            raise AccessError("This session is read-only until it is your turn.")
        if session_id != self.session_id:
            raise AccessError("This turn has ended. The operation was not applied.", 409)
        if ready and (
            self.state.handoff
            or self.state.reset
            or self.state.case_change
            or self.state.model_change
        ):
            raise AccessError("The session is being prepared. Please wait.", 409)
        if ready and (self.state.error or self.state.phase not in ("ready", "working")):
            raise AccessError(
                self.state.error or "The inference service is starting. Please wait.", 503
            )
        self.turn["lastActivityAt"] = self.clock()

    def accept(self, member, turn_id):
        if member["id"] != self.active_id or not self.turn or turn_id != self.turn["id"]:
            raise AccessError("It is not your turn yet.", 409)
        self.turn.update(acceptedAt=self.clock(), lastActivityAt=self.clock())
        self.changed()

    def touch(self, member, session_id):
        self.require_operator(member, session_id, ready=False)
        self.changed()

    def moderate(self, member, target):
        if member["role"] != "admin":
            raise AccessError("Only an administrator can remove users.")
        if target == member["id"]:
            raise ValueError("Use the leave button to exit your own session.")
        other = next((m for m in self.members if m["id"] == target), None)
        if other:
            result = dict(id=other["id"], name=other["name"])
            self.remove(other, f"Administrator {member['name']} removed you.", member["name"])
            return result
        return None

    def begin_capture(self, member, session_id):
        self.require_operator(member, session_id)
        if self.capture:
            raise ValueError("The previous recording is still being saved.")
        self.capture = dict(
            id=uid(), sessionId=self.session_id, ownerId=member["id"], startedAt=self.clock()
        )
        self.changed()
        return dict(self.capture)

    def check_capture(self, member, session_id, capture_id):
        self.require_operator(member, session_id, ready=False)
        if (
            not self.capture
            or self.capture["id"] != capture_id
            or self.capture["ownerId"] != member["id"]
        ):
            raise AccessError("This recording was cancelled.", 409)
        return self.capture

    def public(self):
        self.sweep()
        return dict(
            members=[
                {k: v for k, v in m.items() if k not in ("token", "eventLease")}
                for m in self.members
            ],
            activeId=self.active_id,
            turn=copy.deepcopy(self.turn),
            removals=copy.deepcopy(self.removals),
        )
