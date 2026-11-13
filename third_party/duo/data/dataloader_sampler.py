"""Deterministic occurrence windows, packing and committed optimizer cursors."""

from __future__ import annotations
import hashlib
import math
import random
import copy
import json
from .dataloader_window import STAT_KEYS, maximal_windows


class ConversationSampler:
    """Visit every conversation per pass; random windows precede greedy packing.

    A producer clone may prefetch, but only advance() commits consumption.
    Occurrence seeds and windows are independent of rank and worker placement.
    """

    POLICY = "conversation_random_window_v5"

    def __init__(self, items, world, accumulation, budget, seed=42, state=None):
        if len({s["sample_id"] for s in items}) != len(items):
            raise ValueError("Duplicate conversation identity")
        if not items or min(world, accumulation, budget) < 1:
            raise ValueError("Invalid packing dimensions")
        self.items, self.world, self.accumulation = items, world, accumulation
        self.budget, self.seed = budget, seed
        self.candidates = []
        for item in items:
            self.candidates.append(self.candidate_windows(item))
        self.identity = dict(
            policy=self.POLICY, world=world, accumulation=accumulation, budget=budget, seed=seed,
            items=[dict(sample_id=s["sample_id"], sha256=s.get("sha256", ""),
                        height=s.get("height"), width=s.get("width"),
                        stats={k: s[k] for k in STAT_KEYS},
                        window_index=s.get("window_index")) for s in items])
        self.identity.update(self.policy_identity())
        self.epoch = self.cursor = self.stream_position = self.occurrence_cursor = 0
        self.processed_blocks = 0
        self.processed_seconds = 0.0
        self.packs, self.windows = self._plan(0)
        self._pending = None
        if state is not None:
            if json.dumps(state.get("identity"), sort_keys=True) != json.dumps(self.identity, sort_keys=True):
                raise ValueError("Sampler identity differs; sampling/world size cannot resume")
            for key in ("epoch", "cursor", "stream_position", "occurrence_cursor"):
                if type(state.get(key)) is not int or state[key] < 0:
                    raise ValueError("Invalid sampler cursor: " + key)
            for epoch in range(state["epoch"]):
                packs, windows = self._plan(epoch)
                self.stream_position += len(packs)
                self._count([self._stats(i, windows[i]) for pack in packs for i in pack])
            self.epoch, self.cursor = state["epoch"], state["cursor"]
            self.packs, self.windows = self._plan(self.epoch)
            if (not 0 <= self.cursor < len(self.packs) or state.get("packs") != self.packs
                    or state.get("windows") != self.windows):
                raise ValueError("Saved packing/window order differs")
            used = [i for pack in self.packs[:self.cursor] for i in pack]
            self._count([self._stats(i, self.windows[i]) for i in used])
            self.stream_position += self.cursor
            self.occurrence_cursor = self.epoch * len(items) + len(used)
            if (state["stream_position"] != self.stream_position
                    or state["occurrence_cursor"] != self.occurrence_cursor
                    or self.stream_position % (world * accumulation)
                    or state.get("processed_blocks") != self.processed_blocks
                    or not math.isclose(state.get("processed_seconds", -1), self.processed_seconds,
                                        rel_tol=1e-10, abs_tol=1e-7)):
                raise ValueError("Sampler counters or optimizer boundary differ")
            # Preserve the committed floating-point accumulation order on resume.
            self.processed_seconds = state["processed_seconds"]

    def candidate_windows(self, item):
        if math.ceil(item['packed_rows'] / 128) * 128 <= self.budget:
            return [None]
        if 'window_index' not in item:
            raise ValueError('Oversized conversation requires a CPU block index: ' + item['sample_id'])
        return maximal_windows(item['window_index'], self.budget)

    def policy_identity(self):
        return {}

    def choose_window(self, index, occurrence):
        return random.Random(f'window:{self._seed(index, occurrence)}').choice(self.candidates[index])

    def packing_rows(self, index, selected):
        return self._stats(index, selected)['packed_rows']

    def _seed(self, index, occurrence):
        text = f"{self.POLICY}:{self.seed}:occurrence:{occurrence}:{self.items[index]['sample_id']}"
        return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)

    def _stats(self, index, selected):
        return self.items[index] if selected is None else selected

    def _count(self, stats):
        self.processed_blocks += sum(s["target_blocks"] for s in stats)
        self.processed_seconds += sum(s["supervised_seconds"] for s in stats)

    def _plan(self, epoch):
        order = list(range(len(self.items)))
        random.Random(f"{self.seed}:conversation_epoch:{epoch}").shuffle(order)
        packs, current, rows, windows = [], [], 0, [None] * len(order)
        for position, i in enumerate(order):
            occurrence = epoch * len(order) + position
            windows[i] = self.choose_window(i, occurrence)
            n = self.packing_rows(i, windows[i])
            if current and math.ceil((rows + n) / 128) * 128 > self.budget:
                packs.append(current)
                current, rows = [], 0
            current.append(i)
            rows += n
        if current:
            packs.append(current)
        return packs, windows

    def packs_for_epoch(self, epoch):
        return self._plan(epoch)[0]

    def _window(self):
        if self._pending is not None:
            return self._pending
        epoch, cursor, packs, windows = self.epoch, self.cursor, self.packs, self.windows
        occurrence = self.occurrence_cursor
        rows, indices, stats = [], [], []
        for offset in range(self.world * self.accumulation):
            selected = packs[cursor]
            ids = list(range(occurrence, occurrence + len(selected)))
            seeds = [self._seed(i, oid) for oid, i in zip(ids, selected)]
            rows.append(dict(
                indices=list(selected), windows=[windows[i] for i in selected], valid=True,
                epoch=epoch, pack_cursor=cursor, stream_position=self.stream_position+offset,
                occurrence_ids=ids, sample_seeds=seeds, noise_seed=seeds[0],
                microbatch=offset // self.world))
            indices.extend(selected)
            stats.extend(self._stats(i, windows[i]) for i in selected)
            occurrence += len(selected)
            cursor += 1
            if cursor == len(packs):
                epoch, cursor = epoch+1, 0
                packs, windows = self._plan(epoch)
        totals = {k: sum(s[k] for s in stats) for k in STAT_KEYS}
        totals.update(conversations=len(indices), unique_conversations=len(set(indices)), packs=len(rows),
                      dataset_passes=sorted({row["epoch"] for row in rows}))
        self._pending = rows, totals, (epoch, cursor, packs, windows, occurrence)
        return self._pending

    def batch(self, rank):
        if not 0 <= rank < self.world:
            raise ValueError("Invalid rank")
        rows, totals, _ = self._window()
        return copy.deepcopy(rows[rank::self.world]), copy.deepcopy(totals)

    def advance(self):
        _, totals, following = self._window()
        self._count([totals])
        self.epoch, self.cursor, self.packs, self.windows, self.occurrence_cursor = following
        self.stream_position += self.world * self.accumulation
        self._pending = None

    def state_dict(self):
        return copy.deepcopy(dict(identity=self.identity, epoch=self.epoch, cursor=self.cursor,
                                  packs=self.packs, windows=self.windows, stream_position=self.stream_position,
                                  occurrence_cursor=self.occurrence_cursor, processed_blocks=self.processed_blocks,
                                  processed_seconds=self.processed_seconds))

    def progress(self):
        used = [i for p in self.packs[:self.cursor] for i in p]
        return dict(
            epoch=self.epoch + len(used) / len(self.items), completed_epochs=self.epoch,
            epoch_conversations=len(used), conversations_per_epoch=len(self.items),
            processed_conversations=self.occurrence_cursor,
            epoch_block=sum(self._stats(i, self.windows[i])["target_blocks"] for i in used),
            blocks_per_epoch=sum(self._stats(i, self.windows[i])["target_blocks"] for i in range(len(self.items))),
            source_blocks_per_epoch=sum(s["target_blocks"] for s in self.items),
            processed_blocks=self.processed_blocks, processed_seconds=self.processed_seconds,
            pack_cursor=self.cursor, packs_per_epoch=len(self.packs), sampler_policy=self.POLICY,
            stream_position=self.stream_position, occurrence_cursor=self.occurrence_cursor,
            distributed_padding_packs=0)


class EvaluationSampler:
    """One ordered pass, one conversation per rank, preserving validation seeds."""

    def __init__(self, items, world, accumulation, budget, seed=42, state=None):
        if not items or world < 1 or len(items) % world or accumulation != 1:
            raise ValueError("Evaluation requires one microbatch and equal rank partitions")
        if budget < 1 or any(math.ceil(row["packed_rows"] / 128) * 128 > budget for row in items):
            raise ValueError("Evaluation budget cannot truncate a conversation")
        self.items, self.world, self.seed, self.cursor = items, world, seed, 0
        self.identity = dict(policy="conversation_validation_v1", world=world, seed=seed,
                             budget=budget, items=[(r["sample_id"], r["packed_rows"], r.get("sha256", "")) for r in items])
        if state is not None:
            if json.dumps(state.get("identity"), sort_keys=True) != json.dumps(self.identity, sort_keys=True):
                raise ValueError("Evaluation identity differs")
            self.cursor = state.get("cursor")
            if type(self.cursor) is not int or not 0 <= self.cursor <= len(items) // world:
                raise ValueError("Invalid evaluation cursor")

    def batch(self, rank):
        if not 0 <= rank < self.world:
            raise ValueError("Invalid rank")
        if self.cursor == len(self.items) // self.world:
            raise StopIteration
        index = self.cursor * self.world + rank
        seed = self.seed + 100000 + index
        row = dict(indices=[index], valid=True, sample_seeds=[seed], noise_seed=seed,
                   occurrence_ids=[index], microbatch=0)
        selected = self.items[self.cursor * self.world:(self.cursor + 1) * self.world]
        totals = {key: sum(item[key] for item in selected) for key in (
            "valid_video_weight", "valid_audio_weight", "target_blocks", "packed_rows", "supervised_seconds")}
        return [row], totals

    def advance(self):
        self.batch(0)
        self.cursor += 1

    def state_dict(self):
        return copy.deepcopy(dict(identity=self.identity, cursor=self.cursor))

    def progress(self):
        return dict(processed_conversations=self.cursor * self.world,
                    conversations_per_epoch=len(self.items))
