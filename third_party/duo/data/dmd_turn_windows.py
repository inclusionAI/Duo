"""CPU dialogue indexes, Assistant-ending windows and length-biased sampling."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
import math
import random

from .dataloader_sampler import ConversationSampler
from .dataloader_window import STAT_KEYS

POLICY = 'dmd_assistant_tail_window_v3'


def dialogue_runs(canonical):
    """Index aligned speech in annotation order, merging adjacent equal roles.

    Keep leading Assistant and trailing User speech. Windows need an Assistant
    endpoint, but no longer require an entire User/Assistant pair at their start.
    """
    runs, seen = [], set()
    for turn in canonical['fine_alignment']['turns']:
        identifier, role = turn['turn_id'], turn['role']
        if identifier in seen or role not in ('user', 'assistant'):
            raise ValueError('Invalid/duplicate canonical dialogue turn')
        seen.add(identifier)
        words = [w for w in turn['words'] if str(w['text']).strip()
                 and w['global_end_ms'] > w['global_start_ms']]
        if not words:
            raise ValueError(f'Natural turn {identifier} has no aligned words')
        lo = min(w['global_start_ms'] for w in words)
        hi = max(w['global_end_ms'] for w in words)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo < 0:
            raise ValueError('Invalid aligned natural turn time')
        if runs and runs[-1]['role'] == role:
            run = runs[-1]
            run['turn_ids'].append(identifier)
            run['start_ms'], run['end_ms'] = min(lo, run['start_ms']), max(hi, run['end_ms'])
        else:
            runs.append(dict(role=role, turn_ids=[identifier], start_ms=lo, end_ms=hi))
    return runs


def turn_windows(index, runs, text_rows, budget, max_new_tokens):
    """Allow every block start before an Assistant's natural endpoint.

    Prefix sums make media/text budget and supervision statistics independent
    of window duration. GT bodies in the generated tail are replaced by bounded
    NTP reservations; earlier Assistant speech in that tail keeps GT prefix room.
    """
    blocks = index['blocks']
    if (index.get('version') != 1 or not blocks or len(text_rows) != len(blocks)
            or budget < 1 or max_new_tokens < 1):
        raise ValueError('Invalid Assistant window index or budget')
    ends = [b['end_ms'] for b in blocks]
    keys = [k for k in STAT_KEYS if k != 'target_blocks']
    prefix = {key: [0] for key in keys}
    text_prefix = [0]
    for block, rows in zip(blocks, text_rows):
        for key in keys:
            prefix[key].append(prefix[key][-1] + block[key])
        text_prefix.append(text_prefix[-1] + rows)
    candidates, minimum = [], None
    for number, last in enumerate(runs):
        if last['role'] != 'assistant':
            continue
        first = bisect_right(ends, last['start_ms'])
        stop = bisect_left(ends, last['end_ms']) + 1
        if stop > len(blocks) or first >= stop:
            raise ValueError('Natural Assistant speech extends beyond feature media')
        history = runs[:number]
        # Earlier overlapping speech can extend into several generated blocks.
        gt_prefix = [0]
        for k, block in enumerate(blocks[:stop]):
            shared = any(r['role'] == 'assistant' and r['end_ms'] > block['start_ms']
                         and r['start_ms'] < block['end_ms'] for r in history)
            gt_prefix.append(gt_prefix[-1] + (text_rows[k] if shared else 0))
        # This is an annotation-order bound; materialization also clips token clocks.
        allowed = [t for r in runs[:number + 1] for t in r['turn_ids']]
        for start in range(stop):
            rollout_start = max(start, first)
            packed = index['shared_rows'] + prefix['packed_rows'][stop] - prefix['packed_rows'][start]
            old_rows = text_prefix[stop] - text_prefix[rollout_start]
            prefix_rows = gt_prefix[stop] - gt_prefix[rollout_start]
            reserved = packed - old_rows + (stop - rollout_start) * max_new_tokens + prefix_rows
            minimum = reserved if minimum is None else min(minimum, reserved)
            if math.ceil(reserved / 128) * 128 > budget:
                continue
            selected = {key: prefix[key][stop] - prefix[key][start] for key in keys}
            selected['packed_rows'] = packed
            selected.update(target_blocks=stop-start, start=start, stop=stop,
                start_ms=blocks[start]['start_ms'], end_ms=blocks[stop-1]['end_ms'],
                frame_start=blocks[start]['frame_start'], frame_stop=blocks[stop-1]['frame_stop'])
            candidates.append(dict(selected, budget_rows=reserved, rollout_start=rollout_start,
                prefix_reservation_rows=prefix_rows,
                assistant_start_ms=last['start_ms'], assistant_end_ms=last['end_ms'],
                turn_ids=allowed, last_assistant_turn_ids=last['turn_ids']))
    if not candidates:
        raise ValueError(f'No Assistant-ending window fits max_seq_len={budget}; '
                         f'minimum reserved rows={minimum}')
    return candidates


class DMDTurnSampler(ConversationSampler):
    POLICY = POLICY

    def __init__(self, *args, max_new_tokens=128, **kwargs):
        self.max_new_tokens = max_new_tokens
        super().__init__(*args, **kwargs)

    def candidate_windows(self, item):
        try:
            return turn_windows(item['window_index'], item['dialogue_runs'], item['block_text_rows'],
                                self.budget, self.max_new_tokens)
        except ValueError as error:
            raise ValueError(item['sample_id'] + ': ' + str(error)) from error

    def policy_identity(self):
        return dict(dmd_turn_policy=dict(max_new_tokens=self.max_new_tokens, length_power=2,
            sources=[dict(sample_id=r['sample_id'], source={k:v for k,v in r['rollout_source'].items() if k != 'path'},
                          runs=r['dialogue_runs'], text_rows=r['block_text_rows']) for r in self.items]))

    def choose_window(self, index, occurrence):
        buckets = defaultdict(list)
        for selected in self.candidates[index]:
            length = math.ceil(selected['budget_rows'] / 128)
            buckets[length].append(selected)
        lengths = sorted(buckets)
        rng = random.Random(f'window:{self._seed(index, occurrence)}')
        length = rng.choices(lengths, weights=[n * n for n in lengths], k=1)[0]
        return rng.choice(buckets[length])

    def packing_rows(self, index, selected):
        return selected['budget_rows']
