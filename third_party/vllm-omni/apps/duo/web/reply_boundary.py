# SPDX-License-Identifier: Apache-2.0
# Reply-boundary helpers for chunked Intent output.
"""Stop a recording's reply when Intent returns to waiting for User input."""


class ReplyBoundary:
    # A chunk EOS only ends one 17-frame chunk. Allow a short pause inside an
    # answer; four unforced WAIT chunks (2.833 s) end this reply early.
    WAIT_CHUNKS = 4

    def __init__(self, input_chunks, tail_chunks):
        self.input_chunks = input_chunks
        self.stop_block = input_chunks + tail_chunks
        self.waiting = 0
        self.limit_reached = False

    def observe(self, local, action, gated):
        if local < self.input_chunks or gated or action != 'wait':
            self.waiting = 0
        else:
            self.waiting += 1
        settled = self.waiting >= self.WAIT_CHUNKS
        self.limit_reached = not settled and local + 1 >= self.stop_block
        return settled or self.limit_reached
