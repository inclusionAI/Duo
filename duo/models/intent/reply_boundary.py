"""Stop a recording's reply when Intent returns to waiting for User input."""


class ReplyBoundary:
    # A chunk EOS only ends one 17-frame chunk. Allow a short pause inside an
    # answer; four unforced WAIT chunks (2.833 s) end this reply. Speaking is
    # never limited by a fixed number of response chunks.
    WAIT_CHUNKS = 4

    def __init__(self, input_chunks):
        self.input_chunks = input_chunks
        self.waiting = 0
        self.complete = False

    def observe(self, local, action, gated):
        if local < self.input_chunks or gated or action != "wait":
            self.waiting = 0
        else:
            self.waiting += 1
        self.complete = self.waiting >= self.WAIT_CHUNKS
        return self.complete
