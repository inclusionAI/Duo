# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Low-latency H.264 delivery for browsers with WebCodecs."""

import base64
from fractions import Fraction

import av


class LiveH264Encoder:
    def __init__(self):
        self.jpeg = av.CodecContext.create("mjpeg", "r")
        self.jpeg.thread_count = 1
        self.codec = None
        self.avc_codec = None

    def encode(self, frames):
        packets = []
        for frame in frames:
            image = self.jpeg.decode(av.Packet(base64.b64decode(frame.pop("data"))))[0]
            # JPEG is full-range; browser H.264 uses the original limited-range output.
            image = image.reformat(format="yuv420p", src_color_range=2, dst_color_range=1)
            if self.codec is None:
                self.codec = av.CodecContext.create("libx264", "w")
                self.codec.width, self.codec.height = image.width, image.height
                self.codec.pix_fmt = "yuv420p"
                self.codec.time_base = Fraction(1, 24)
                self.codec.framerate = Fraction(24, 1)
                self.codec.thread_count = 2
                self.codec.options = {
                    "preset": "ultrafast",
                    "tune": "zerolatency",
                    "crf": "20",
                    "profile": "baseline",
                    "level": "3.1",
                    "x264-params": "keyint=48:scenecut=0:repeat-headers=1:bframes=0",
                }
            value = image
            value.pts = frame["pts"]
            value.time_base = Fraction(1, 24)
            emitted = self.codec.encode(value)
            assert len(emitted) == 1
            packet = emitted[0]
            raw = bytes(packet)
            if self.avc_codec is None:
                start = raw.index(b"\x00\x00\x00\x01\x67") + 5
                self.avc_codec = "avc1." + raw[start : start + 3].hex()
            frame["keyframe"] = packet.is_keyframe
            packets.append(raw)
        return packets
