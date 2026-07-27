#!/usr/bin/env python3
"""Self-test for the 43-byte STM32 frame parser. No serial hardware needed."""

from __future__ import annotations

from frame_parser_v20 import STM32FrameParserV20, build_frame, FRAME_SIZE


def main() -> None:
    parser = STM32FrameParserV20()
    values1 = list(range(1, 21))
    values2 = [1000 + i for i in range(20)]

    # Mix noise + fragmented frames to verify stream resynchronization.
    stream = b"noise" + build_frame(values1)[:10]
    out = parser.feed(stream, timestamp_unix=1.0)
    assert len(out) == 0

    stream = build_frame(values1)[10:] + b"bad" + build_frame(values2)
    out = parser.feed(stream, timestamp_unix=2.0)
    assert len(out) == 2, f"expected 2 frames, got {len(out)}"
    assert out[0].values == tuple(values1)
    assert out[1].values == tuple(values2)

    suspicious = [5000] + list(range(1, 20))
    out_bad_value = parser.feed(build_frame(suspicious), timestamp_unix=3.0)
    assert len(out_bad_value) == 1
    assert out_bad_value[0].plausible is False
    assert parser.stats.implausible_frames == 1

    print("Parser self-test passed.")
    print(f"FRAME_SIZE={FRAME_SIZE}")
    print(f"stats={parser.stats}")
    print("first frame row sample:")
    print(out[0].as_csv_dict(start_timestamp=1.0))


if __name__ == "__main__":
    main()
