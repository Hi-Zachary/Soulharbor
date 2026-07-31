from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


def iter_json_array(path: str | Path, *, encoding: str = "utf-8", chunk_size: int = 1 << 20) -> Iterator[Any]:
    """
    Stream a top-level JSON array from disk without loading the whole file.

    Supports files like: [ {..}, {..}, ... ]

    This avoids adding an `ijson` dependency (useful for very large datasets).
    """
    p = Path(path)
    decoder = json.JSONDecoder()

    with p.open("r", encoding=encoding) as f:
        buf = ""
        eof = False

        def _read_more() -> None:
            nonlocal buf, eof
            if eof:
                return
            chunk = f.read(chunk_size)
            if chunk == "":
                eof = True
            else:
                buf += chunk

        _read_more()
        i = 0

        def _skip_ws() -> None:
            nonlocal i
            while True:
                while i < len(buf) and buf[i].isspace():
                    i += 1
                if i < len(buf) or eof:
                    return
                _read_more()

        _skip_ws()
        while i >= len(buf) and not eof:
            _read_more()
            _skip_ws()
        if i >= len(buf):
            raise ValueError(f"Empty JSON file: {p}")
        if buf[i] != "[":
            raise ValueError(f"Expected JSON array '[' at start: {p}")
        i += 1

        while True:
            _skip_ws()
            while i >= len(buf) and not eof:
                _read_more()
                _skip_ws()
            if i >= len(buf) and eof:
                raise ValueError(f"Unexpected EOF while reading JSON array: {p}")

            if buf[i] == "]":
                return

            # Parse next value
            while True:
                try:
                    obj, end = decoder.raw_decode(buf, i)
                    i = end
                    yield obj
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    _read_more()

            _skip_ws()
            while i >= len(buf) and not eof:
                _read_more()
                _skip_ws()
            if i >= len(buf) and eof:
                raise ValueError(f"Unexpected EOF after JSON value: {p}")

            if buf[i] == ",":
                i += 1
                continue
            if buf[i] == "]":
                return
            # Sometimes there's trailing whitespace; otherwise it's malformed.
            raise ValueError(f"Expected ',' or ']' after JSON value at pos={i} in {p}")


def peek_first(path: str | Path) -> Optional[Any]:
    for obj in iter_json_array(path):
        return obj
    return None

