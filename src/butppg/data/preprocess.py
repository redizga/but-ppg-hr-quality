"""Windowing & preprocessing (E1, Golikov).

10 s windows -> PPG 300 samples @ 30 Hz; decide ACC handling (keep 100 Hz vs
resample to 30 Hz) and record the choice in the config. ECG is used only to
derive the reference HR label, never as a model input. Writes processed arrays
into data/processed/ and the record registry (see registry.py). Implemented in E1.
"""

from __future__ import annotations


def main() -> None:  # pragma: no cover - E1
    raise NotImplementedError("E1 (Golikov): implement BUT PPG preprocessing + registry build")


if __name__ == "__main__":
    main()
