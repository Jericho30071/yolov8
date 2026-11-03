"""Traffic light output drivers.

MockSignalDriver  — PC testing; keeps the colors in memory (shown on the dashboard).
GpioSignalDriver  — Raspberry Pi; switches the red/yellow/green relays through gpiozero.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class MockSignalDriver:
    def __init__(self, *_args, **_kwargs):
        self.state: dict[str, str] = {}

    def apply(self, lights: dict[str, str]) -> None:
        changed = {k: v for k, v in lights.items() if self.state.get(k) != v}
        if changed:
            log.info("signal %s", " ".join(f"{k}={v}" for k, v in sorted(lights.items())))
        self.state = dict(lights)

    def close(self) -> None:
        pass


class GpioSignalDriver(MockSignalDriver):
    """Drives one relay per lamp. Lanes without pins in the config are virtual."""

    def __init__(self, pins: dict[str, dict[str, int]], active_low: bool = True):
        super().__init__()
        from gpiozero import OutputDevice  # only available on the Raspberry Pi

        self.outputs = {
            lane: {color: OutputDevice(pin, active_high=not active_low, initial_value=False)
                   for color, pin in colors.items()}
            for lane, colors in pins.items()
        }

    def apply(self, lights: dict[str, str]) -> None:
        super().apply(lights)
        for lane, lamps in self.outputs.items():
            color = lights.get(lane, "red")
            for lamp_color, dev in lamps.items():
                dev.on() if lamp_color == color else dev.off()

    def close(self) -> None:
        # Leave every lamp red on shutdown; the pre-wired fixed-time fallback in the
        # relay hardware takes over if the Pi itself dies (Table 2, paper).
        for lamps in self.outputs.values():
            for lamp_color, dev in lamps.items():
                dev.on() if lamp_color == "red" else dev.off()


def make_driver(hw_cfg):
    if hw_cfg.driver == "gpio":
        return GpioSignalDriver(hw_cfg.pins, hw_cfg.relay_active_low)
    return MockSignalDriver()
