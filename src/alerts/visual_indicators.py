# Owner: Siddhant
# Responsibility: Control GPIO LEDs on Raspberry Pi to indicate system state (OK, warning, alert).
# Goals:
# - Provide simple API to set LED colors or blink patterns
# - Abstract underlying GPIO library (RPi.GPIO or alternative for AI Hat)
# Integration points:
# - Called by orchestrator and alert logger when suspicious activity is detected
# Testing requirements:
# - Hardware-in-the-loop tests on a Raspberry Pi 5. Provide mock interface for unit tests.

import typing as t

# Dependencies: RPi.GPIO (or a mock for development on non-Pi systems)


class VisualIndicators:
    """
    Manage LED indicators through GPIO pins.

    API:
        - set_ok(): green steady
        - set_warning(): yellow blink
        - set_alert(): red blink
        - cleanup(): reset pins

    TODOs:
        - Wire default pins in config.system_config
        - Implement PWM brightness control if needed
    """

    def __init__(self, pin_map: t.Optional[dict] = None):
        self.pin_map = pin_map or {'red': 17, 'yellow': 27, 'green': 22}
        # TODO: initialize GPIO and set pin modes

    def set_ok(self) -> None:
        print('LED: OK (green)')

    def set_warning(self) -> None:
        print('LED: WARNING (yellow blink)')

    def set_alert(self) -> None:
        print('LED: ALERT (red blink)')

    def cleanup(self) -> None:
        print('GPIO cleanup')


if __name__ == '__main__':
    v = VisualIndicators()
    v.set_ok()
    v.set_warning()
    v.set_alert()
    v.cleanup()
