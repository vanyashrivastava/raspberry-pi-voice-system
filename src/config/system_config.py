"""
Owner: Config Team
Responsibility: System-level configuration such as GPIO pin mapping and logging locations.

Contains example GPIO mapping for LED indicators and paths for logs and data. Owners (Siddhant)
should keep these values aligned with hardware wiring and system permissions.

Integration:
 - `alerts.visual_indicators.VisualIndicators` reads LED_PIN_MAP
 - `alerts.alert_logger.AlertLogger` may use LOG_PATH
"""

from dataclasses import dataclass


@dataclass
class SystemConfig:
    # LED pin mapping (BCM pin numbers) - adjust to wiring on the Pi
    LED_PIN_MAP = {
        'red': 17,
        'yellow': 27,
        'green': 22,
    }

    # Path for storing persistent logs (ensure the process user can write here)
    LOG_PATH: str = '/var/log/raspi_scam_alerts.log'

    # Web UI settings
    WEB_HOST: str = '0.0.0.0'
    WEB_PORT: int = 5000

    # Resource limits
    MAX_MEMORY_MB: int = 512


if __name__ == '__main__':
    print('SystemConfig example', SystemConfig())
