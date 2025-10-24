# Owner: Siddhant
# Responsibility: Log alerts to local storage and optionally forward to remote logging/dashboard.
# Goals:
# - Keep an immutable log of alert events with timestamps, probabilities, and metadata
# - Provide rotation and size limits for log files
# Integration points:
# - Called by inference engine when a suspicious event is detected
# - Web dashboard can read recent alerts from these logs
# Testing requirements:
# - Tests for serialization, rotation policy, and permission handling

import json
import os
import time
import typing as t


class AlertLogger:
    """
    Persist alert events locally and provide simple query by time.

    Methods:
        - log_alert(event_dict)
        - recent(n=100)

    TODOs:
        - Integrate with systemd or a remote syslog/HTTP endpoint
        - Add encryption or HMAC if logs are sensitive
    """

    def __init__(self, path: str = '/var/log/raspi_scam_alerts.log'):
        self.path = path
        # ensure directory exists
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                # fallback to current dir
                self.path = './raspi_scam_alerts.log'

    def log_alert(self, event: dict) -> None:
        entry = {'ts': time.time(), **event}
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def recent(self, n: int = 100) -> t.List[dict]:
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[-n:]
            return [json.loads(l) for l in lines]
        except FileNotFoundError:
            return []


if __name__ == '__main__':
    logger = AlertLogger()
    logger.log_alert({'type': 'test', 'scam_probability': 99.5, 'source': 'unit-test'})
    print('recent', logger.recent(5))
