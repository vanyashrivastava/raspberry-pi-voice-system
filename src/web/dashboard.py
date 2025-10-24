# Owner: Nicole
# Responsibility: Provide a lightweight Flask dashboard showing recent alerts, system health, and quick controls.
# Goals:
# - Serve an index page with recent alerts and system status
# - Provide websocket or SSE endpoints for live updates (optional)
# Integration points:
# - Reads alerts from `alerts.alert_logger.AlertLogger`
# - Can display model confidence, recent call/email samples (sanitized), and allow mute/acknowledge actions
# Testing requirements:
# - Integration tests for HTTP endpoints and template rendering; security tests for XSS in displayed text

from flask import Flask, render_template, jsonify, request
import os
from ..alerts.alert_logger import AlertLogger


def create_app(alert_log_path: str = None) -> Flask:
    """Create and configure the Flask dashboard app.

    Args:
        alert_log_path: optional path to alerts log file
    Returns:
        Flask application object
    """
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'), static_folder=os.path.join(os.path.dirname(__file__), 'static'))
    logger = AlertLogger(path=alert_log_path) if alert_log_path else AlertLogger()

    @app.route('/')
    def index():
        alerts = logger.recent(50)
        return render_template('index.html', alerts=alerts)

    @app.route('/api/alerts')
    def api_alerts():
        return jsonify(logger.recent(200))

    @app.route('/api/ack', methods=['POST'])
    def ack():
        data = request.json or {}
        # TODO: implement acknowledge action (persist, set flag)
        return jsonify({'ok': True, 'received': data})

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
