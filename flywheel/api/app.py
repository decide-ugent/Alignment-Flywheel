"""Flask app — single process serving 4 blueprints (oracle, proposer, flywheel, enforcement)."""

import threading
import time
from typing import Any, Dict

import requests
from flask import Flask, jsonify

from flywheel.api import state
from flywheel.api.blueprints.oracle_bp import bp as oracle_bp
from flywheel.api.blueprints.proposer_bp import bp as proposer_bp
from flywheel.api.blueprints.flywheel_bp import bp as flywheel_bp
from flywheel.api.blueprints.enforcement_bp import bp as enforcement_bp


def create_app(components: Dict[str, Any]) -> Flask:
    """Build a Flask app wired to the supplied component instances."""
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    state.set_components(
        oracle_=components.get("oracle") or components.get("oracle_adapter"),
        proposer_=components.get("proposer"),
        flywheel_=components.get("flywheel_overlay"),
        enforcement_=components.get("enforcement"),
    )

    app.register_blueprint(oracle_bp)
    app.register_blueprint(proposer_bp)
    app.register_blueprint(flywheel_bp)
    app.register_blueprint(enforcement_bp)

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


def start_api_in_thread(components: Dict[str, Any], port: int = 5000) -> threading.Thread:
    """Start the Flask app on a background daemon thread."""
    app = create_app(components)

    def _run():
        # Werkzeug dev server with multi-threading for concurrent requests
        app.run(host="127.0.0.1", port=port, threaded=True,
                use_reloader=False, debug=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Wait for /health
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            r = requests.get(f"{base}/health", timeout=0.5)
            if r.status_code == 200:
                return t
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"API failed to start on port {port}")
