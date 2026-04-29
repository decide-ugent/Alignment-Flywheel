"""Enforcement blueprint — /enforcement/* routes."""

from flask import Blueprint, jsonify, request

from flywheel.api import state
from flywheel.protocols.artifacts.unified_query_result import UnifiedQueryResult


bp = Blueprint("enforcement", __name__, url_prefix="/enforcement")


@bp.route("/decide", methods=["POST"])
def decide():
    unified = UnifiedQueryResult.from_dict(request.get_json() or {})
    result = state.enforcement.decide(unified)
    return jsonify(result.to_dict())
