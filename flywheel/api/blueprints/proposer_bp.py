"""Proposer blueprint — /proposer/* routes."""

from flask import Blueprint, jsonify, request

from flywheel.api import state
from flywheel.protocols.artifacts.context import Context


bp = Blueprint("proposer", __name__, url_prefix="/proposer")


@bp.route("/propose", methods=["POST"])
def propose():
    body = request.get_json() or {}
    context = Context.from_dict(body.get("context", {}))
    kwargs = body.get("kwargs", {})
    trajectory = state.proposer.propose(context, **kwargs)
    return jsonify(trajectory.to_dict())
