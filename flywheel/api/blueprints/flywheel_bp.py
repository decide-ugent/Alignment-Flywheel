"""Flywheel blueprint — /flywheel/* routes."""

from flask import Blueprint, jsonify, request

from flywheel.api import state
from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch


bp = Blueprint("flywheel", __name__, url_prefix="/flywheel")


@bp.route("/overlay", methods=["POST"])
def overlay():
    body = request.get_json() or {}
    context = Context.from_dict(body.get("context", {}))
    trajectory = Trajectory.from_dict(body.get("trajectory", {}))
    out = state.flywheel.overlay(context, trajectory)
    return jsonify(out.to_dict())


@bp.route("/apply_batch", methods=["POST"])
def apply_batch():
    batch = GovernanceBatch.from_dict(request.get_json() or {})
    applied = state.flywheel.apply_batch(batch)
    return jsonify({"applied": applied, "governance_version": state.flywheel.get_version()})


@bp.route("/norms", methods=["GET"])
def norms():
    return jsonify({"norms": [n.to_dict() for n in state.flywheel.get_norms()]})


@bp.route("/version", methods=["GET"])
def version():
    return jsonify({"version": state.flywheel.get_version()})
