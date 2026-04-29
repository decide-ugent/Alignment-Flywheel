"""Oracle blueprint — /oracle/* routes."""

from flask import Blueprint, jsonify, request

from flywheel.api import state
from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch
from flywheel.protocols.interfaces.base_spatial_oracle_adapter import BaseSpatialOracleAdapter


bp = Blueprint("oracle", __name__, url_prefix="/oracle")


@bp.route("/query", methods=["POST"])
def query():
    body = request.get_json() or {}
    if "points" in body:
        # spatial adapter style
        if not isinstance(state.oracle, BaseSpatialOracleAdapter):
            return jsonify({"error": "Oracle is not a spatial adapter"}), 400
        result = state.oracle.query_points(
            body["points"],
            include_uncertainty=body.get("include_uncertainty", True),
        )
        return jsonify(result)

    # full BaseOracle predict
    context = Context.from_dict(body.get("context", {}))
    trajectory = Trajectory.from_dict(body.get("trajectory", {}))
    out = state.oracle.predict(context, trajectory)
    return jsonify(out.to_dict())


@bp.route("/apply_batch", methods=["POST"])
def apply_batch():
    batch = GovernanceBatch.from_dict(request.get_json() or {})
    if isinstance(state.oracle, BaseSpatialOracleAdapter):
        return jsonify(state.oracle.send_patch(batch))
    applied = state.oracle.apply_batch(batch)
    return jsonify({"applied": applied, "oracle_version": state.oracle.get_version()})


@bp.route("/version", methods=["GET"])
def version():
    return jsonify({"version": state.oracle.get_version()})
