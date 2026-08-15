"""LaneDisciplineObserver — full-grid, velocity-aware, cell-classified.

Offline Red Team scanner that:
  1. Classifies every non-wall cell by corridor type (H_STRAIGHT,
     V_STRAIGHT, CORNER_xx, T_JUNCTION, etc.) using neighbor analysis.
  2. For each cell, for each relevant velocity direction, queries
     R(x, y, vx, vy) from the oracle.
  3. Computes the cell's lane position (0=left wall, 0.5=center,
     1=right wall) relative to the velocity direction and corridor
     geometry.
  4. Emits CandidateFlaws for:
     - Norm 1 (LANE_DISCIPLINE): lane_pos < 0.45 → left side patch
     - Norm 3 (CENTER_AVOIDANCE): 0.40 < lane_pos < 0.60 → soft center patch

The output is a set of LANE_DIRECTION_PATCH corrections — the distilled
artifact.  At inference the oracle evaluates these via vectorized
Gaussian kernels with no geometry reasoning.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from flywheel.protocols.ooda.observe_step import ObserveStep
from flywheel.protocols.artifacts.candidate_flaw import CandidateFlaw


# ── Cell classification ──────────────────────────────────────────

# Cell types
H_STRAIGHT = "H_STRAIGHT"      # horizontal corridor (open E/W, walls N/S)
V_STRAIGHT = "V_STRAIGHT"      # vertical corridor (open N/S, walls E/W)
CORNER_NE  = "CORNER_NE"       # corner: open North+East
CORNER_NW  = "CORNER_NW"       # corner: open North+West
CORNER_SE  = "CORNER_SE"       # corner: open South+East
CORNER_SW  = "CORNER_SW"       # corner: open South+West
T_JUNCTION = "T_JUNCTION"      # 3 open sides
CROSS      = "CROSS"           # 4 open sides
DEAD_END   = "DEAD_END"        # 1 open side
OPEN       = "OPEN"            # open area (2+ sides, doesn't fit above)


def _is_wall(layout_map, r, c):
    """Check if cell (r, c) is a wall (or out of bounds)."""
    rows, cols = len(layout_map), len(layout_map[0])
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return True
    v = layout_map[r][c]
    return v == 1


def classify_cells(layout_map):
    """Classify every non-wall cell by corridor type.

    Returns dict: (row, col) → cell_type string.
    """
    rows = len(layout_map)
    cols = len(layout_map[0])
    result = {}

    for r in range(rows):
        for c in range(cols):
            if _is_wall(layout_map, r, c):
                continue

            # Check 4 cardinal neighbors: N(r-1), S(r+1), W(c-1), E(c+1)
            n_open = not _is_wall(layout_map, r - 1, c)
            s_open = not _is_wall(layout_map, r + 1, c)
            w_open = not _is_wall(layout_map, r, c - 1)
            e_open = not _is_wall(layout_map, r, c + 1)

            open_dirs = []
            if n_open: open_dirs.append("N")
            if s_open: open_dirs.append("S")
            if w_open: open_dirs.append("W")
            if e_open: open_dirs.append("E")

            n_open_count = len(open_dirs)

            if n_open_count == 1:
                result[(r, c)] = DEAD_END
            elif n_open_count == 4:
                result[(r, c)] = CROSS
            elif n_open_count == 3:
                result[(r, c)] = T_JUNCTION
            elif n_open_count == 2:
                dirs = set(open_dirs)
                if dirs == {"N", "S"}:
                    result[(r, c)] = V_STRAIGHT
                elif dirs == {"E", "W"}:
                    result[(r, c)] = H_STRAIGHT
                elif dirs == {"N", "E"}:
                    result[(r, c)] = CORNER_NE
                elif dirs == {"N", "W"}:
                    result[(r, c)] = CORNER_NW
                elif dirs == {"S", "E"}:
                    result[(r, c)] = CORNER_SE
                elif dirs == {"S", "W"}:
                    result[(r, c)] = CORNER_SW
                else:
                    result[(r, c)] = OPEN
            else:
                result[(r, c)] = OPEN

    return result


def _tile_center(row, col, grid_size=8):
    """Tile (row, col) → XY center coordinates."""
    x = col - (grid_size - 1) / 2.0
    y = (grid_size - 1) / 2.0 - row
    return np.array([x, y], dtype=np.float64)


def _get_corridor_axes(cell_type):
    """Return list of corridor axis unit vectors for a cell type.

    Each axis is a direction in which the corridor runs.
    For a straight corridor, there's one axis (with traffic both ways).
    For corners/T/cross, there are multiple axes.
    """
    # East = (+1, 0), North = (0, +1) in XY coords
    E = np.array([1.0, 0.0])
    N = np.array([0.0, 1.0])

    if cell_type == H_STRAIGHT:
        return [E]           # road runs East-West
    elif cell_type == V_STRAIGHT:
        return [N]           # road runs North-South
    elif cell_type == CORNER_NE:
        return [N, E]
    elif cell_type == CORNER_NW:
        return [N, E]        # axes are still N and E
    elif cell_type == CORNER_SE:
        return [N, E]
    elif cell_type == CORNER_SW:
        return [N, E]
    elif cell_type in (T_JUNCTION, CROSS):
        return [E, N]        # both axes
    elif cell_type == DEAD_END:
        return [E, N]        # check both
    else:
        return [E, N]


def _relevant_velocities(cell_type, speed):
    """Return only the velocity directions relevant for this cell type.

    Straight corridors only need the two directions along the corridor.
    Corners and junctions need all directions for their open sides.
    This avoids generating spurious patches for perpendicular velocities
    that the agent would never actually travel in a given corridor.
    """
    E = np.array([speed, 0.0])
    W = np.array([-speed, 0.0])
    N = np.array([0.0, speed])
    S = np.array([0.0, -speed])

    if cell_type == H_STRAIGHT:
        return [E, W]
    elif cell_type == V_STRAIGHT:
        return [N, S]
    elif cell_type in (CORNER_NE, CORNER_NW, CORNER_SE, CORNER_SW):
        # Corners need all 4 directions: the forward pair (incoming/outgoing)
        # plus their reverses, to suppress wrong-direction traffic.
        return [E, W, N, S]
    elif cell_type == T_JUNCTION:
        return [E, W, N, S]     # all 4
    elif cell_type == CROSS:
        return [E, W, N, S]
    elif cell_type == DEAD_END:
        return [E, W, N, S]     # check all, pruned by reward threshold
    else:
        return [E, W, N, S]


# Inner corner offsets (from cell center) for each corner type.
# At a corner the two walls meet at this point.  The inner lane hugs
# this corner tightly; the outer lane swings wide.
_INNER_CORNER_OFFSETS = {
    CORNER_NE: np.array([-0.5, -0.5]),  # walls S & W → meet at SW
    CORNER_NW: np.array([ 0.5, -0.5]),  # walls S & E → meet at SE
    CORNER_SE: np.array([-0.5,  0.5]),  # walls N & W → meet at NW
    CORNER_SW: np.array([ 0.5,  0.5]),  # walls N & E → meet at NE
}

_CORNER_INNER_FRAC = 0.25   # inner lane = 25 % of physical corridor width


def _compute_lane_position(cell_center, point_xy, velocity, corridor_axis,
                           cell_width=1.0, cell_type=None):
    """Compute lane position for a point given velocity and corridor axis.

    Returns lane_pos in [0, 1]: 0=left wall, 0.5=center, 1=right wall,
    relative to the velocity's alignment with the corridor axis.

    For corner cells the mapping is non-linear: the inner lane (hugging
    the wall) occupies only 25 % of the corridor width while the outer
    lane takes 75 %.

    Two cases for the base (linear) computation:
    A) Velocity aligned with corridor axis (|dot| > 0.3):
       Cross-track offset perpendicular to velocity determines lane.
    B) Velocity perpendicular to corridor axis (|dot| <= 0.3):
       Position offset relative to the ROAD's direction determines lane.
       The road has a "canonical" travel direction; the agent should be
       on the right side of the road regardless of crossing velocity.
    """
    vel_norm = np.linalg.norm(velocity)
    if vel_norm < 1e-6:
        return 0.5  # stationary → center, no lane assignment

    vel_unit = velocity / vel_norm
    axis_dot = np.dot(vel_unit, corridor_axis)

    # Offset from cell center
    offset = point_xy - cell_center

    if abs(axis_dot) > 0.3:
        # Case A: velocity aligned with corridor
        # Travel direction = sign of dot × axis
        travel_dir = corridor_axis * np.sign(axis_dot)
        # Right = clockwise rotation of travel direction
        right_dir = np.array([travel_dir[1], -travel_dir[0]])
        # Cross-track position: project offset onto right_dir
        cross = np.dot(offset, right_dir)
        # Normalize: -0.5*width → 0.0, 0 → 0.5, +0.5*width → 1.0
        lane_pos = 0.5 + cross / cell_width
    else:
        # Case B: velocity perpendicular (crossing the road)
        # Use the road's canonical direction (positive axis)
        # "Right side of the road" = right of canonical direction
        right_dir = np.array([corridor_axis[1], -corridor_axis[0]])
        cross = np.dot(offset, right_dir)
        lane_pos = 0.5 + cross / cell_width

    lane_pos = float(np.clip(lane_pos, 0.0, 1.0))

    # ── Corner warping: inner lane = 25 %, outer lane = 75 % ─────
    # At a corner, the lane hugging the wall (inner) is narrow while
    # the lane swinging wide (outer) fills most of the corridor.
    if cell_type in _INNER_CORNER_OFFSETS:
        inner_off = _INNER_CORNER_OFFSETS[cell_type]
        # Cross product tells us whether the inner corner is LEFT or
        # RIGHT of the velocity direction.
        cross_sign = vel_unit[0] * inner_off[1] - vel_unit[1] * inner_off[0]
        f = _CORNER_INNER_FRAC  # 0.25

        if cross_sign > 0:
            # Inner corner is to the LEFT (low lane_pos)
            # Physical [0, f] → warped [0, 0.5]   (inner, compressed)
            # Physical [f, 1] → warped [0.5, 1.0] (outer, expanded)
            if lane_pos < f:
                lane_pos = lane_pos / f * 0.5
            else:
                lane_pos = 0.5 + (lane_pos - f) / (1.0 - f) * 0.5
        elif cross_sign < 0:
            # Inner corner is to the RIGHT (high lane_pos)
            # Physical [0, 1-f] → warped [0, 0.5]   (outer, expanded)
            # Physical [1-f, 1] → warped [0.5, 1.0] (inner, compressed)
            if lane_pos < (1.0 - f):
                lane_pos = lane_pos / (1.0 - f) * 0.5
            else:
                lane_pos = 0.5 + (lane_pos - (1.0 - f)) / f * 0.5

    return float(np.clip(lane_pos, 0.0, 1.0))


# ── Main observer class ─────────────────────────────────────────

class LaneDisciplineObserver(ObserveStep):
    """Full-grid Red Team observer for lane discipline.

    Scans every non-wall cell, classifies it, then for each relevant
    velocity direction queries R(s) and checks lane position.

    context must contain:
      - oracle:          BaseSpatialOracleAdapter
      - layout_map:      8×8 grid (1=wall, 0/other=free)
      - grid_size:       int (default 8)
      - reward_threshold: float (default 0.3)
      - lane_target:     float (default 0.75)
      - lane_dead_zone:  float (default 0.08)
      - center_zone:     tuple (default (0.40, 0.60))
      - query_speed:     float (default 3.0)
      - sub_resolution:  int (default 5) — sub-samples per cell
    """

    def observe(self, context: Dict[str, Any]) -> Dict[str, Any]:
        oracle = context["oracle"]
        layout_map = context["layout_map"]
        grid_size = context.get("grid_size", 8)
        threshold = context.get("reward_threshold", 0.3)
        lane_target = context.get("lane_target", 0.75)
        dead_zone = context.get("lane_dead_zone", 0.08)
        center_lo = context.get("center_zone", (0.40, 0.60))[0]
        center_hi = context.get("center_zone", (0.40, 0.60))[1]
        speed = context.get("query_speed", 3.0)
        sub_res = context.get("sub_resolution", 5)
        center_strength_scale = context.get("center_strength_scale", 0.4)
        corner_strength_scale = context.get("corner_strength_scale", 0.8)
        junction_strength_scale = context.get("junction_strength_scale", 0.5)

        # Step 1: classify every cell
        cell_types = classify_cells(layout_map)

        # Cell-type strength multiplier
        # Corners get strong patches (lane commitment happens there)
        # T-junctions / crosses / dead-ends get moderate reduction
        _CORNER_TYPES = {CORNER_NE, CORNER_NW, CORNER_SE, CORNER_SW}
        _OTHER_JUNCTION_TYPES = {T_JUNCTION, CROSS, DEAD_END}

        candidates: List[CandidateFlaw] = []
        cells_scanned = 0
        cells_skipped_wall = 0

        for (r, c), cell_type in cell_types.items():
            cell_center = _tile_center(r, c, grid_size)
            corridor_axes = _get_corridor_axes(cell_type)
            # Only check velocity directions relevant to this cell type
            vel_dirs = _relevant_velocities(cell_type, speed)
            if cell_type in _CORNER_TYPES:
                cell_str_scale = corner_strength_scale
                # Corners: tight cell radius
                cell_radius = 0.5
            elif cell_type in _OTHER_JUNCTION_TYPES:
                cell_str_scale = junction_strength_scale
                cell_radius = 0.5
            else:
                cell_str_scale = 1.0
                # Straight corridors: wide radius so patches span cell
                # boundaries along the corridor. The half-space gate
                # still prevents cross-corridor bleed.
                cell_radius = 2.5

            # Sub-sample within the cell for finer coverage.
            # Sample up to ±0.48 to cover near-edge regions.
            offsets = np.linspace(-0.48, 0.48, sub_res)

            for dx in offsets:
                for dy in offsets:
                    pt = cell_center + np.array([dx, dy])

                    for vel in vel_dirs:
                        vel_f = vel.astype(np.float32)

                        # Query R(x, y, vx, vy)
                        query = [float(pt[0]), float(pt[1]),
                                 float(vel_f[0]), float(vel_f[1])]
                        result = oracle.query_points([query])
                        reward = result["values"][0]

                        if reward < threshold:
                            continue

                        # Find the most relevant corridor axis for this velocity
                        vel_unit = vel / (np.linalg.norm(vel) + 1e-9)
                        best_axis = corridor_axes[0]
                        best_relevance = 0.0
                        for axis in corridor_axes:
                            relevance = abs(np.dot(vel_unit, axis))
                            # For perpendicular velocity, the road axis
                            # with highest perpendicular component matters
                            perp_relevance = 1.0 - relevance
                            total = max(relevance, perp_relevance * 0.8)
                            if total > best_relevance:
                                best_relevance = total
                                best_axis = axis

                        lane_pos = _compute_lane_position(
                            cell_center, pt, vel, best_axis,
                            cell_type=cell_type)

                        # Determine travel direction for patch
                        axis_dot = np.dot(vel_unit, best_axis)
                        if abs(axis_dot) > 0.3:
                            travel_dir = best_axis * np.sign(axis_dot)
                        else:
                            # Crossing — use road canonical direction
                            travel_dir = best_axis.copy()

                        # Lane normal: perpendicular to travel_dir,
                        # pointing toward the WRONG side.  Combined with
                        # cell_center, this creates a half-space gate that
                        # blocks suppression on the correct side of the
                        # corridor.
                        # Left of travel_dir = (-travel_dir[1], travel_dir[0])
                        # For keep-right (target>0.5): wrong side = left
                        # For keep-left  (target<0.5): wrong side = right
                        left_normal = np.array(
                            [-travel_dir[1], travel_dir[0]],
                            dtype=np.float32)
                        lane_normal = left_normal if lane_target > 0.5 else -left_normal

                        # --- Norm 1: LANE_DISCIPLINE (wrong side) ---
                        # If target > 0.5 (keep-right), wrong side = lane_pos < 0.5
                        # If target < 0.5 (keep-left),  wrong side = lane_pos > 0.5
                        on_wrong_side = ((lane_target > 0.5 and lane_pos < 0.50) or
                                         (lane_target < 0.5 and lane_pos > 0.50))
                        if on_wrong_side:
                            dist_from_target = abs(lane_pos - lane_target)
                            if dist_from_target >= dead_zone:
                                # Strength scales with distance from target,
                                # but always at least 0.90 to ensure strong
                                # suppression even near the corridor centre.
                                strength = max(0.90, min(1.0, dist_from_target / 0.5))
                                strength *= cell_str_scale
                                candidates.append(self._make_flaw(
                                    pt, vel_f, travel_dir, lane_pos,
                                    dist_from_target, strength, reward,
                                    "lane_discipline", cell_type, result,
                                    cell_center=cell_center,
                                    lane_normal=lane_normal,
                                    cell_radius=cell_radius))

                        # --- Norm 3: CENTER_AVOIDANCE ---
                        if center_lo < lane_pos < center_hi:
                            dist_from_target = abs(lane_pos - lane_target)
                            if dist_from_target >= dead_zone:
                                # Center avoidance: also needs strong suppression
                                strength = max(0.70, min(1.0, dist_from_target / 0.5))
                                strength *= center_strength_scale * cell_str_scale
                                candidates.append(self._make_flaw(
                                    pt, vel_f, travel_dir, lane_pos,
                                    dist_from_target, strength, reward,
                                    "center_avoidance", cell_type, result,
                                    cell_center=cell_center,
                                    lane_normal=lane_normal,
                                    cell_radius=cell_radius))

            cells_scanned += 1

        return {
            "candidates": candidates,
            "oracle_version": oracle.get_version(),
            "cells_scanned": cells_scanned,
            "cells_skipped_wall": cells_skipped_wall,
            "cell_types": cell_types,
            "lane_target": lane_target,
        }

    @staticmethod
    def _make_flaw(pt, vel, travel_dir, lane_pos, dist_from_target,
                   strength, reward, norm_name, cell_type, oracle_result,
                   cell_center=None, lane_normal=None, cell_radius=0.5):
        return CandidateFlaw(
            context={
                "point": pt.tolist(),
                "velocity": vel.tolist(),
                "travel_direction": travel_dir.tolist(),
                "lane_position": float(lane_pos),
                "dist_from_target": float(dist_from_target),
                "patch_strength": float(strength),
                "reward": float(reward),
                "norm": norm_name,
                "cell_type": cell_type,
                "cell_center": cell_center.tolist() if cell_center is not None else None,
                "lane_normal": lane_normal.tolist() if lane_normal is not None else None,
                "cell_radius": float(cell_radius),
            },
            trajectory={
                "kind": norm_name,
                "steps": [{"payload": {
                    "point": pt.tolist(),
                    "velocity": vel.tolist(),
                }}],
            },
            s=float(reward),
            u=0.2,
            u_thresh=0.5,
            v_O=oracle_result.get("oracle_version", "oracle:v0"),
        )
