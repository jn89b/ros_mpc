#!/usr/bin/env python3
import rclpy
import numpy as np
import time
import mavros  # noqa: F401 (kept in case you extend with MAVROS topics)
from mavros.base import SENSOR_QOS  # noqa: F401

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from ros_mpc.models.MathModel import PlaneKinematicModel
from ros_mpc.TrajNode import TrajNode, build_model
from optitraj.utils.data_container import MPCParams
from optitraj.close_loop import CloseLoopSim
from ros_mpc.SourceOptimalControl import SourceOptimalControl

# =================== Config ===================

# Target speed at waypoint
XF_SPEED = 20.0

# After wrapping to WP0, ignore reach checks briefly to avoid instant re-advance
POST_WRAP_GUARD_S = 1.0

# Waypoints-as-obstacles
WP_OBS_RADIUS = 30.0        # keep-out radius (m) for each waypoint obstacle
EXCLUDE_ACTIVE_WP = True    # exclude the active waypoint as an obstacle
EXCLUDE_NEIGHBORS = 1       # also exclude +/- this many neighbors (wrap-safe)

# =================== Obstacle model ===================

@dataclass
class Obstacle:
    center: Tuple[float, float]   # ENU (E, N)
    radius: float                 # meters

# =================== Helpers ===================

def _active_wp(traj_node: TrajNode) -> Optional[np.ndarray]:
    """Returns active waypoint ENU [E,N,U] or None."""
    if not hasattr(traj_node, "path_enu") or len(traj_node.path_enu) == 0:
        return None
    if traj_node.wp_idx >= len(traj_node.path_enu):
        return None
    E, N, U = traj_node.path_enu[traj_node.wp_idx]
    return np.array([E, N, U], dtype=float)

def _reached_wp(traj_node: TrajNode, xy_reach: float = 15.0, z_reach: float = 10.0) -> bool:
    """
    Reached if within cylinder: XY <= xy_reach AND |Z| <= z_reach,
    OR passed along-track relative to previous waypoint.
    """
    wp = _active_wp(traj_node)
    if wp is None or np.any(np.isnan(traj_node.enu_state[:3])):
        return False

    p = traj_node.enu_state[:3]
    d = p - wp
    near = (np.linalg.norm(d[:2]) <= xy_reach) and (abs(d[2]) <= z_reach)

    # along-track check
    if traj_node.wp_idx > 0:
        prev = np.array(traj_node.path_enu[traj_node.wp_idx - 1], dtype=float)
        seg = wp - prev
        passed = np.dot(p - wp, seg) > 0.0
    else:
        passed = False
    return near or passed

def _advance_wp_with_wrap(traj_node: TrajNode) -> None:
    """
    Advance to next waypoint; wrap to 0 if at end and looping is enabled
    (TrajNode already has loop_mission, lap_count, max_laps fields).
    """
    if len(traj_node.path_enu) == 0:
        return
    last_idx = len(traj_node.path_enu) - 1

    if traj_node.wp_idx < last_idx:
        traj_node.wp_idx += 1
        # traj_node.get_logger().info(f"[omni_traj] Advancing to waypoint idx={traj_node.wp_idx}")
        return

    # At last waypoint
    if getattr(traj_node, "loop_mission", True):
        traj_node.wp_idx = 0
        traj_node.lap_count = getattr(traj_node, "lap_count", 0) + 1
        # traj_node.get_logger().info(f"[omni_traj] Wrapped to first waypoint (lap {traj_node.lap_count}).")
        if getattr(traj_node, "max_laps", None) is not None and traj_node.lap_count >= traj_node.max_laps:
            traj_node.loop_mission = False
            # traj_node.get_logger().info("[omni_traj] Reached max_laps; disabling looping.")
    else:
        traj_node.get_logger().info("[omni_traj] At last waypoint; looping disabled. Holding last.")

def _wp_obstacles_from_path(traj_node: TrajNode, radius: float) -> List[Obstacle]:
    """Make a 2D circular obstacle (E,N) from every waypoint in the mission."""
    obs: List[Obstacle] = []
    for (E, N, _U) in getattr(traj_node, "path_enu", []):
        obs.append(Obstacle(center=(float(E), float(N)), radius=float(radius)))
    return obs

def _mask_exclusions_for_active(idx: int, length: int, exclude_neighbors: int) -> set:
    """
    Indices to exclude around the active index (wrap-safe for looping paths).
    """
    if length == 0 or idx < 0:
        return set()
    to_exclude = {idx}
    for k in range(1, exclude_neighbors + 1):
        to_exclude.add((idx - k) % length)
        to_exclude.add((idx + k) % length)
    return to_exclude

def _set_obstacles_on_solver(ocp: SourceOptimalControl, obstacles: List[Obstacle]) -> None:
    """
    Push obstacles to the solver. Supports either a set_obstacles() method
    or a public 'obs_params' field. Adjust here if your solver expects a different format.
    """
    # Serialize to a simple dict list (common for many custom solvers)
    obs_serialized = [{"cx": o.center[0], "cy": o.center[1], "r": o.radius} for o in obstacles]

    if hasattr(ocp, "set_obstacles") and callable(getattr(ocp, "set_obstacles")):
        ocp.set_obstacles(obs_serialized)
    elif hasattr(ocp, "obs_params"):
        ocp.obs_params = obs_serialized
    else:
        # If your solver needs a different injection method, add it here.
        pass

# =================== Main ===================

def main(args=None):
    rclpy.init(args=args)
    traj_node = TrajNode(node_name="omni_traj")
    rclpy.spin_once(traj_node)

    # ----- limits / model -----
    control_limits: Dict[str, Dict[str, float]] = {
        'u_phi':   {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
        'u_theta': {'min': -np.deg2rad(10), 'max': np.deg2rad(10)},
        'u_psi':   {'min': -np.deg2rad(30), 'max': np.deg2rad(30)},
        'v_cmd':   {'min': 18.0, 'max': 22.0},
    }
    state_limits: Dict[str, Dict[str, float]] = {
        'x': {'min': -np.inf, 'max': np.inf},
        'y': {'min': -np.inf, 'max': np.inf},
        'z': {'min': 30.0,    'max': 100.0},
        'phi':   {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
        'theta': {'min': -np.deg2rad(15), 'max': np.deg2rad(15)},
        'psi':   {'min': -np.pi,          'max': np.pi},
        'v':     {'min': 18.0,            'max': 22.0},
    }
    model: PlaneKinematicModel = build_model(control_limits, state_limits)

    # ----- MPC weights/params -----
    Q = np.diag([1.0, 1.0, 1.0, 0, 0, 0, 0])
    R = np.diag([0.2, 0.25, 0.25, 1.0])
    mpc_params = MPCParams(Q=Q, R=R, N=15, dt=0.1)

    # ----- solver + sim -----
    ocp = SourceOptimalControl(mpc_params=mpc_params, casadi_model=model, obs_params=[])
    xF = np.array([0, 0, 0, 0, 0, 0, XF_SPEED], dtype=float)
    u0 = np.array([0, 0, 0, 15], dtype=float)

    sim = CloseLoopSim(
        optimizer=ocp,
        x_init=traj_node.enu_state,
        x_final=xF,     # overwritten every tick
        print_every=1000,
        u0=u0,
        N=100
    )

    if np.all(np.isnan(traj_node.current_enu_controls)):
        traj_node.current_enu_controls = np.array([0, 0, 0, 15], dtype=float)

    last_wrap_time = 0.0

    # =================== main loop ===================
    while rclpy.ok():
        try:
            rclpy.spin_once(traj_node, timeout_sec=0.1)

            # Require valid state and a loaded path
            if np.any(np.isnan(traj_node.enu_state)):
                continue
            if len(getattr(traj_node, "path_enu", [])) == 0:
                continue

            # Advance if reached/passed (guard after wrap)
            if (time.time() - last_wrap_time) >= POST_WRAP_GUARD_S:
                bumps = 0
                while _reached_wp(traj_node) and bumps < 10:
                    before = traj_node.wp_idx
                    _advance_wp_with_wrap(traj_node)
                    if traj_node.wp_idx == 0 and before != 0:
                        last_wrap_time = time.time()
                    bumps += 1

            # Build goal from ACTIVE waypoint (ENU)
            awp = _active_wp(traj_node)
            if awp is not None:
                xF[:] = [awp[0], awp[1], awp[2], 0.0, 0.0, 0.0, XF_SPEED]

            # -------- Waypoints as obstacles (exclude active +/- neighbors) --------
            wp_obs: List[Obstacle] = _wp_obstacles_from_path(traj_node, radius=WP_OBS_RADIUS)
            if EXCLUDE_ACTIVE_WP and len(wp_obs) > 0:
                L = len(traj_node.path_enu)
                excl = _mask_exclusions_for_active(traj_node.wp_idx, L, EXCLUDE_NEIGHBORS)
                wp_obs = [o for j, o in enumerate(wp_obs) if j not in excl]

            # Push obstacle set to solver
            _set_obstacles_on_solver(ocp, wp_obs)

            # Solve one MPC step
            t0 = time.time()
            sim.x_init = traj_node.enu_state
            sol: Dict[str, Any] = sim.run_single_step(
                xF=xF,
                x0=traj_node.enu_state,
                u0=traj_node.current_enu_controls
            )
            dt = time.time() - t0

            # Logging
            dist_xy = np.linalg.norm(traj_node.enu_state[:2] - xF[:2])
            print(f"[omni_traj] wp_idx={traj_node.wp_idx} dist→goal: {dist_xy:.1f} m  "
                  f"obs={len(wp_obs)}  solve: {dt*1000:.1f} ms")

            # Publish (note: TrajNode.publish_traj expects z_goal THEN delta time)
            traj_node.publish_traj(sol, xF[2], dt, idx_buffer=1)

        except KeyboardInterrupt:
            print("Keyboard interrupt")
            break

    traj_node.destroy_node()
    rclpy.shutdown()
    return

if __name__ == "__main__":
    main()