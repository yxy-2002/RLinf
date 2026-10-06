# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Dexterous-hand intervention with independent input and target submission."""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import gymnasium as gym
import numpy as np

# monotonic timestamp, wall timestamp, glove sequence, left/right, intervening,
# six SpaceMouse axes, a hand-specific number of normalized target joints.
SNAPSHOT_HEADER = 12


def _resolve_joint_limits(
    physical_lower: np.ndarray,
    physical_upper: np.ndarray,
    lower: list[float] | None,
    upper: list[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate optional limits in native hand units and retain physical limits."""
    bounds = [
        np.asarray(physical_lower, dtype=float),
        np.asarray(physical_upper, dtype=float),
    ]
    for index, (name, value) in enumerate(
        (("joint_lower_limits", lower), ("joint_upper_limits", upper))
    ):
        if value is None:
            continue
        array = np.asarray(value, dtype=float)
        if array.shape != bounds[index].shape or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain {bounds[index].size} finite values")
        bounds[index] = (
            np.maximum(bounds[index], array)
            if index == 0
            else np.minimum(bounds[index], array)
        )
    if np.any(bounds[0] > bounds[1]):
        raise ValueError(
            "Joint lower limits must not exceed upper limits within physical bounds"
        )
    return bounds[0], bounds[1]


def _bounded_hand_command(
    target: np.ndarray,
    current: np.ndarray,
    spec,
    lower: np.ndarray,
    upper: np.ndarray,
    max_delta: float,
) -> np.ndarray:
    """Bound the mapped target and rate-limited command in their respective units."""
    from rlinf_dexhand.wuji_spec import to_normalized

    target = np.clip(target, lower, upper)
    if spec.hand_type == "wuji1hand":
        target = to_normalized(spec, target)
        lower, upper = to_normalized(spec, lower), to_normalized(spec, upper)
    command = current + np.clip(target - current, -max_delta, max_delta)
    # Bounds take precedence when measured/reset state starts outside the region.
    return np.clip(command, lower, upper)


def _write_snapshot(shared, values):
    with shared.get_lock():
        shared[:] = values


def _read_snapshot(shared):
    # A terminated writer must not leave the collector blocked on the IPC lock.
    if not shared.get_lock().acquire(timeout=0.5):
        raise RuntimeError("Teleop snapshot lock unavailable")
    try:
        return np.asarray(shared[:], dtype=np.float64)
    finally:
        shared.get_lock().release()


def _feedback_target(hand) -> np.ndarray:
    """Project measured feedback into command limits, as WujiHand.reset does."""
    from rlinf_dexhand.wuji_spec import to_normalized

    from rlinf.utils.logging import get_logger

    measured = np.asarray(hand.get_state(), dtype=np.float64)
    if measured.shape != (hand.spec.action_dim,) or not np.isfinite(measured).all():
        raise ValueError("Invalid measured hand feedback")
    clipped = np.clip(measured, hand.spec.lower, hand.spec.upper)
    changed = np.flatnonzero(measured != clipped)
    if changed.size:
        get_logger().warning(
            "[WUJI_FEEDBACK_CLAMP] joints=%s measured_rad=%s target_rad=%s "
            "max_excess_rad=%.6f",
            changed.tolist(),
            measured[changed].tolist(),
            clipped[changed].tolist(),
            float(np.max(np.abs(measured - clipped))),
        )
    return (
        to_normalized(hand.spec, clipped)
        if hand.spec.hand_type == "wuji1hand"
        else clipped
    )


class RuiyanControllerClient:
    """Use the existing Ray controller API; the original serial owner is unchanged."""

    def __init__(self, config):
        import ray
        import ray.cloudpickle as cloudpickle
        from rlinf_dexhand.types import HandSpec

        self.ray = ray
        ray.init(
            address=config["ray_address"],
            namespace=config["ray_namespace"],
            logging_level="ERROR",
        )
        self.controller = cloudpickle.loads(config["controller_handle"])
        self.spec = HandSpec(
            "ruiyanhand",
            config["hand"].get("side", "left"),
            ("thumb_rotation", "thumb_bend", "index", "middle", "ring", "pinky"),
            "normalized",
            (0.0,) * 6,
            (1.0,) * 6,
        )

    def attach(self):
        self.get_state()

    def get_state(self):
        return self.ray.get(self.controller.get_hand_state.remote(), timeout=5)

    def command(self, action):
        target = self.spec.validate(action)
        self.ray.get(self.controller.command_end_effector.remote(target), timeout=5)

    def clear_trajectory(self):
        # Ruiyan has no spline buffer; its existing driver holds the last target.
        pass

    def _call(self, name, enabled):
        # Commands above are acknowledged RPCs. The lifecycle loop does not ACK
        # pause until the previous command completed, so reset cannot race it.
        if name != "set_teleop":
            raise ValueError(name)

    def hold(self):
        self.command(np.clip(self.get_state(), 0, 1))

    def shutdown(self):
        self.ray.shutdown()  # Disconnect this client; never destroy the actor.


def _teleop_worker(connection, shared, config):
    """Own input devices and submit targets without calling the collection env."""
    from rlinf_dexhand.glove import GloveExpert
    from rlinf_dexhand.wuji_spec import to_normalized

    from rlinf.envs.realworld.common.hand.wuji_hand import WujiHand
    from rlinf.envs.realworld.common.ros import ROSController
    from rlinf.envs.realworld.common.spacemouse.spacemouse_expert import (
        SpaceMouseExpert,
    )
    from rlinf.utils.logging import get_logger

    logger = get_logger()
    hand = glove = mouse = None
    active = False
    try:
        if config["hand_type"] == "wuji_hand":
            hand = WujiHand(ROSController(), **config["hand"])
        else:
            hand = RuiyanControllerClient(config)
        lower, upper = _resolve_joint_limits(
            hand.spec.lower,
            hand.spec.upper,
            config.get("joint_lower_limits"),
            config.get("joint_upper_limits"),
        )
        custom_limits = (
            config.get("joint_lower_limits") is not None
            or config.get("joint_upper_limits") is not None
        )
        hand.attach()
        mouse = SpaceMouseExpert()
        glove = GloveExpert(
            frequency=config["glove_frequency"],
            pipeline_config=config["pipeline_config"],
            scale_file=config["scale_file"],
        )
        glove.get_target()  # Startup errors are reported before readiness.
        current = _feedback_target(hand)
        baseline = base = None
        previous_left = False
        wait_release = True
        last_intervene = 0.0
        connection.send(("ready", os.getpid()))
        while True:
            started = time.monotonic()
            if connection.poll():
                command = connection.recv()
                if command == "stop":
                    break
                if command == "pause":
                    # Disarm command timeout and hold measured position before ACK.
                    hand._call("set_teleop", False)
                    active = False
                elif command == "resume":
                    hand._call("set_teleop", True)
                    _write_snapshot(shared, [0.0] * len(shared))
                    current = _feedback_target(hand)
                    baseline = base = None
                    previous_left = False
                    wait_release = True
                    active = True
                else:
                    raise ValueError(f"Unknown teleop command: {command}")
                connection.send((command, None))
            if os.getppid() != config["parent_pid"]:
                raise RuntimeError("Collection process exited")
            if active:
                sample = glove.get_target()
                if sample.spec != hand.spec:
                    raise ValueError("Glove/driver hand specifications differ")
                if not 0 <= time.time() - sample.timestamp <= 0.5:
                    raise RuntimeError("No fresh glove target for 0.5 seconds")
                arm, buttons = mouse.get_action()
                left, right = bool(buttons[1]), bool(buttons[0])
                if not left:
                    wait_release = False
                controlling = left and not wait_release
                raw = np.asarray(sample.values, dtype=np.float64)
                if custom_limits:
                    raw = np.clip(raw, lower, upper)
                if controlling:
                    if not previous_left:
                        baseline = raw.copy()
                        base = hand.get_state()
                        hand.clear_trajectory()
                    target = (
                        raw if config["mode"] == "absolute" else base + raw - baseline
                    )
                    if custom_limits:
                        current = _bounded_hand_command(
                            target,
                            current,
                            hand.spec,
                            lower,
                            upper,
                            config["max_delta"],
                        )
                    else:
                        target = np.clip(target, hand.spec.lower, hand.spec.upper)
                        if hand.spec.hand_type == "wuji1hand":
                            target = to_normalized(hand.spec, target)
                        limit = config["max_delta"]
                        current = current + np.clip(target - current, -limit, limit)
                previous_left = controlling
                if (
                    controlling
                    or np.linalg.norm(arm) > 0.001
                    or (right and not config["right_button_labels_only"])
                ):
                    last_intervene = time.monotonic()
                hand.command(current)
                _write_snapshot(
                    shared,
                    [
                        time.monotonic(),
                        time.time(),
                        sample.sequence,
                        left,
                        right,
                        time.monotonic() - last_intervene < config["timeout"],
                        *arm,
                        *current,
                    ],
                )
            # IPC wakes the wait immediately for pause/stop; no queued samples.
            connection.poll(
                max(0, 1 / config["frequency"] - (time.monotonic() - started))
            )
    except BaseException as exc:
        logger.exception("Dexterous-hand teleop process failed")
        try:
            connection.send(("error", repr(exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if hand is not None and active:
            try:
                hand.hold()
            except Exception:
                logger.exception("Could not hold hand during teleop shutdown")
        if hand is not None:
            hand.shutdown()  # Attached clients release only their own command lock.
        for device in (glove, mouse):
            if device is not None:
                device.close()
        connection.close()


class TeleopProcess:
    """One child process, acknowledged lifecycle commands and one latest snapshot."""

    def __init__(self, config: dict, worker=_teleop_worker):
        context = mp.get_context("spawn")
        self.shared = context.Array("d", SNAPSHOT_HEADER + config["hand_dim"])
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=worker,
            args=(child, self.shared, {**config, "parent_pid": os.getpid()}),
            name="dexhand-teleop",
            daemon=True,
        )
        self.process.start()
        child.close()
        try:
            self._response("ready", 30)
        except BaseException:
            self.close()
            raise

    def _response(self, expected: str, timeout: float = 15):
        if not self.connection.poll(timeout):
            raise TimeoutError(f"Teleop did not acknowledge {expected}")
        try:
            status, detail = self.connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(
                f"Teleop disconnected while waiting for {expected}"
            ) from exc
        if status != expected:
            raise RuntimeError(f"Teleop {status}: {detail}")

    def request(self, command: str) -> None:
        """Wait until the child completes a lifecycle transition."""
        if not self.process.is_alive():
            raise RuntimeError("Teleop process exited")
        try:
            self.connection.send(command)
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"Teleop disconnected before {command}") from exc
        self._response(command)

    def snapshot(self) -> np.ndarray:
        """Read current state without queuing work on the teleop process."""
        if not self.process.is_alive():
            raise RuntimeError("Teleop process exited")
        if self.connection.poll():
            status, detail = self.connection.recv()
            raise RuntimeError(f"Teleop {status}: {detail}")
        data = _read_snapshot(self.shared)
        if time.monotonic() - data[0] > 0.5:
            raise RuntimeError("Teleop snapshot is stale")
        return data

    def close(self) -> None:
        """Stop and join the child before its owner destroys the driver."""
        if self.process.is_alive():
            try:
                self.connection.send("stop")
            except (BrokenPipeError, OSError):
                pass
            self.process.join(3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
        self.connection.close()


class DexHandIntervention(gym.Wrapper):
    """Keep hand teleop independent of the arm/observation collection step."""

    def __init__(self, env, teleop_frequency=60, **kwargs):
        super().__init__(env)
        hand_type = env.unwrapped.config.end_effector_type
        if hand_type not in ("wuji_hand", "ruiyan_hand"):
            raise ValueError("DexHandIntervention requires Wuji or Ruiyan")
        if kwargs.get("release_behavior", "hold") != "hold":
            raise ValueError("Independent hand teleop requires release_behavior=hold")
        hand_dim = 20 if hand_type == "wuji_hand" else 6
        if env.action_space.shape != (6 + hand_dim,):
            raise ValueError("Hand action dimension does not match end effector")
        from rlinf_dexhand.pipeline import load_config

        pipeline = load_config(
            kwargs["pipeline_config"], scale_file=kwargs.get("scale_file")
        )
        expected = "wuji1hand" if hand_type == "wuji_hand" else "ruiyanhand"
        hand_config = dict(env.unwrapped.config.end_effector_config)
        if pipeline["hand"]["type"] != expected or (
            hand_type == "wuji_hand"
            and pipeline["hand"]["side"] != hand_config.get("side", "left")
        ):
            raise ValueError("Pipeline and robot hand specifications differ")
        if kwargs.get("intervention_mode", "relative") not in ("relative", "absolute"):
            raise ValueError("Invalid intervention mode")
        if not np.isfinite(teleop_frequency) or teleop_frequency <= 0:
            raise ValueError("teleop_frequency must be positive and finite")
        lower = kwargs.get("joint_lower_limits")
        upper = kwargs.get("joint_upper_limits")
        if lower is not None or upper is not None:
            if hand_type == "wuji_hand":
                from rlinf_dexhand.wuji_spec import wuji_spec

                spec = wuji_spec(pipeline["hand"]["side"])
                physical_lower, physical_upper = spec.lower, spec.upper
            else:
                physical_lower, physical_upper = np.zeros(hand_dim), np.ones(hand_dim)
            _resolve_joint_limits(physical_lower, physical_upper, lower, upper)
        process_config = {
            "joint_lower_limits": lower,
            "joint_upper_limits": upper,
            "hand": hand_config,
            "hand_type": hand_type,
            "hand_dim": hand_dim,
            "right_button_labels_only": kwargs.get("right_button_labels_only", False),
            "frequency": teleop_frequency,
            "glove_frequency": kwargs.get("glove_frequency", 60),
            "pipeline_config": kwargs["pipeline_config"],
            "scale_file": kwargs.get("scale_file"),
            "mode": kwargs.get("intervention_mode", "relative"),
            "timeout": kwargs.get("timeout", 0.5),
            "max_delta": env.unwrapped.config.hand_max_delta_per_step,
        }
        if hand_type == "ruiyan_hand":
            import ray
            import ray.cloudpickle as cloudpickle

            context = ray.get_runtime_context()
            controller = env.unwrapped._controller.worker_info_list
            if len(controller) != 1:
                raise ValueError("Expected one Franka controller for hand teleop")
            process_config.update(
                controller_handle=cloudpickle.dumps(controller[0].worker),
                ray_address=context.gcs_address,
                ray_namespace=context.namespace,
            )
        if hand_type == "ruiyan_hand":
            process_config["hand"]["side"] = pipeline["hand"]["side"]
        self._teleop = TeleopProcess(process_config)
        env.unwrapped._external_hand_control = True

    def pause_hand_teleop(self) -> None:
        """Pause output before saving an episode or handing control to reset."""
        self._teleop.request("pause")

    def reset(self, **kwargs):
        """Transfer hand control to reset only after teleop acknowledges pause."""
        self._teleop.request("pause")
        result = self.env.reset(**kwargs)
        self._teleop.request("resume")
        deadline = time.monotonic() + 5
        while True:
            try:
                self._teleop.snapshot()
                break
            except RuntimeError:
                if not self._teleop.process.is_alive() or time.monotonic() > deadline:
                    raise
                time.sleep(0.01)
        return result

    def step(self, action):
        """Use current arm input; hand output continues throughout image capture."""
        started = time.time()
        before = self._teleop.snapshot()
        chosen = np.array(action, dtype=np.float64, copy=True)
        if before[5]:
            chosen[:6] = before[6:12]
        chosen[6:] = before[12:]
        obs, reward, done, truncated, info = self.env.step(chosen)
        after = self._teleop.snapshot()
        info["left"], info["right"] = bool(after[3]), bool(after[4])
        # The hand target is asynchronous, so it is a snapshot, not one action
        # held for the entire collection step. Preserve both sample boundaries.
        info["teleop_snapshot"] = np.array(
            [started, time.time(), before[1], after[1], after[2], *after[12:]]
        )
        executed = np.asarray(info.get("executed_action", chosen)).copy()
        executed[6:] = after[12:]
        info["executed_action"] = executed
        if before[5]:
            info["intervene_action"] = info["executed_action"].copy()
        return obs, reward, done, truncated, info

    def close(self):
        """Stop hand output before closing the controller and its driver."""
        try:
            self._teleop.close()
        finally:
            self.env.close()
