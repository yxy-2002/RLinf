# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Dexterous-hand intervention with independent input and target submission."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from multiprocessing.util import Finalize

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
    """Convert optional normalized action limits to native hand units.

    Bounds use the same physical-range normalization as hand reset/actions,
    independent of the task's permitted interval. Omitted bounds retain the
    physical limits.
    """
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
        if np.any((array < 0) | (array > 1)):
            raise ValueError(f"{name} must contain normalized values in [0, 1]")
        physical_low = np.asarray(physical_lower, dtype=float)
        physical_high = np.asarray(physical_upper, dtype=float)
        bounds[index] = physical_low + array * (physical_high - physical_low)
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


def _begin_hand_intervention(
    hand, current: np.ndarray, mode: str, policy_mode: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Keep relative hold targets continuous across a button press.

    Feedback may lag or differ under load. Replacing a held command with that
    feedback (including through a driver trajectory reset) introduces motion
    even when the glove delta is zero.
    """
    from rlinf_dexhand.wuji_spec import to_radians

    measured = np.asarray(hand.get_state(), dtype=np.float64).copy()
    if policy_mode:
        # The policy may have moved the hand since release.
        current = _feedback_target(hand)
        base = measured
    elif mode == "relative":
        base = (
            to_radians(hand.spec, current)
            if hand.spec.hand_type == "wuji1hand"
            else current.copy()
        )
    else:
        hand.clear_trajectory()
        base = measured
    return base, current


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
    policy_mode = config.get("release_behavior", "hold") == "policy"
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
        mouse_generation = 0
        glove_generation = 0
        input_generation = 0
        wait_release = True
        last_intervene = 0.0
        failure_enabled = config.get("right_button_ends_episode", False)
        failure_sequence = right_press_baseline = 0
        connection.send(("ready", os.getpid()))
        while True:
            started = time.monotonic()
            if connection.poll():
                command = connection.recv()
                if command == "stop":
                    break
                if command == "pause":
                    # Disarm command timeout and hold measured position before ACK.
                    if not policy_mode:
                        hand._call("set_teleop", False)
                    active = False
                elif command == "resume":
                    if not policy_mode:
                        hand._call("set_teleop", True)
                    _write_snapshot(shared, [0.0] * len(shared))
                    current = _feedback_target(hand)
                    baseline = base = None
                    previous_left = False
                    wait_release = True
                    # Ignore reset-time clicks and buttons still held on resume.
                    failure_sequence = 0
                    if failure_enabled:
                        right_press_baseline = mouse.get_button_press_counts()[0]
                    active = True
                else:
                    raise ValueError(f"Unknown teleop command: {command}")
                connection.send((command, None))
            if os.getppid() != config["parent_pid"]:
                raise RuntimeError("Collection process exited")
            if active:
                mouse_state = mouse.get_state()
                if failure_enabled:
                    # Counts survive reconnects. Retain a tap read just before
                    # unplugging even when its button level is already invalid.
                    right_presses = mouse_state.press_counts[0]
                    failure_sequence += right_presses - right_press_baseline
                    right_press_baseline = right_presses
                glove_state = glove.get_state()
                if not mouse_state.connected or not glove_state.connected:
                    # Keep lifecycle IPC responsive while the reader reconnects.
                    # Neither disconnected device may publish cached human input.
                    baseline = base = None
                    previous_left = False
                    wait_release = True
                    last_intervene = 0.0
                    if not policy_mode:
                        hand.command(current)
                    _write_snapshot(
                        shared,
                        [
                            time.monotonic(),
                            time.time(),
                            0,
                            0,
                            0,
                            0,
                            *np.zeros(6),
                            *current,
                            *([failure_sequence] if failure_enabled else []),
                            0,
                            input_generation,
                        ],
                    )
                    connection.poll(1 / config["frequency"])
                    continue
                if (
                    mouse_state.generation != mouse_generation
                    or glove_state.generation != glove_generation
                ):
                    current = _feedback_target(hand)
                    baseline = base = None
                    previous_left = False
                    wait_release = True
                    mouse_generation = mouse_state.generation
                    glove_generation = glove_state.generation
                    input_generation += 1
                sample = glove_state.target
                if sample.spec != hand.spec:
                    raise ValueError("Glove/driver hand specifications differ")
                arm, buttons = mouse_state.action, mouse_state.buttons
                left, right = bool(buttons[1]), bool(buttons[0])
                if not left:
                    wait_release = False
                controlling = left and not wait_release
                raw = np.asarray(sample.values, dtype=np.float64)
                if custom_limits:
                    raw = np.clip(raw, lower, upper)
                if controlling:
                    press_edge = not previous_left
                    if press_edge:
                        baseline = raw.copy()
                        base, current = _begin_hand_intervention(
                            hand, current, config["mode"], policy_mode
                        )
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
                if not policy_mode:
                    hand.command(current)
                _write_snapshot(
                    shared,
                    [
                        time.monotonic(),
                        time.time(),
                        sample.sequence,
                        left,
                        right,
                        controlling
                        if policy_mode
                        else time.monotonic() - last_intervene < config["timeout"],
                        *arm,
                        *current,
                        *([failure_sequence] if failure_enabled else []),
                        1,
                        input_generation,
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
        if hand is not None and active and not policy_mode:
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
        self.shared = context.Array(
            "d",
            SNAPSHOT_HEADER
            + config["hand_dim"]
            + int(config.get("right_button_ends_episode", False))
            + 2,  # Combined input health and recovery generation.
        )
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=worker,
            args=(child, self.shared, {**config, "parent_pid": os.getpid()}),
            name="dexhand-teleop",
            # The HID reader is a supervised child of this process. The loop
            # monitors parent_pid and exits if its collection owner disappears.
            daemon=False,
        )
        # Keep the public field name; either input device advances this epoch.
        self.mouse_generation = 0
        self.process.start()
        child.close()
        # Non-daemon children otherwise keep the interpreter waiting at exit.
        self._finalizer = Finalize(
            self, self._shutdown, args=(self.process, self.connection), exitpriority=10
        )
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
        """Wait for healthy input without submitting actions or advancing steps.

        Teleop heartbeats remain live during mouse/glove outages. Fatal reader
        and process errors still raise; fresh inputs resume collection automatically.
        """
        while True:
            if not self.process.is_alive():
                raise RuntimeError("Teleop process exited")
            if self.connection.poll():
                status, detail = self.connection.recv()
                raise RuntimeError(f"Teleop {status}: {detail}")
            data = _read_snapshot(self.shared)
            if time.monotonic() - data[0] > 0.5:
                raise RuntimeError("Teleop snapshot is stale")
            if data[-2]:
                self.mouse_generation = int(data[-1])
                return data[:-2]
            time.sleep(0.01)

    @staticmethod
    def _shutdown(process, connection) -> None:
        if process.is_alive():
            try:
                connection.send("stop")
            except (BrokenPipeError, OSError):
                pass
            process.join(3)
            if process.is_alive():
                process.terminate()
                process.join(3)
            if process.is_alive():
                process.kill()
                process.join()
        connection.close()

    def close(self) -> None:
        """Stop the child before its owner destroys the driver; safe to repeat."""
        self._finalizer()


class DexHandIntervention(gym.Wrapper):
    """Keep hand teleop independent of the arm/observation collection step."""

    def __init__(self, env, teleop_frequency=60, **kwargs):
        super().__init__(env)
        hand_type = env.unwrapped.config.end_effector_type
        if hand_type not in ("wuji_hand", "ruiyan_hand"):
            raise ValueError("DexHandIntervention requires Wuji or Ruiyan")
        self._release_behavior = kwargs.get("release_behavior", "hold")
        if self._release_behavior not in ("hold", "policy"):
            raise ValueError("release_behavior must be hold or policy")
        self._was_intervening = False
        self._policy_control_state = (False, False)
        self._lamp_prepared_input = None
        self._last_arm_intervene = float("-inf")
        self._right_button_ends_episode = kwargs.get("right_button_ends_episode", False)
        self._failure_sequence = 0
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
            "right_button_labels_only": kwargs.get("right_button_labels_only", False)
            or self._right_button_ends_episode,
            "right_button_ends_episode": self._right_button_ends_episode,
            "frequency": teleop_frequency,
            "glove_frequency": kwargs.get("glove_frequency", 60),
            "pipeline_config": kwargs["pipeline_config"],
            "scale_file": kwargs.get("scale_file"),
            "mode": kwargs.get("intervention_mode", "relative"),
            "timeout": kwargs.get("timeout", 0.5),
            "max_delta": env.unwrapped.config.hand_max_delta_per_step,
            "release_behavior": self._release_behavior,
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
        env.unwrapped._external_hand_control = self._release_behavior == "hold"

    def pause_hand_teleop(self) -> None:
        """Pause output before saving an episode or handing control to reset."""
        self._teleop.request("pause")

    def reset(self, **kwargs):
        """Transfer hand control to reset only after teleop acknowledges pause."""
        self._teleop.request("pause")
        self._was_intervening = False
        self._policy_control_state = (False, False)
        self._lamp_prepared_input = None
        self._last_arm_intervene = float("-inf")
        self._failure_sequence = 0
        result = self.env.reset(**kwargs)
        self._teleop.request("resume")
        deadline = time.monotonic() + 5
        while True:
            try:
                self._teleop.snapshot()
                self._mouse_generation = self._teleop.__dict__.get(
                    "mouse_generation", 0
                )
                break
            except RuntimeError:
                if not self._teleop.process.is_alive() or time.monotonic() > deadline:
                    raise
                time.sleep(0.01)
        return result

    def _policy_intervention_state(self, snapshot) -> tuple[bool, bool]:
        """Separate motion-triggered arm control from button-gated hand control."""
        now = time.monotonic()
        if np.linalg.norm(snapshot[6:12]) > 0.001:
            self._last_arm_intervene = now
        last_arm = self.__dict__.get("_last_arm_intervene", float("-inf"))
        # Match the native SpaceMouse wrapper's motion-release grace period.
        return now - last_arm < 0.5, bool(snapshot[5])

    def _mouse_reconnected(self) -> bool:
        """Detect an outage even if it recovered between collection reads."""
        generation = self._teleop.__dict__.get("mouse_generation", 0)
        return generation != self.__dict__.get("_mouse_generation", generation)

    def prepare_lamp_chunk_step(self, fresh_chunk: bool) -> bool:
        """Latch input for one step; reject a stale suffix on component release.

        The next step consumes this exact snapshot so the release check and
        action selection cannot observe different control states.
        """
        if self._release_behavior != "policy":
            raise ValueError("Continuous LAMP intervention requires policy release")
        self._lamp_prepared_input = None
        snapshot = self._teleop.snapshot()
        recovered = self._mouse_reconnected()
        if recovered and not fresh_chunk:
            return False
        control = self._policy_intervention_state(snapshot)
        previous = self.__dict__.get("_policy_control_state", (False, False))
        if not fresh_chunk and any(
            old and not new for old, new in zip(previous, control)
        ):
            return False
        if fresh_chunk:
            # The old suffix has already been discarded and replanned.
            self._policy_control_state = control
        self._lamp_prepared_input = (snapshot, control)
        return True

    def _failure_pending(self, snapshot: np.ndarray) -> bool:
        """Check whether the input process latched a new right-button press."""
        return self.__dict__.get("_right_button_ends_episode", False) and int(
            snapshot[-1]
        ) > self.__dict__.get("_failure_sequence", 0)

    def _apply_manual_failure(self, result: tuple, before, after) -> tuple:
        """Turn a right-button press into an ordinary failed terminal transition."""
        if not self.__dict__.get("_right_button_ends_episode", False):
            return result
        obs, reward, done, truncated, info = result
        failed = self._failure_pending(before) or self._failure_pending(after)
        self._failure_sequence = int(after[-1])
        info["right_click"] = failed
        info["manual_failure"] = failed
        if failed:
            from rlinf.utils.logging import get_logger

            reward, done, truncated = 0.0, True, False
            info.update(
                manual_failure_source="right_click", success=False, succeed=False
            )
            get_logger().info(
                "[manual_failure] SpaceMouse right button: episode failed, reward=0"
            )
        return obs, reward, done, truncated, info

    def step(self, action):
        """Use current arm input; hand output continues throughout image capture."""
        started = time.time()
        prepared = self.__dict__.get("_lamp_prepared_input")
        self._lamp_prepared_input = None
        before = self._teleop.snapshot() if prepared is None else prepared[0]
        recovered = self._mouse_reconnected()
        self._mouse_generation = self._teleop.__dict__.get("mouse_generation", 0)
        if recovered:
            self._last_arm_intervene = float("-inf")
        chosen = np.array(action, dtype=np.float64, copy=True)
        hand_end = SNAPSHOT_HEADER + len(chosen) - 6
        if getattr(self, "_release_behavior", "hold") == "policy":
            # The input process never publishes hand commands in this mode.
            # Both policy and human commands pass through the same env/limits.
            arm_control, hand_control = (
                self._policy_intervention_state(before)
                if prepared is None
                else prepared[1]
            )
            intervening = arm_control or hand_control
            if arm_control:
                chosen[:6] = before[6:12]
            if hand_control:
                chosen[6:] = before[12:hand_end]
            if recovered:
                # A plan computed before the outage is no longer current. Send
                # one stationary step and ask LAMP to replan from its observation.
                chosen[:6] = 0
                chosen[6:] = before[12:hand_end]

            obs, reward, done, truncated, info = self.env.step(chosen)
            after = self._teleop.snapshot()
            recovered = recovered or self._mouse_reconnected()
            info["left"], info["right"] = bool(after[3]), bool(after[4])
            current = (arm_control, hand_control)
            following = self._policy_intervention_state(after)
            previous = self.__dict__.get("_policy_control_state", (False, False))
            info["lamp_control_changed"] = (
                recovered or current != following or current != previous
            )
            info["lamp_control_released"] = recovered or any(
                (old and not active) or (active and not new)
                for old, active, new in zip(previous, current, following)
            )
            self._policy_control_state = following
            self._was_intervening = any(following)
            info["arm_intervening"] = arm_control
            info["hand_intervening"] = hand_control
            if "executed_action" not in info:
                raise RuntimeError(
                    "Policy hand control requires executed_action feedback"
                )
            if intervening:
                info["intervene_action"] = np.asarray(info["executed_action"]).copy()
            return self._apply_manual_failure(
                (obs, reward, done, truncated, info), before, after
            )
        if recovered:
            chosen[:6] = 0
        elif before[5]:
            chosen[:6] = before[6:12]
        chosen[6:] = before[12:hand_end]
        obs, reward, done, truncated, info = self.env.step(chosen)
        after = self._teleop.snapshot()
        info["left"], info["right"] = bool(after[3]), bool(after[4])
        # The hand target is asynchronous, so it is a snapshot, not one action
        # held for the entire collection step. Preserve both sample boundaries.
        info["teleop_snapshot"] = np.array(
            [started, time.time(), before[1], after[1], after[2], *after[12:hand_end]]
        )
        executed = np.asarray(info.get("executed_action", chosen)).copy()
        executed[6:] = after[12:hand_end]
        info["executed_action"] = executed
        if before[5]:
            info["intervene_action"] = info["executed_action"].copy()
        info["arm_intervening"] = bool(before[5])
        info["hand_intervening"] = bool(before[3])
        return self._apply_manual_failure(
            (obs, reward, done, truncated, info), before, after
        )

    def close(self):
        """Stop hand output before closing the controller and its driver."""
        try:
            self._teleop.close()
        finally:
            self.env.close()
