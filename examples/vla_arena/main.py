# pyright: reportMissingImports=false, reportMissingModuleSource=false

import collections
import dataclasses
import logging
import math
import os
import pathlib
import sys
import time
from collections.abc import Iterable

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

VLA_ARENA_DUMMY_ACTION = [0.0] * 6 + [-1.0]
VLA_ARENA_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    # Policy server
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # VLA-Arena setup
    vla_arena_root: str = "../VLA-Arena"
    task_suite_name: str = "safety_static_obstacles"
    task_level: int = 0
    num_steps_wait: int = 10
    num_trials_per_task: int = 10
    seed: int = 7

    # Robustness options
    add_noise: bool = False
    adjust_light: bool = False
    randomize_color: bool = False
    camera_offset: bool = False

    # Initial state selection
    init_state_selection_mode: str = "first"  # first | episode_idx
    init_state_offset: int = 0
    init_state_offset_random: bool = False

    # Effort (TA-VLA)
    use_effort: bool = True
    effort_dim: int = 7
    effort_history_len: int = 1

    # Logging / output
    save_video_mode: str = "first_success_failure"  # all | first_success_failure | none
    local_log_dir: str = "./experiments/logs"
    use_local_log: bool = True


def _ensure_vla_arena_importable(vla_arena_root: str) -> pathlib.Path:
    root = pathlib.Path(vla_arena_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"vla_arena_root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float32).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)

    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _get_effort_from_env(env, effort_dim: int) -> np.ndarray:
    # Keep the same source as replay_lerobot_dataset.py: arm controller torques.
    try:
        robot = env.robots[0]
        arm_name = robot.arms[0]
        controller = robot.part_controllers[arm_name]
        torques = np.asarray(controller.torques, dtype=np.float32).reshape(-1)
    except Exception:
        torques = np.zeros((effort_dim,), dtype=np.float32)

    if torques.shape[0] >= effort_dim:
        return torques[:effort_dim]

    out = np.zeros((effort_dim,), dtype=np.float32)
    out[: torques.shape[0]] = torques
    return out


def _get_env(task, resolution, add_noise, randomize_color, adjust_light, camera_offset):
    from vla_arena.vla_arena import get_vla_arena_path
    from vla_arena.vla_arena.envs import OffScreenRenderEnv

    task_bddl_file = os.path.join(
        get_vla_arena_path("bddl_files"),
        task.problem_folder,
        f"level_{task.level}",
        task.bddl_file,
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_offset": camera_offset,
        "color_randomize": randomize_color,
        "add_noise": add_noise,
        "light_adjustment": adjust_light,
    }
    return OffScreenRenderEnv(**env_args)


def _log(msg: str, log_file=None):
    logging.info(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def _save_video_if_needed(args: Args, rollout_images: list[np.ndarray], success: bool, episode_idx: int, task_name: str):
    if args.save_video_mode == "none":
        return
    if args.save_video_mode == "first_success_failure":
        # handled by caller
        pass

    date_str = time.strftime("%Y_%m_%d")
    time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
    out_dir = pathlib.Path("./rollouts/tavla_vla_arena") / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_task = task_name.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:60]
    out_path = out_dir / f"{time_str}--episode={episode_idx}--success={success}--task={safe_task}.mp4"
    imageio.mimwrite(out_path, [np.asarray(x) for x in rollout_images], fps=10)


def _resolve_task_suites(task_suite_name: str | Iterable[str]) -> list[str]:
    if isinstance(task_suite_name, str):
        return [task_suite_name]
    return list(task_suite_name)


def main(args: Args) -> None:
    np.random.seed(args.seed)
    _ensure_vla_arena_importable(args.vla_arena_root)

    from vla_arena.vla_arena import benchmark
    from vla_arena.vla_arena.utils.eval_init_state import select_init_state_index

    run_id = f"EVAL-{args.task_suite_name}-L{args.task_level}-{time.strftime('%Y%m%d_%H%M%S')}"
    log_file = None
    if args.use_local_log:
        pathlib.Path(args.local_log_dir).mkdir(parents=True, exist_ok=True)
        log_path = pathlib.Path(args.local_log_dir) / f"{run_id}.txt"
        log_file = open(log_path, "w")
        _log(f"Logging to: {log_path}", log_file)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    _log(f"Connected to websocket policy server: {args.host}:{args.port}", log_file)

    benchmark_dict = benchmark.get_benchmark_dict()
    suites = _resolve_task_suites(args.task_suite_name)

    grand_total_episodes = 0
    grand_total_successes = 0
    grand_total_costs = 0.0
    grand_success_costs = 0.0
    grand_failure_costs = 0.0
    grand_cost_episodes = 0

    for suite_name in suites:
        if suite_name not in benchmark_dict:
            raise ValueError(f"Unknown task suite: {suite_name}")

        task_suite = benchmark_dict[suite_name]()
        num_tasks = 10 if suite_name == "long_horizon" and args.task_level == 0 else 5
        _log(f"Evaluating suite={suite_name}, level={args.task_level}, tasks={num_tasks}", log_file)

        suite_total_episodes = 0
        suite_total_successes = 0

        for task_id in range(num_tasks):
            task = task_suite.get_task_by_level_id(args.task_level, task_id)
            task_description = task.language[0] if isinstance(task.language, list) else str(task.language)
            initial_states = task_suite.get_task_init_states(args.task_level, task_id)

            env = _get_env(
                task,
                resolution=VLA_ARENA_ENV_RESOLUTION,
                add_noise=args.add_noise,
                randomize_color=args.randomize_color,
                adjust_light=args.adjust_light,
                camera_offset=args.camera_offset,
            )

            rng = np.random.default_rng(args.seed)
            first_success_saved = False
            first_failure_saved = False

            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"{suite_name}-task{task_id}"):
                env.reset()
                init_idx = select_init_state_index(
                    num_initial_states=len(initial_states),
                    episode_idx=episode_idx,
                    selection_mode=args.init_state_selection_mode,
                    offset=args.init_state_offset,
                    offset_random=args.init_state_offset_random,
                    rng=rng if args.init_state_offset_random else None,
                )
                obs = env.set_init_state(initial_states[init_idx]) if init_idx is not None else env.get_observation()

                action_plan = collections.deque()
                effort_history = collections.deque(maxlen=max(1, args.effort_history_len))
                t = 0
                max_steps = 600 if suite_name == "long_horizon" and args.task_level >= 1 else 300
                done = False
                cost = 0.0
                rollout_images = []

                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, _, done, info = env.step(VLA_ARENA_DUMMY_ACTION)
                        if "cost" in info:
                            cost += float(info["cost"])
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    rollout_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ).astype(np.float32),
                            "prompt": task_description,
                        }

                        if args.use_effort:
                            current_effort = _get_effort_from_env(env, args.effort_dim)
                            effort_history.append(current_effort)
                            while len(effort_history) < max(1, args.effort_history_len):
                                effort_history.appendleft(np.zeros((args.effort_dim,), dtype=np.float32))
                            element["observation/effort"] = np.stack(list(effort_history), axis=0).astype(np.float32)

                        infer_result = client.infer(element)
                        action_chunk = infer_result["actions"]
                        if len(action_chunk) < args.replan_steps:
                            raise ValueError(
                                f"replan_steps={args.replan_steps}, but policy only returns {len(action_chunk)} actions"
                            )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, _, done, info = env.step(action.tolist())
                    if "cost" in info:
                        cost += float(info["cost"])

                    if done:
                        break
                    t += 1

                success = bool(done)

                suite_total_episodes += 1
                grand_total_episodes += 1
                grand_cost_episodes += 1
                suite_total_successes += int(success)
                grand_total_successes += int(success)
                grand_total_costs += cost
                if success:
                    grand_success_costs += cost
                else:
                    grand_failure_costs += cost

                _log(f"Task: {task_description}", log_file)
                _log(f"Episode success: {success}", log_file)
                _log(f"Episode cost: {cost}", log_file)
                _log(f"# episodes completed so far: {grand_total_episodes}", log_file)
                _log(f"# successes: {grand_total_successes}", log_file)

                if args.save_video_mode == "all":
                    _save_video_if_needed(args, rollout_images, success, episode_idx, task_description)
                elif args.save_video_mode == "first_success_failure":
                    if success and not first_success_saved:
                        _save_video_if_needed(args, rollout_images, success, episode_idx, task_description)
                        first_success_saved = True
                    if (not success) and (not first_failure_saved):
                        _save_video_if_needed(args, rollout_images, success, episode_idx, task_description)
                        first_failure_saved = True

            env.close()

        suite_sr = 100.0 * suite_total_successes / max(1, suite_total_episodes)
        _log(f"[{suite_name}] success rate: {suite_sr:.2f}%", log_file)

    total_sr = 100.0 * grand_total_successes / max(1, grand_total_episodes)
    avg_cost = grand_total_costs / max(1, grand_cost_episodes)
    _log(f"Total success rate: {total_sr:.2f}%", log_file)
    _log(f"Total episodes: {grand_total_episodes}", log_file)
    _log(f"Total successes: {grand_total_successes}", log_file)
    _log(f"Average cost: {avg_cost}", log_file)
    _log(f"Success costs: {grand_success_costs}", log_file)
    _log(f"Failure costs: {grand_failure_costs}", log_file)

    if log_file is not None:
        log_file.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
