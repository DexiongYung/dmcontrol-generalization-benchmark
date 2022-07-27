import torch
import os
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder


def evaluate(env, agent, video, num_episodes, L, step, args, test_env=False):
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            video.record(env)
            episode_reward += reward

        if L is not None:
            if test_env:
                _test_env = "_" + args.eval_mode
                _test_env = (
                    _test_env + "_" + str(args.distracting_cs_intensity)
                    if args.distracting_cs_intensity
                    else _test_env
                )
            else:
                _test_env = ""
            video.save(f"{step}{_test_env}.mp4")
            L.log(f"eval/episode_reward{_test_env}", episode_reward, step)
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)


def main(args):
    # Set seed
    utils.set_seed_everywhere(args.seed)

    # Initialize environments
    gym.logger.set_level(40)
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode=args.train_mode,
        intensity=args.train_distracting_cs_intensity,
    )
    test_env = (
        make_env(
            domain_name=args.domain_name,
            task_name=args.task_name,
            seed=args.seed + 42,
            episode_length=args.episode_length,
            action_repeat=args.action_repeat,
            image_size=args.image_size,
            mode=args.eval_mode,
            intensity=args.distracting_cs_intensity,
        )
        if args.eval_mode is not None
        else None
    )

    # Create working directory
    work_dir = os.path.join(
        args.log_dir,
        args.domain_name + "_" + args.task_name,
        args.algorithm,
        args.id,
        "seed_" + str(args.seed),
    )
    print("Working directory:", work_dir)
    args.__dict__["train_date"] = time.strftime("%m-%d-%y", time.gmtime())

    if not args.test_code_mode:
        assert not os.path.exists(
            os.path.join(work_dir, "train.log")
        ), "Specified working directory has existing train.log. Ending program."
    os.makedirs(work_dir, exist_ok=True)
    model_dir = utils.make_dir(os.path.join(work_dir, "model"))

    if args.save_video:
        video_dir = utils.make_dir(os.path.join(work_dir, "video"))
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

    utils.write_info(args, os.path.join(work_dir, "info.log"))
    utils.dump_args_json(args=args, log_dir=work_dir, model_dir=model_dir)

    # Prepare agent
    assert torch.cuda.is_available(), "must have cuda enabled"
    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.train_steps,
        batch_size=args.batch_size,
    )
    cropped_obs_shape = (
        3 * args.frame_stack,
        args.image_crop_size,
        args.image_crop_size,
    )
    print("Observations:", env.observation_space.shape)
    print("Cropped observations:", cropped_obs_shape)
    agent = make_agent(
        obs_shape=cropped_obs_shape, action_shape=env.action_space.shape, args=args
    )

    start_step, episode, episode_reward, done = 0, 0, 0, True
    L = Logger(work_dir)
    start_time = time.time()
    for step in range(start_step, args.train_steps + 1):
        if done:
            if step > start_step:
                L.log("train/duration", time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # Evaluate agent periodically
            if step % args.eval_freq == 0:
                num_episodes = (
                    args.eval_episodes_final_step
                    if step == args.train_steps
                    else args.eval_episodes
                )
                print("Evaluating:", work_dir)
                L.log("eval/episode", episode, step)
                evaluate(env, agent, video, num_episodes, L, step, args=args)
                if test_env is not None:
                    evaluate(
                        test_env,
                        agent,
                        video,
                        num_episodes,
                        L,
                        step,
                        test_env=True,
                        args=args,
                    )
                L.dump(step)

            # Save agent periodically
            if (
                step > start_step
                and step % args.save_freq == 0
                or step == args.train_steps - 1
            ):
                torch.save(agent, os.path.join(model_dir, f"{step}.pt"))

            L.log("train/episode_reward", episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log("train/episode", episode, step)

        # Sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # Run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        # Take step
        next_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        episode_reward += reward
        obs = next_obs

        episode_step += 1

    print("Completed training for", work_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
