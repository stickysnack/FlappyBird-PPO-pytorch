import argparse
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained PPO FlappyBird policy")
    parser.add_argument("--actor-path", type=str, default="trained_models/flappy_bird_actor_good",
                        help="Path to actor checkpoint (*.pth saved by training)")
    parser.add_argument("--fps", type=int, default=60,
                        help="Render FPS; set 0 or use --no-render for fastest headless play")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable pygame window (sets SDL_VIDEODRIVER=dummy if not already set)")
    parser.add_argument("--sample-action", action="store_true",
                        help="Sample from policy instead of greedy argmax")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Number of episodes to run; 0 means run forever")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser.parse_args()


def main():
    args = parse_args()

    # Headless mode needs SDL dummy driver before importing pygame/FlappyBird.
    if args.no_render and "SDL_VIDEODRIVER" not in os.environ:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    from train_ppo_good import PolicyNet
    from src.flappy_bird import FlappyBird

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    actor = PolicyNet().to(device)

    checkpoint = torch.load(args.actor_path, map_location=device)
    actor.load_state_dict(checkpoint["net"])
    actor.eval()

    env = FlappyBird("ppo_inference", device=device, fps=args.fps, render=not args.no_render)
    state, _, _ = env.step(0)

    episodes_finished = 0
    try:
        with torch.no_grad():
            while True:
                probs = actor(state)
                if args.sample_action:
                    action = torch.distributions.Categorical(probs).sample().item()
                else:
                    action = probs.argmax(dim=1).item()
                state, _, done = env.step(action)
                if done:
                    episodes_finished += 1
                    if args.episodes and episodes_finished >= args.episodes:
                        break
                    state, _, _ = env.step(0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
