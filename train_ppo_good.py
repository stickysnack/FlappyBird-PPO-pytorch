
import argparse
import os
from torch.utils.data import   BatchSampler, SubsetRandomSampler
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.flappy_bird import FlappyBird

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of PPO to play Flappy Bird""")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num_iters", type=int, default=20000)
    parser.add_argument("--log_path", type=str, default="tensorboard_ppo")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--batch_size",type=int, default=2048 )
    parser.add_argument("--mini_batch_size",type=int, default=64 )
    parser.add_argument("--fps", type=int, default=60, help="Pygame render FPS; higher speeds up simulated steps")
    parser.add_argument("--no_render", action="store_true", help="Disable pygame window to run as fast as possible")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoints")
    parser.add_argument("--actor_path", type=str, default=None, help="Custom actor checkpoint path")
    parser.add_argument("--critic_path", type=str, default=None, help="Custom critic checkpoint path")

    args = parser.parse_args()
    return args


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        self.flat = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.Tanh())
        self.drop = nn.Dropout(0.5)
        self.fc3 = nn.Sequential(nn.Linear(512, 2))
    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.flat(output)
        output = self.drop(output)
        output = self.fc1(output)
        return nn.functional.softmax(self.fc3(output), dim=1)



class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512), nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, input):
        return self.net(input)

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage = 0.0
    advantages = []
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantages.append(advantage)
    advantages = np.array(advantages[::-1], dtype=np.float32)
    return torch.from_numpy(advantages)
def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(1993)
    else:
        torch.manual_seed(123)
    actor = PolicyNet().to(device)
    critic = ValueNet().to(device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=opt.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=opt.lr)

    # checkpoint loading (optional)
    actor_ckpt_path = opt.actor_path or os.path.join(opt.saved_path, "flappy_bird_actor_good")
    critic_ckpt_path = opt.critic_path or os.path.join(opt.saved_path, "flappy_bird_critic_good")
    start_iter = 0
    if opt.resume and os.path.exists(actor_ckpt_path):
        checkpoint = torch.load(actor_ckpt_path, map_location=device)
        actor.load_state_dict(checkpoint["net"])
        if "optimizer" in checkpoint:
            actor_optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = checkpoint.get("iter", start_iter)
        print(f"Loaded actor from {actor_ckpt_path}")
    if opt.resume and os.path.exists(critic_ckpt_path):
        checkpoint = torch.load(critic_ckpt_path, map_location=device)
        critic.load_state_dict(checkpoint["net"])
        if "optimizer" in checkpoint:
            critic_optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = max(start_iter, checkpoint.get("iter", start_iter))
        print(f"Loaded critic from {critic_ckpt_path}")
    if opt.resume and start_iter >= opt.num_iters:
        # if user sets num_iters equal to a finished checkpoint, extend so training continues
        opt.num_iters = start_iter + opt.num_iters
        print(f"num_iters increased to {opt.num_iters} (loaded checkpoint at iter {start_iter})")

    writer = SummaryWriter(opt.log_path)
    game_state = FlappyBird("ppo", device=device, fps=opt.fps, render=not opt.no_render)
    state, reward, terminal = game_state.step(0)
    max_reward = 0
    iter = start_iter
    print(f"Starting training from iter {iter} aiming for {opt.num_iters}")
    replay_memory = []
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []
    while iter < opt.num_iters:
        terminal = False
        episode_return = 0.0

        while not terminal:
            prediction = actor(state)
            action_dist = torch.distributions.Categorical(prediction)
            action_sample = action_dist.sample()
            action = action_sample.item()
            next_state, reward, terminal = game_state.step(action)
            replay_memory.append([state, action, reward, next_state, terminal])
            state = next_state
            episode_return += reward

            if len(replay_memory) > opt.batch_size:
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*replay_memory)
                states = torch.cat(state_batch, dim=0).to(device)
                actions = torch.tensor(action_batch, device=device).view(-1, 1)
                rewards = torch.tensor(reward_batch, device=device).view(-1, 1)
                dones = torch.tensor(terminal_batch, device=device).view(-1, 1).int()
                next_states = torch.cat(next_state_batch, dim=0).to(device)

                with torch.no_grad():
                    td_target = rewards + opt.gamma * critic(next_states) * (1 - dones)
                    td_delta = td_target - critic(states)
                    advantage = compute_advantage(opt.gamma, opt.lmbda, td_delta.cpu()).to(device)
                    old_log_probs = torch.log(actor(states).gather(1, actions)).detach()

                for _ in range(opt.epochs):
                    for index in BatchSampler(SubsetRandomSampler(range(opt.batch_size)), opt.mini_batch_size, False):
                        log_probs = torch.log(actor(states[index]).gather(1, actions[index]))
                        ratio = torch.exp(log_probs - old_log_probs[index])
                        surr1 = ratio * advantage[index]
                        surr2 = torch.clamp(ratio, 1 - opt.eps, 1 + opt.eps) * advantage[index]  # 截断
                        actor_loss = torch.mean(-torch.min(surr1, surr2))
                        critic_loss = torch.mean(
                            nn.functional.mse_loss(critic(states[index]), td_target[index].detach()))
                        actor_optimizer.zero_grad()
                        critic_optimizer.zero_grad()
                        actor_loss.backward()
                        critic_loss.backward()
                        actor_optimizer.step()
                        critic_optimizer.step()
                replay_memory = []

        if episode_return > max_reward:
            max_reward = episode_return
            print(" max_reward Iteration: {}/{}, Reward: {}".format(iter + 1, opt.num_iters, episode_return))

        iter += 1
        if (iter+1) % 10 == 0:
            evaluate_num += 1
            evaluate_rewards.append(episode_return)
            print("evaluate_num:{} \t episode_return:{} \t".format(evaluate_num, episode_return))
            writer.add_scalar('step_rewards', evaluate_rewards[-1], global_step= iter)
        if (iter+1) % 1000 == 0:
            actor_dict = {"net": actor.state_dict(), "optimizer": actor_optimizer.state_dict(), "iter": iter}
            critic_dict = {"net": critic.state_dict(), "optimizer": critic_optimizer.state_dict(), "iter": iter}
            torch.save(actor_dict, "{}/flappy_bird_actor_good".format(opt.saved_path))
            torch.save(critic_dict, "{}/flappy_bird_critic_good".format(opt.saved_path))



if __name__ == "__main__":
    opt = get_args()
    train(opt)
