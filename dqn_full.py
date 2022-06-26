import math, random, os, time, json, argparse
import numpy as np, pandas as pd
from typing import Counter
from collections import deque, Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38

# import environment
from sales_blending_env import SalesBlendingEnv

# parse arguments
parser = argparse.ArgumentParser()
# DQN hyperparams
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--buffer_size", default=20000, type=int)
parser.add_argument("--target_update_interval", default=400, type=int)
parser.add_argument("--double_q", default=True, action="store_true")
parser.add_argument("--gamma", default=0.999, type=float)
parser.add_argument("--epsilon_start", default=1.0, type=float)
parser.add_argument("--epsilon_final", default=0.1, type=float)
parser.add_argument("--epsilon_lead_in", default=0, type=int)
parser.add_argument("--epsilon_decay", default=20000, type=int)
parser.add_argument("--alpha_start", default=0.5, type=float)
parser.add_argument("--lr_decay", default=True, action="store_true")
# training and env params
parser.add_argument("--num_frames", default=100000, type=int)
parser.add_argument("--train_every", default=4, type=int)
parser.add_argument("--n_repeats", default=5, type=int)
parser.add_argument("--prefix", default="results/experimental2/test", type=str)
parser.add_argument("--do_evals", default=True, action="store_true")
parser.add_argument("--env_id", default="SalesBlendingEnv", type=str)

lod_in_state = False

Nonlin = nn.ReLU
fnonlin = F.relu

# adaptive gradient clipping
class AutoClip:
    def __init__(self):
        self.grad_history = []
        
    def compute_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm 

    def __call__(self, model, percentile):
        grad_norm = self.compute_grad_norm(model)
        self.grad_history.append(grad_norm)
        clip_value = np.percentile(self.grad_history, percentile)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value) 


# define class ReplayBuffer to store experiences
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) # quicker append and pop operations from BOTH the ends

    def push(self, state, action, lod, reward, next_state, done, expanded_act):
        ######## state = dict({'action_mask':[], 'real_obs':[]})
        ######## action = (lod action, final action)
        expand_state = {k: np.expand_dims(v, 0) for k,v in state.items()}
        expand_next_state = {k: np.expand_dims(v, 0) for k,v in next_state.items()}

        self.buffer.append((expand_state, action, reward, expand_next_state, done, expanded_act)) # add to the right

    def sample(self, batch_size):
        state, action, reward, next_state, done, expanded_act = zip(*random.sample(self.buffer, batch_size))
        keys = ['action_mask', 'real_obs']
        concate_state = dict(zip(keys, [np.concatenate([d[k] for d in state]) for k in keys]))
        concate_next_state = dict(zip(keys, [np.concatenate([d[k] for d in next_state]) for k in keys]))
        return concate_state, action, reward, concate_next_state, done, expanded_act

    def __len__(self):
        return len(self.buffer)


# prioritized buffer
class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.0):
        self.prob_alpha = prob_alpha 
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32) # priorities to select an experience

    def push(self, state, action, reward, next_state, done, expanded_act):
        expand_state = {k: np.expand_dims(v, 0) for k,v in state.items()}
        expand_next_state = {k: np.expand_dims(v, 0) for k,v in next_state.items()}

        max_prio = self.priorities.max() if self.buffer else 1.0 # set highest priority to new experience

        if len(self.buffer) < self.capacity:
            self.buffer.append((expand_state, action, reward, expand_next_state, done, expanded_act))
        else:
            self.buffer[self.pos] = (expand_state, action, reward, expand_next_state, done, expanded_act)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity



    def sample(self, batch_size, alpha=None):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        if alpha is None:
            probs  = prios ** self.prob_alpha
        else:
            probs  = prios ** alpha
        probs /= probs.sum() # normalized

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        batch       = list(zip(*samples))

        keys = ['action_mask', 'real_obs']
        concate_state = dict(zip(keys, [np.concatenate([d[k] for d in batch[0]]) for k in keys]))
        concate_next_state = dict(zip(keys, [np.concatenate([d[k] for d in batch[3]]) for k in keys]))
        
        actions     = batch[1]
        rewards     = batch[2]
        dones       = batch[4]
        expanded_acts = batch[5]

        return concate_state, actions, rewards, concate_next_state, dones, expanded_acts, indices


    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


USE_CUDA = False 

#############################################################################################
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.autoclip = AutoClip()
        self.input_size = env.observation_space['real_obs'].shape[0]
        self.output_size = env.max_avail_actions
        self.linear_model = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU())
        self.fc_out = nn.Linear(64, self.output_size)


    def forward(self, x, action_mask):
        h = self.linear_model(x)
        h = self.fc_out(h)
        mask = torch.where(action_mask==1, FLOAT_MIN, 0.0)
        return h + mask

    
    def act(self, state, epsilon): # state = {'action_mask': (max action,), 'real_obs': (num age,)}
        # exploitation: select action with highest value
        if random.random() >= epsilon:
            # without updating weights, calculate q values for all actions 
            with torch.no_grad():
                obs   = torch.tensor(state['real_obs'], dtype=torch.float).unsqueeze(0)
                m = torch.tensor(state['action_mask'], dtype=torch.float).unsqueeze(0)
                q_value = self.forward(obs, m)
            action  = q_value.max(1)[1].item()  # ALREADY CONSIDER MASK IN FORWARD FUNCTION

        else: # exploration: take random valid action
            mask = state['action_mask']        
            assert len(mask) == env.max_avail_actions

            action = random.choice( \
                list(filter(lambda a: a[1] == 0, enumerate(mask)))
                )[0]  # randomly select VALID action 
        return action



def compute_td_loss(batch_size,
                    gamma,
                    num_frames, # for lr decay
                    alpha,
                    double_q=True,
                    lr_decay=False
                    ):
    # sampling from replay buffer
    state, actions, reward, next_state, done, expanded_actions, indices = replay_buffer.sample(batch_size, alpha)

    state_obs       = torch.tensor(state['real_obs'], dtype=torch.float)
    state_mask      = torch.tensor(state['action_mask'], dtype=torch.float)
    next_state_obs  = torch.tensor(next_state['real_obs'], dtype=torch.float)
    next_state_mask = torch.tensor(next_state['action_mask'], dtype=torch.float)
    
    actions     = torch.tensor(actions, dtype=torch.long)
    reward     = torch.tensor(reward, dtype=torch.float)
    done       = torch.tensor(done, dtype=torch.float)
    expanded_actions = torch.tensor(expanded_actions, dtype=torch.long)

    # Q-network/ online network
    q_values = model(state_obs, state_mask)
    q_values_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # target network (evaluation)
    with torch.no_grad():
        next_q_values = target_model(next_state_obs, next_state_mask)

    # double q learning: 
    if double_q:
        # use online model to select action
        next_q_values_current_model = model(next_state_obs, next_state_mask)  
        # select action with highest q value based on values from online model
        next_max_actions = next_q_values_current_model.max(1)[1] # ALREADY CONSIDER MASK IN FORWARD FUNCTION
        # use target model to evaluate 
        next_q_values_max = next_q_values.gather(1, next_max_actions.unsqueeze(1)).squeeze(1)
    else: # target model
        next_q_values_max = next_q_values.max(1)[0]

    targets = reward + gamma * next_q_values_max * (1 - done)

    # calculate TD loss 
    loss = torch.tensor(0.0, requires_grad=True)
    prios = torch.ones(batch_size, requires_grad=False)*1e-5

    loss = F.smooth_l1_loss(q_values_taken, targets)
    prios = prios + torch.abs(q_values_taken - targets)

    # update priorities of replay buffer
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())

    # learning rate decay
    if lr_decay:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*(1 - 1/(num_frames))

    optimizer.zero_grad()
    loss.backward()
    model.autoclip(model, 10) # gradient clipping
    optimizer.step()

    return loss, q_values_taken.mean()


# update target network using weights from online network
def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def eval(n_episodes=10):
    eval_rewards = []
    eval_episode_reward = 0
    state = eval_env.reset()
    for _ in range(n_episodes):
        done = False
        while not done: 
            action = model.act(state, 0)
            next_state, reward, done, expanded_act = eval_env.step(action)
            state = next_state
            eval_episode_reward += reward

            if done:
                state = eval_env.reset()
                eval_rewards.append(eval_episode_reward)
                eval_episode_reward = 0
    return(eval_rewards)


def train(
    model,
    target_model,
    env,
    eval_env,
    num_frames,
    batch_size,
    target_update_interval,
    gamma,
    epsilon_start,
    epsilon_final,
    epsilon_decay,
    prefix,
    do_evals,
    train_every,
    double_q,
    lr_decay
):
    run_data = {
        "loss": [],
        "value": [],
        "returns": [],
    }

    sales_actions_taken = []
    full_actions_taken = []
    episode_reward = 0

    state = env.reset()
    frame_idx = 1
    env_running_time = 0
    start_time = time.time()

    while frame_idx <= num_frames:
        # get parameters
        alpha = alpha_by_frame(frame_idx)
        epsilon = epsilon_by_frame(frame_idx)
        action = model.act(state, epsilon)

        # obtain experience
        next_state, reward, done, expanded_act = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done, expanded_act)
        sales_actions_taken.append(action)
        full_actions_taken.append(expanded_act)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            run_data["returns"].append((frame_idx, episode_reward))
            episode_reward = 0

        # compute TD error
        if len(replay_buffer) > 1000 and frame_idx % train_every == 0:
            loss, value = compute_td_loss(batch_size,
                                            gamma,
                                            num_frames,
                                            alpha,
                                            double_q=double_q,
                                            lr_decay=lr_decay)
            # save training results
            run_data["loss"].append((frame_idx, loss.item()))
            run_data["value"].append((frame_idx, value.item()))

        # update target network
        if frame_idx % target_update_interval == 0:
            update_target(model, target_model)

        # report logging data
        if frame_idx % 1000 == 0:
            print("{:8} frames | R {:.2f} (max {:.2f}) |  eps {:.2f} | Q {:.2f} | loss {:.6f} ~ {:4f}% | sales act {:.2f} : {:.2f} | {} : {:.2f}".format(
                frame_idx,                                          # current frame indx
                np.mean([x[1] for x in run_data["returns"][-10:]]), # smoothed mean return of last 10 episodes
                np.max([x[1] for x in run_data["returns"]]),        # current max episode return
                epsilon,                                            # epsilon greedy exploration
                np.mean([x[1] for x in run_data["value"][-10:]]),   # smoothed mean value of last 10 episodes
                np.mean([x[1] for x in run_data["loss"][-10:]]),    # smoothed mean loss of last 10 episodes
                100*np.mean([x[1] for x in run_data["loss"][-10:]]) / np.mean([x[1] for x in run_data["value"][-10:]]), # ratio of loss/value
                Counter(sales_actions_taken).most_common(1)[0][0], # most common sales action
                Counter(sales_actions_taken).most_common(1)[0][1], # most common sales action (# of times)
                env.sales_blend_actions_counter[Counter(full_actions_taken).most_common(1)[0][0]], # most common action
                Counter(full_actions_taken).most_common(1)[0][1] # most common action (# of times)
                )) 
            sales_actions_taken.clear()
            full_actions_taken.clear()        
        
        frame_idx +=1

    finish_time = time.time()
    print('Finish training model after', finish_time-start_time)
    print('Env running time {} | Model running time {}'.format(
        env_running_time,
        finish_time-start_time-env_running_time
    ))

    # evaluate at the end of training
    if do_evals:
        eval_reward = eval(n_episodes=10)
        print('Evaluation: Mean {:6f} | Max {:6f} | Min {:6f} | Std {:6f}'.format(
            np.mean(eval_reward),
            np.max(eval_reward),
            np.min(eval_reward),
            np.std(eval_reward)
        ))
    
    return run_data


if __name__ == "__main__":
    args = parser.parse_args([])

    # helper functions
    def epsilon_by_frame(frame_idx):
        return args.epsilon_final + max(0, (args.epsilon_start - args.epsilon_final) * min(1, (1 - (frame_idx - args.epsilon_lead_in) / args.epsilon_decay)))

    def alpha_by_frame(frame_idx):
        return 0 + max(0, (args.alpha_start - 0) * min(1, (1 - frame_idx / args.num_frames)))

    #########################################
    # BASE CASE: environment params
    time_limit = 1000
    num_prods = 2           #int, #number of target products sold
    num_ages = 7            #int, #number of ages in inventory
    num_inv_levels = 7      #int, #inventory level in discrete state case
    target_ages = [3,5]     #List[int], #ages of target products sold                
    det_demands = [3,3]     #List[int], #deterministic demand for each product
    holding_cost = 24       #float, #holding cost per year
    mkt_start = 24; mkt_step = 24; mkt_max_age = 3  #float, float, int # supermarket contributions
    brd_start = None; brd_exp = None; brd_contrib = [500, 600] # contributions of branded products
    q_weibull= 0.875; beta_weibull= 0.8; max_decay=0.3 #float, #params of decay distribution


    action_type = 'full'
    purchase_rule = 'upto' # upto or fixed
    purchase_level = 25
    purchase_cost = 150

    env_params = [
        time_limit,
        num_prods, num_ages, num_inv_levels, 
        target_ages, det_demands,
        holding_cost,
        mkt_start, mkt_step, mkt_max_age, 
        brd_start, brd_exp, brd_contrib,            
        q_weibull, beta_weibull, max_decay,
        action_type,
        purchase_rule, purchase_level, purchase_cost        
        ]


    ############################################
    load_model = False # evalue trained / train new models

    for repeat in range(args.n_repeats): # # of runs
        # initiate environment
        env = SalesBlendingEnv(*env_params)
        eval_env = SalesBlendingEnv(*env_params)

        # initiate networks
        model = DQN()
        target_model  = DQN()

        # load_model == True => run simulation of trained models
        if load_model:
            path = f"{args.prefix}_full_model_{repeat}.pt"
            model.load_state_dict(torch.load(path))

            env_params[0] = 100000 # # of periods
            random_env = SalesBlendingEnv(*env_params)
            state = random_env.reset()
            
            results = []
            iter_reward = []
            
            for step in range(int(random_env.time_limit)):
                current_inv = random_env.obs
                selected_action = model.act(state, 0)
                next_state, reward, done, expanded_act = random_env.step(selected_action)
                (sales,blending) = random_env.sales_blend_actions_counter[expanded_act]
                blending = [blending[k] for k in random_env.range_ages]
                results.append([step, current_inv, expanded_act, sales, blending, reward, random_env.obs])
                iter_reward.append(reward)
                print(step, current_inv, expanded_act, sales, blending, reward, random_env.obs)
                state = next_state
            
            print('\nReward: Sum', sum(iter_reward), 'Mean', np.mean(iter_reward), 'Max', max(iter_reward), 'Min', min(iter_reward))

            results = pd.DataFrame(results, columns=['Step', 'Current inventory', 'Action', 'Sales qty', 'Blending', 'Reward', 'Next inventory'])
            results.to_csv('results_full_model_{}.csv'.format(repeat))
            print('Saving done')


        else: # load_model = False => train new models
            # initiate optimizer
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, eps=1e-2)
            # initiate replay buffer
            replay_buffer = NaivePrioritizedBuffer(args.buffer_size)
            # train models
            run_data = train(
                model,
                target_model,
                env,
                eval_env,
                num_frames=args.num_frames,
                batch_size=args.batch_size,
                target_update_interval=args.target_update_interval,
                gamma=args.gamma,
                epsilon_start=args.epsilon_start,
                epsilon_final=args.epsilon_final,
                epsilon_decay=args.epsilon_decay,
                prefix=args.prefix,
                do_evals=args.do_evals,
                train_every=args.train_every,
                double_q=args.double_q,
                lr_decay=args.lr_decay
            )

            # save model
            path = f"{args.prefix}_full_model_{repeat}.pt"
            torch.save(model.state_dict(), path)
            print('Save model done.')

            # save trainning result
            filename = f"{args.prefix}_full_{repeat}.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.write(json.dumps(run_data))
            print('Save results done.')


