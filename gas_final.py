import math, random, time, copy, json, os, argparse
import numpy as np, pandas as pd
from numpy.core.numeric import full
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


################################################################
# parse arguments
parser = argparse.ArgumentParser()

# DQN params
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--buffer_size", default=20000, type=int)
parser.add_argument("--target_update_interval", default=400, type=int)
parser.add_argument("--gamma", default=0.999, type=float)
parser.add_argument("--double_q", default=True, action="store_true")
parser.add_argument("--epsilon_start", default=1.0, type=float)
parser.add_argument("--epsilon_final", default=0.1, type=float)
parser.add_argument("--epsilon_lead_in", default=0, type=int)
parser.add_argument("--epsilon_decay", default=20000, type=int)
parser.add_argument("--alpha_start", default=0.5, type=float)
parser.add_argument("--lr_decay", default=True, action="store_true")
# GAS params
parser.add_argument("--lod_lead_in", default=20000, type=int)
parser.add_argument("--eps_h_start", default=0.5, type=float)
parser.add_argument("--eps_h_final", default=0.1, type=float)
parser.add_argument("--eps_h_decay", default=40000, type=int)
# parser.add_argument("--max_targets", default=False, type=bool) # !!!!
# Training params
parser.add_argument("--num_frames", default=100000, type=int)
parser.add_argument("--train_every", default=4, type=int)
parser.add_argument("--n_repeats", default=5, type=int)
parser.add_argument("--only_train_last", default=False, action="store_true") # only train the space of the last level
parser.add_argument("--do_evals", default=True, action="store_true")
parser.add_argument("--prefix", default="results/experimental2/test", type=str)
# Environment param
parser.add_argument("--env_id", default="SalesBlendingEnv", type=str)
parser.add_argument("--n_levels", default=2, type=int)
parser.add_argument("--simplify_mode", default='extensive', type=str)


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

        self.buffer.append((expand_state, action, lod, reward, expand_next_state, done, expanded_act)) # add to the right

    def sample(self, batch_size):
        state, action, lod, reward, next_state, done, expanded_act = zip(*random.sample(self.buffer, batch_size))
        keys = ['action_mask', 'real_obs']
        concate_state = dict(zip(keys, [np.concatenate([d[k] for d in state]) for k in keys]))
        concate_next_state = dict(zip(keys, [np.concatenate([d[k] for d in next_state]) for k in keys]))
        return concate_state, action, lod, reward, concate_next_state, done, expanded_act

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

    def push(self, state, action, lod, reward, next_state, done, expanded_act):
        expand_state = {k: np.expand_dims(v, 0) for k,v in state.items()}
        expand_next_state = {k: np.expand_dims(v, 0) for k,v in next_state.items()}

        max_prio = self.priorities.max() if self.buffer else 1.0 # set highest priority to new experience

        if len(self.buffer) < self.capacity:
            self.buffer.append((expand_state, action, lod, reward, expand_next_state, done, expanded_act))
        else:
            self.buffer[self.pos] = (expand_state, action, lod, reward, expand_next_state, done, expanded_act)

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
        concate_next_state = dict(zip(keys, [np.concatenate([d[k] for d in batch[4]]) for k in keys]))
        
        actions     = batch[1]
        lods        = batch[2]
        rewards     = batch[3]
        dones       = batch[5]
        expanded_acts = batch[6]

        return concate_state, actions, lods, rewards, concate_next_state, dones, expanded_acts, indices


    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


USE_CUDA = False 


#############################################################################################

class GrowingActionDQN(nn.Module):
    def __init__(self, env, n_levels=2):
        super(GrowingActionDQN, self).__init__()
        self.autoclip = AutoClip()
        self.sales_blend_lens = env.sales_blend_lens
        self.n_actions = env.n_actions

        self.encoder = nn.Sequential(
            nn.Linear(env.observation_space['real_obs'].shape[0], 128),
            Nonlin(),
            nn.Linear(128, 64),
            Nonlin(),
        )
        self.n_levels = n_levels # no. of action space levels

        self.decoders = nn.ModuleList()
        self.evaluators = nn.ModuleList()
        for i in range(n_levels):
            self.decoders.append(nn.Linear(64,64))
            self.evaluators.append(nn.Linear(64, env.n_actions[i]))   
            if i > 0:
                self.evaluators[-1].bias.detach().zero_()
                self.evaluators[-1].weight.detach().mul_(0.01)


    def forward(self, x, action_mask):
        allQs = [] # q values in different levels
        allDeltas = [] # delta of q values between different levels

        embed = self.encoder(x)
        for i in range(self.n_levels):
            embed = fnonlin(self.decoders[i](embed))
            levelQ = self.evaluators[i](embed)

            if i == 0:
                # mask = batch * [0: valid action, 1: invalid action]
                # transform mask from final level to lower level
                mask = torch.column_stack([y.prod(dim=1) for y in action_mask.split(self.sales_blend_lens, dim=1)]) 
                mask = torch.where(mask==1, FLOAT_MIN, 0.0)
                
                totalQ = levelQ + mask
            else:                 
                mask = torch.where(action_mask==1, FLOAT_MIN, 0.0) 
                levelQ += mask
                allDeltas.append(levelQ)
                # link between action level 0 (sale action) with action level 1 (sale + blending)
                totalQ = levelQ + totalQ.repeat_interleave(torch.tensor(self.sales_blend_lens), dim=1)
                torch.clamp(totalQ, FLOAT_MIN, FLOAT_MAX)
            allQs.append(totalQ)
        return allQs, allDeltas


    def act(self, state, epsilon, eps_rate, lod): # state = {'action_mask': (max action,), 'real_obs': (num age,)}
        # without updating weights, calculate q values for all actions 
        with torch.no_grad():
            obs   = torch.tensor(state['real_obs'], dtype=torch.float).unsqueeze(0)
            m = torch.tensor(state['action_mask'], dtype=torch.float).unsqueeze(0)
            q_values, _ = self.forward(obs, m)
        q_value = q_values[lod]
        action  = q_value.max(1)[1].item() # lod action  # ALREADY CONSIDER MASK IN FORWARD FUNCTION
        
        # full exploitation: select action with highest value
        if random.random() >= epsilon:
            return action

        # exploration: select random action
        else:           
            mask = state['action_mask']
            mask_sales = [y.prod().item() for y in torch.tensor(state['action_mask']).split(self.sales_blend_lens)]          
            # full exploration            
            if lod == 0:
                action = random.choice( \
                    list(filter(lambda a: a[1] == 0, enumerate(mask_sales)))
                    )[0]  # randomly select VALID action 
            else:
                # hierarchical exploration
                if random.random() >= eps_rate:
                    intervals = np.cumsum(self.sales_blend_lens)
                    interval_index = np.searchsorted(intervals, action, side='right')
                    if interval_index == 0:
                        random_range = range(intervals[interval_index])
                    else:
                        random_range = range(intervals[interval_index-1], intervals[interval_index])
                    action = random.choice(
                        list(filter(lambda a: mask[a]==0, random_range))
                    )                    
                else: # full exploration
                    if lod == self.n_levels - 1:
                        mask = state['action_mask']
                    else:
                        mask = [y.prod().item() for y in torch.tensor(state['action_mask']).split(self.sales_blend_lens)]          
                    action = random.choice( \
                        list(filter(lambda a: a[1] == 0, enumerate(mask)))
                        )[0]  # randomly select VALID action         

            return action




#################################################################################################

def expand_q_values(allQs, n_levels): # expand q values at lower space to higher space
    expandedQs = []
    for i, q in enumerate(allQs):
        if i < n_levels - 1:
            expandedQs.append(q.repeat_interleave(torch.tensor(env.sales_blend_lens), dim=1))
        else:
            expandedQs.append(q)
    return expandedQs


def compute_td_loss(batch_size,
                    gamma,
                    n_levels,
                    frame_idx, # num_frames, # for lr decay
                    alpha,
                    only_train_last=False,
                    # max_targets=False,
                    double_q=True,
                    lr_decay=False
                    ):
    # sampling from replay buffer
    state, action, lod, reward, next_state, done, expanded_actions, indices = replay_buffer.sample(batch_size, alpha)

    state_obs       = torch.tensor(state['real_obs'], dtype=torch.float)
    state_mask      = torch.tensor(state['action_mask'], dtype=torch.float)
    next_state_obs  = torch.tensor(next_state['real_obs'], dtype=torch.float)
    next_state_mask = torch.tensor(next_state['action_mask'], dtype=torch.float)
    
    action     = torch.tensor(action, dtype=torch.long)
    lod        = torch.tensor(lod, dtype=torch.long)
    reward     = torch.tensor(reward, dtype=torch.float)
    done       = torch.tensor(done, dtype=torch.float)
    expanded_actions = torch.tensor(expanded_actions, dtype=torch.long)

    # Q-network/ online network
    q_values, deltas = model(state_obs, state_mask)
    q_values = expand_q_values(q_values, n_levels)
    # gather q values corresponding to action at final level
    q_values_taken = [q.gather(1, expanded_actions.unsqueeze(1)).squeeze(1) for q in q_values]

    # target network (evaluation)
    with torch.no_grad():
        next_q_values = expand_q_values(target_model(next_state_obs, next_state_mask)[0], n_levels)

    # double q learning: use online model to select action
    if double_q:
        next_q_values_current_model = expand_q_values(model(next_state_obs, next_state_mask)[0], n_levels)

    # select action with highest q value 
    if double_q: # double q learning (Q-network)
        next_max_actions = [q.max(1)[1] for q in next_q_values_current_model] # ALREADY CONSIDER MASK IN FORWARD FUNCTION
        next_q_values_max = [q.gather(1, a.unsqueeze(1)).squeeze(1) for q,a in zip(next_q_values, next_max_actions)]
    else: # target model
        next_q_values_max = [q.max(1)[0] for q in next_q_values]

    targets = [reward + gamma * next_q * (1 - done) for next_q in next_q_values_max]

    loss = torch.tensor(0.0, requires_grad=True)
    prios = torch.ones(batch_size, requires_grad=False)*1e-5

    # min_lod = int(lod.min().item())
    # if max_targets and not only_train_last:
    #     maxed_targets = [targets[0]]
    #     for l, t in enumerate(targets[1:]):
    #         maxed_targets.append(torch.max(t,maxed_targets[-1].clone()))
    #     targets = maxed_targets
    
    # calculate loss 
    if only_train_last:
        loss = loss + F.smooth_l1_loss(q_values_taken[-1], targets[-1])
        prios = prios + torch.abs(q_values_taken[-1] - targets[-1])
    else:
        masksum = torch.tensor(0.0)
        # compute loss for each level
        for i, (q, target) in enumerate(zip(q_values_taken, targets)):
            mask = lod.le(i).float() # only use experience of lower or equal lod
            if mask.sum().item() <= 32.0:
                continue
            masksum += mask.sum()
            loss = loss + (F.smooth_l1_loss(q, target.detach(), reduction="none")*mask).sum()
            prios = prios + torch.abs(q - target)*mask
        
        # mean reduction
        loss = loss / masksum
        prios = prios / masksum

    reg_loss = torch.tensor(0.0, requires_grad=True)

    # update priorities of replay buffer
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())

    # learning rate decay
    if lr_decay:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*(1 - 1/(args.num_frames))

    # backward
    optimizer.zero_grad()
    (loss + reg_loss).backward()
    # gradient clipping
    if not only_train_last and frame_idx<=args.lod_lead_in:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    else:
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        model.autoclip(model, 10)
    optimizer.step()

    return loss, reg_loss, q_values_taken[-1].mean()

# update target network using weights from online network
def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


##################################################################################
def eval(n_levels, n_episodes=10):
    eval_rewards = []
    eval_episode_reward = 0
    state = eval_env.reset()

    for _ in range(n_episodes):
        done = False
        while not done: 
            # act with highest lod for eval
            action = model.act(state, 0, 0, n_levels-1)
            next_state, reward, done, expanded_act = eval_env.step(action, lod=n_levels-1, mode=args.simplify_mode)
            state = next_state
            eval_episode_reward += reward

            if done:
                state = eval_env.reset()
                eval_rewards.append(eval_episode_reward)
                eval_episode_reward = 0
    return(eval_rewards)


##################################################################################
def get_lod(only_train_last, n_levels, frame_idx):
    if only_train_last:
        lod = n_levels - 1
        plod = 0
    else:
        plod = lod_by_frame(frame_idx)
        p_grow_lod, baselod = math.modf(plod)
        lod = int(baselod + int(np.random.random() < p_grow_lod))
    return lod, plod


##################################################################################
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
    n_levels,
    train_every,
    only_train_last,
    max_targets,
    double_q,
    lr_decay
):
    run_data = {
        "loss": [],
        "reg_loss": [],
        "value": [],
        "lod": [],
        "returns": []
    }

    actions_taken = []
    episode_reward = 0

    state = env.reset()
    frame_idx = 1

    lod, plod = get_lod(only_train_last, n_levels, frame_idx)
    env_running_time = 0
    start_time = time.time()
    
    while frame_idx <= num_frames:
        # get parameters
        alpha = alpha_by_frame(frame_idx)
        epsilon, eps_rate = epsilon_by_frame(frame_idx)
        run_data["lod"].append((frame_idx, lod))
        action = model.act(state, epsilon, eps_rate, lod)

        # obtain experience
        temp_time = time.time()
        next_state, reward, done, expanded_act = env.step(action, lod=lod, mode=args.simplify_mode) #, mask_preload=True
        env_running_time += time.time()-temp_time
        replay_buffer.push(state, action, lod, reward, next_state, done, expanded_act)
        actions_taken.append(expanded_act)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            lod, plod = get_lod(only_train_last, n_levels, frame_idx)
            run_data["returns"].append((frame_idx, episode_reward))
            episode_reward = 0

        # compute TD error
        if len(replay_buffer) > 1000 and frame_idx % train_every == 0: 
            loss, reg_loss, value = compute_td_loss(batch_size,
                                                    gamma,
                                                    n_levels,
                                                    frame_idx, 
                                                    alpha,
                                                    only_train_last=only_train_last,
                                                    max_targets=max_targets,
                                                    double_q=double_q,
                                                    lr_decay=lr_decay)
            # save training results
            run_data["loss"].append((frame_idx, loss.item()))
            run_data["reg_loss"].append((frame_idx, reg_loss.item()))
            run_data["value"].append((frame_idx, value.item()))

        # update target network
        if frame_idx % target_update_interval == 0:
            update_target(model, target_model)

        # report logging data
        if len(replay_buffer) > 1000 and frame_idx % 1000 == 0:
            print("{:8} frames | R {:.2f} (max {:.2f}) |  eps {:.2f} {:.2f} | Q {:.2f} | loss {:.6f} ~ {:4f}% | common act {} : {:.2f} | lod {:.2f}".format(
                frame_idx,                                          # current frame indx
                np.mean([x[1] for x in run_data["returns"][-10:]]), # smoothed mean return of last 10 episodes
                np.max([x[1] for x in run_data["returns"]]),        # current max episode return
                epsilon, eps_rate,                                  # double epsilon greedy exploration
                np.mean([x[1] for x in run_data["value"][-10:]]),   # smoothed mean value of last 10 episodes
                np.mean([x[1] for x in run_data["loss"][-10:]]),    # smoothed mean loss of last 10 episodes
                100*np.mean([x[1] for x in run_data["loss"][-10:]]) / np.mean([x[1] for x in run_data["value"][-10:]]), # ratio of loss/value
                env.sales_blend_actions_counter[Counter(actions_taken).most_common(1)[0][0]], # most common action
                Counter(actions_taken).most_common(1)[0][1], # most common action (# of times)
                run_data["lod"][-1][1] # current lod
                ))
            actions_taken.clear()
        
        # run evaluation simulation when expanding space
        if frame_idx == args.lod_lead_in:
            eval_reward = eval(n_levels)
            print('Evaluation: Mean {:6f} | Max {:6f} | Min {:6f} | Std {:6f}'.format(
                np.mean(eval_reward),
                np.max(eval_reward),
                np.min(eval_reward),
                np.std(eval_reward)
            ))

        frame_idx +=1
    
    finish_time = time.time()
    print('Finish training model after', finish_time-start_time)
    print('Env running time {} | Model running time {}'.format(
        env_running_time,
        finish_time-start_time-env_running_time
    ))

    # evaluate at the end of training
    if do_evals:
        eval_reward = eval(n_levels)
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
        return args.epsilon_final + max(0, (args.epsilon_start - args.epsilon_final) * min(1, (1 - (frame_idx - args.epsilon_lead_in) / args.epsilon_decay))), \
            args.eps_h_final + max(0, (args.eps_h_start - args.eps_h_final) * min(1, (1 - (frame_idx - args.lod_lead_in) / args.eps_h_decay)))
            
    def lod_by_frame(frame_idx):
        if frame_idx < args.lod_lead_in:
            return 0
        else:
            return args.n_levels-1

    def alpha_by_frame(frame_idx):
        if frame_idx <= args.lod_lead_in:
            return 0
        else:
            return 0 + max(0, (args.alpha_start - 0) * min(1, (1 - (frame_idx - args.lod_lead_in) / (args.num_frames - args.lod_lead_in))))


    #########################################
    # Environment setting
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

    action_type = 'full' # full or sales
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

    for repeat in range(args.n_repeats): # of runs
        # initiate environment
        start = time.time()
        env = SalesBlendingEnv(*env_params)
        eval_env = SalesBlendingEnv(*env_params)
        print('Environment setting done', time.time()-start)

        # initiate networks
        model = GrowingActionDQN(env, n_levels=args.n_levels)
        target_model  = GrowingActionDQN(env, n_levels=args.n_levels)

        # load_model == True => run simulation of trained models
        if load_model:
            path = f"{args.prefix}_model_{repeat}.pt"
            model.load_state_dict(torch.load(path))

            env_params[0] = 100000 # # of periods
            random_env = SalesBlendingEnv(*env_params)
            state = random_env.reset()
            
            results = []
            iter_reward = []
            
            for step in range(int(random_env.time_limit)):
                current_inv = random_env.obs
                selected_action = model.act(state, 0, 0, args.n_levels-1)
                next_state, reward, done, expanded_act = random_env.step(selected_action, lod=args.n_levels-1, mode=args.simplify_mode)
                (sales,blending) = random_env.sales_blend_actions_counter[expanded_act]
                blending = [blending[k] for k in random_env.range_ages]
                results.append([step, current_inv, expanded_act, sales, blending, reward, random_env.obs])
                iter_reward.append(reward)
                print(step, current_inv, expanded_act, sales, blending, reward, random_env.obs)
                state = next_state
            
            print('\nReward: Sum', sum(iter_reward), 'Mean', np.mean(iter_reward), 'Max', max(iter_reward), 'Min', min(iter_reward))

            results = pd.DataFrame(results, columns=['Step', 'Current inventory', 'Action', 'Sales qty', 'Blending', 'Reward', 'Next inventory'])
            results.to_csv('results_{}.csv'.format(repeat))
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
                n_levels=args.n_levels,
                train_every=args.train_every,
                only_train_last=args.only_train_last,
                max_targets=args.max_targets,
                double_q=args.double_q,
                lr_decay=args.lr_decay
            )

            # save trainning result
            filename = f"{args.prefix}{repeat}.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.write(json.dumps(run_data))
            print('Save results done.')

            # save model
            path = f"{args.prefix}_model_{repeat}.pt"
            torch.save(model.state_dict(), path)
            print('Save model done.')

