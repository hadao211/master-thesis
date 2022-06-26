import sales_blending_env
import numpy as np, pandas as pd, multiprocessing as mp
import time, itertools, copy

from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager


#########################################
# BASE CASE: environment params
time_limit = 1000
num_prods = 2           #int, #number of target products sold
num_ages = 7            #int, #number of ages in inventory
num_inv_levels = 7      #int, #inventory level in discrete state case
target_ages = [3,5]     #List[int], #ages of target products sold                
det_demands = [3,3]     #List[int], #deterministic demand for each product
holding_cost = 24       #float, #holding cost per year
mkt_start = 24; mkt_step = 24; mkt_max_age = 3  #float, float, int # supermarket price
brd_start = None; brd_exp = None; brd_contrib = [333, 1000] # branded product price
q_weibull= 0.875; beta_weibull= 0.8; max_decay=0.3 #float, #params of decay distribution


action_type = 'full' # 'sales' 
purchase_rule = 'upto'
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



#######################################################################################
def bellman_optimality_update(  profit, before_decay,
                                shm_val_name, 
                                shm_next_value_name, 
                                s
                                ):
    # Locate the shared memory by its name
    shm_val = SharedMemory(shm_val_name)
    shm_next_value = SharedMemory(shm_next_value_name)


    # Create the np.array from the buffer of the shared memory
    decay_val = np.frombuffer(buffer=shm_val.buf)
    decay_next_value = np.frombuffer(buffer=shm_next_value.buf)


    assert len(profit) == len(before_decay) # check length
    q = copy.deepcopy(profit) # = contributions from branded products + outdating
    q = q.astype('float')
    for idx,before_decay_enc in enumerate(before_decay):
        if before_decay_enc == -1: # invalid inventory state
            q[idx] = (-np.inf)
        else:
            assert before_decay_enc % num_inv_levels == 0
            decay_idx = before_decay_enc // num_inv_levels # convert index before decay (num_ages) -> index before decay (num_ages-1)
            decay = decay_val[decay_idx] # contributions from decayed 
            next_state_value = decay_next_value[decay_idx] # pre-calculated each call, = sum of p*v
            q[idx] += decay + 0.999*next_state_value

    best_action = np.argmax(q)                          # best action
    best_value = q[best_action]                        # value of best action
    
    return (s, best_action, best_value)



#######################################################################################
# simulation of policy obtained from value iteration
def simulation (env_params, policy, n_iters, mode='extensive'):
    results = []
    random_env = sales_blending_env.SalesBlendingEnv(*env_params)
    random_env.load_data()

    for iter in range(n_iters):
        random_env.reset()

        iter_reward = []
        
        for step in range(int(random_env.time_limit)):
            current_inv = random_env.obs
            current_state_enc = random_env.obs_dict_rev[tuple(current_inv)]
            selected_action = policy[current_state_enc] # select action based on policy
            next_state, reward, done, expanded_act = random_env.step(selected_action, mode=mode) # take action
            (sales,blending) = random_env.sales_blend_actions_counter[expanded_act]
            blending = [blending[k] for k in random_env.range_ages]
            results.append([iter, step, current_inv, expanded_act, sales, blending, reward, random_env.obs])
            iter_reward.append(reward)
            print(iter, step, current_inv, expanded_act, sales, blending, reward, random_env.obs)
        
        print('Iter', iter, '\nReward: Sum', sum(iter_reward), 'Mean', np.mean(iter_reward), 'Max', max(iter_reward), 'Min', min(iter_reward))

    results = pd.DataFrame(results, columns=['Iteration', 'Step', 'Current inventory', 'Action', 'Sales qty', 'Blending', 'Reward', 'Next inventory'])
    results.to_csv('results.csv')
    print('Saving done')

    return results


##############################################################
# simulation with greedy policy of sales
def simulation_greedy(env_params, n_iters, mode='extensive'):
    results = []
    random_env = sales_blending_env.SalesBlendingEnv(*env_params)
    # sort action enc from high sales qty to low, from older product to younger
    ind = sorted(range(len(random_env.sales_actions)),
                key=lambda i: (random_env.sales_actions[i][1],random_env.sales_actions[i][0]), 
                reverse=True)  

    for iter in range(n_iters):
        random_env.reset()

        iter_reward = []
      
        for step in range(int(random_env.time_limit)):
            current_inv = random_env.obs
            mask = random_env.mask
            # get mask of sales sub-actions from full mask
            mask_sales = [np.prod(i) for i in np.split(mask, np.cumsum(random_env.sales_blend_lens[:-1]))]

            for i in ind:
                if mask_sales[i] == 0:
                    selected_sales_action = i # select valid action with highest ranking
                    break

            next_state, reward, done, expanded_act = random_env.step(selected_sales_action, lod=0, mode=mode) # take action
            (sales,blending) = random_env.sales_blend_actions_counter[expanded_act]
            blending = [blending[k] for k in random_env.range_ages]
            results.append([iter, step, current_inv, expanded_act, sales, blending, reward, random_env.obs])
            iter_reward.append(reward)
            print(iter, step, current_inv, expanded_act, sales, blending, reward, random_env.obs)
        
        print('Iter', iter, '\nReward: Sum', sum(iter_reward), 'Mean', np.mean(iter_reward), 'Max', max(iter_reward), 'Min', min(iter_reward))

    results = pd.DataFrame(results, columns=['Iteration', 'Step', 'Current inventory', 'Action', 'Sales qty', 'Blending', 'Reward', 'Next inventory'])
    results.to_csv('results.csv')
    print('Saving done')

    return results




#######################################################################################
if __name__ == '__main__':

    env = sales_blending_env.SalesBlendingEnv(*env_params)
    env.reset() 
    nS = env.num_inv_levels**env.num_ages       # number of possible states
    nA = env.max_avail_actions                 # number of available actions


    # PREPARE ENV ####################################################
    env.prepare_env()



    # TRAINING #######################################################
    gamma = 0.999      # discount factor
    epsilon = 1e-2


    ##########################################
    # running iterations
    bounds = range(50000, nS+50000, 50000)
    # bounds = [nS+1]

    V = np.zeros(nS) # np.load('V.npy')            # initialize v(0) to arbitrary value
    prev_V = copy.deepcopy(V)
    pi = np.zeros(nS)  # np.load('pi.npy')     # initialize policy

    print('Order-up-to level', env.purchase_level)
    print('Epsilon {}'.format(epsilon))
    start_time = time.time()

    iter = 0
    while True:
        print('Start iter', iter)
        t = time.time()

        U = 0
        u = np.inf

        prev_interval = 0
        results = []


        ################################
        # load saved data
        start = time.time()

        obs_dict = dict(enumerate(itertools.product(*[range(num_inv_levels) for _ in range(1,num_ages+1)])))
        obs_dict_rev = dict(map(reversed, obs_dict.items()))

        decay_val = np.load('data_base/decay_val.npy') # decay reward
        decay_len= np.load('data_base/decay_len.npy') # number of after decay state
        decay_prob = np.load('data_base/decay_prob_flatten.npy') #decay prob
        
        decay_after_idx = np.load('data_base/decay_after_flatten.npy') # after decay state
        decay_after_value = V[decay_after_idx] # current value estimates corresponding to each state after decay
        del decay_after_idx

        decay_len = np.cumsum(decay_len)[:-1] # change from splitting length to splitting indicies
        # current value estimates corresponding to each state before decay
        decay_next_state_value = np.array([sum(np.prod([p,v], axis=0)) for p,v in \
                                            zip(np.split(decay_prob, decay_len), \
                                                np.split(decay_after_value, decay_len))]) 
        del decay_prob
        del decay_after_value
        del decay_len

        print('loading data done after', time.time()-start)


           
        ##############################################################

        with SharedMemoryManager() as smm:
            # Create share memories
            shm_val = smm.SharedMemory(decay_val.nbytes)
            # Create a np.recarray using the buffer of shm
            to_shm_val = np.frombuffer(buffer=shm_val.buf, dtype=decay_val.dtype)
            # Copy the data into the shared memory
            np.copyto(to_shm_val, decay_val)
            del decay_val

            shm_next_value = smm.SharedMemory(decay_next_state_value.nbytes)
            to_shm_next_value = np.frombuffer(buffer=shm_next_value.buf, dtype=decay_next_state_value.dtype)
            np.copyto(to_shm_next_value, decay_next_state_value)
            del decay_next_state_value


            for interval_end in bounds:
                # load state-dependent data
                # self.state_mask = np.load("state_mask.npy")
                state_profit = np.load("data_base/state_profit_{}.npy".format(interval_end))
                state_before_decay = np.load("data_base/state_next_state_{}.npy".format(interval_end))


                args = list(zip(state_profit, state_before_decay,
                                itertools.repeat(shm_val.name), 
                                itertools.repeat(shm_next_value.name),
                                range(prev_interval, prev_interval+len(state_profit)),
                                ))
                print(args[0])
                
                # run multi-processing
                with mp.Pool(processes=6) as pool:
                    temp = pool.starmap(bellman_optimality_update, args)  

                results += temp
                prev_interval = interval_end  

            for i, act, val in results:
                V[i] = val
                pi[i] = act                


        U = np.max(np.abs(prev_V - V))             # assign the change in value per iteration to U and u
        u = np.min(np.abs(prev_V - V))

        if (U-u)/u < epsilon:      # check converge condition
            print('Iter', iter, 'U', U, 'u', u, '\nDone after', time.time() - t)                                       
            break                                         # if change gets to negligible 
                                                        # --> converged to optimal value         
        else:
            print('Iter', iter, 'U', round(U, 2), 'u', round(u,2), '\nCurrent running time', time.time() - t)
        
        prev_V = copy.deepcopy(V)
        np.save('V', V)
        np.save('pi', pi)
        iter += 1


    print('Done.', time.time()-start)




# TESTING ########################################################
env_params[0] = 5e6

pi = np.load('pi.npy')
start = time.time()
print('Start simulation')
results = simulation(env_params, pi, 1, mode='extensive') # ,
# results = simulation_greedy(env_params, 1,  mode='extensive')
print('Done after', time.time()-start)

