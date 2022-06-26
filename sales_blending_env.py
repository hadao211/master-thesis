# require python >= 3.8
import gym
from gym import spaces, utils

import math, itertools, copy, time, csv, json
import numpy as np
from collections import defaultdict, Counter
from typing import Counter, List

#############################################################
import logging

from numpy.core.fromnumeric import repeat
logger = logging.getLogger(__name__)



####################################################################################
def define_action_space(det_demands, target_ages, num_ages, num_inv_levels):
    # start_time = time.time()

    # initialize
    sales_blend_actions = np.zeros((1,len(target_ages)+sum(det_demands))) 
    sales_blend_actions_counter = [ ( tuple([0]*len(det_demands)), \
                                      Counter()                      )]
    sales_blend_lens = [1]  # sales = 0
    sales_actions = list(itertools.product(*[range(d+1) for d in det_demands]))
    # print('Finish sales', time.time()-start_time)

    # create a dictionary of different blending options with sum for all possible order quantities
    min_target_age = min(target_ages)
    sum_blend = dict()
    for order_qty in range(max(det_demands)+1):
        sum_blend.update({order_qty: defaultdict(list)})
        blend_comb = itertools.combinations_with_replacement(range(1, num_ages+1), order_qty)
        for b in blend_comb:
            temp_sum = sum(b)
            if temp_sum >= (min_target_age*order_qty):
                sum_blend[order_qty][temp_sum].append( (list(b), Counter(b)) ) 
    # print('Finish sum blend', time.time()-start_time)
    
    # match blending actions with sales actions
    for sales_act in sales_actions:
        if sum(sales_act) == 0:
            continue

        blend_acts = []
        curr_pos = 0

        # gather all possible blending action for each product
        while curr_pos < len(sales_act):
            sum_age_target = sales_act[curr_pos] * target_ages[curr_pos]
            n_pad = det_demands[curr_pos]-sales_act[curr_pos]

            if len(blend_acts) == 0:
                for i in sum_blend[sales_act[curr_pos]]:
                    if i >= sum_age_target:
                        blend_acts += sum_blend[sales_act[curr_pos]][i]  
                # padding
                act, count = zip(*blend_acts)
                blend_acts = list( zip(np.pad(act, ((0,0), (0,n_pad)) ), 
                                    count ) ) 
            else:
                temp = []
                for (j, count_j) in blend_acts: # previous actions
                    for i in sum_blend[sales_act[curr_pos]]: # current action to check
                        if i >= sum_age_target: # age sum condition
                            for (act, count_act) in sum_blend[sales_act[curr_pos]][i]:
                                c = count_j + count_act
                                if not bool(c) or c.most_common(1)[0][1] < num_inv_levels: # max inventory condition
                                    temp.append( (np.hstack( ( j, np.pad(act,(0,n_pad)) ) ), 
                                                        c) ) 
                
                blend_acts = copy.deepcopy(temp)
            curr_pos += 1

        # save blending option for each sales option
        sales_blend_lens.append(len(blend_acts))
        all_blend_opts, all_blend_opts_counter = zip(*blend_acts)
        sales_blend_actions = np.vstack(( sales_blend_actions, 
                                         np.hstack((np.tile(sales_act, (len(all_blend_opts),1)), 
                                                            all_blend_opts))
                                        )) 
        sales_blend_actions_counter += [(sales_act, blend_counter) for blend_counter in all_blend_opts_counter]
        # print('Finish', time.time()-start_time)
        
        
    # with open('action_space.csv', 'w') as f:
    #     # using csv.writer method from CSV package
    #     write = csv.writer(f)
    #     write.writerows(sales_blend_actions)

    return sales_actions, sales_blend_actions_counter, sales_blend_lens, sales_blend_actions 



# create mask corresponding to an inventory state 
def get_mask(sales_blend_actions_counter, sales_blend_lens,
                inventory, range_age, action_type = 'full'):
    assert action_type in ['sales', 'full']
    inv_counter = Counter(dict(zip(range_age, inventory)))
    mask = []
    for _,bl in sales_blend_actions_counter:
        if all(inv_counter[k] >= bl[k] for k in bl): # check
            mask.append(0) # valid action: mask = 0
        else:
            mask.append(1) # invalid action: mask = 1
    if action_type == 'sales': # aggregate to create mask corresponding to sales action space
        return [np.prod(i) for i in np.split(mask, np.cumsum(sales_blend_lens[:-1]))]
    else:
        return mask
        



# define the decay prob for each inventory scenerio
def define_decay_rate(num_ages, range_ages, num_inv_levels,
                      q_weibull= 0.8415, beta_weibull= 0.8, max_decay=0.2):

    #probability mass function 
    pmf_weibull = [q_weibull**((k-1)**(beta_weibull)) - q_weibull**((k)**(beta_weibull)) for k in range_ages] 
    decay_prob = [max_decay*pmf_weibull[k-1]/sum(pmf_weibull) for k in range_ages] # normalized

    # initialize
    decay_dict = dict.fromkeys(range(1,num_inv_levels), None)
    
    # fill in decay dictionary
    for inv_level in decay_dict:
        decay_dict[inv_level] = np.zeros((num_ages, inv_level+1))

        for age in range_ages:
            decay_cdf_prob = 0
            for decay_amount in range(inv_level+1):    
                # pmf of a certain decay amount follows binomial distribution
                decay_cdf_prob += math.comb(inv_level, decay_amount) * (decay_prob[age-1]**decay_amount) * \
                                        ((1-decay_prob[age-1])**(inv_level-decay_amount))
                decay_dict[inv_level][age-1, decay_amount] = decay_cdf_prob
                    
    return decay_dict





# ref: https://github.com/ray-project/ray/blob/560fd1556807ade19ac940328d4796311d821eb6/rllib/examples/env/parametric_actions_cartpole.py#L7
##############################################################
class SalesBlendingEnv(gym.Env, utils.EzPickle):
  
    def __init__(self,
               time_limit: int, # episode length
               num_prods: int,  # number of branded products
               num_ages: int,   # number of age classes
               num_inv_levels: int, # number of inventory levels of each age class (including 0)
               target_ages: List[int], # target ages of branded products
               det_demands: List[int], # demands of branded products
               holding_cost: float, 
               mkt_start: float, mkt_step: float, mkt_max_age: int, # parameters to define profit contribution from supermarkets 
               brd_start: float, brd_exp: float, brd_contrib: List[float], # parameterx to define profit contributions of branded products
               q_weibull, beta_weibull, max_decay, # parameters of decay distribution
               action_type = 'full', # sales / full
               purchase_rule = 'fixed', # fixed / upto
               purchase_level = None, purchase_cost = None
               ):
        
        assert action_type in ['sales', 'full']
        assert purchase_rule in ['fixed', 'upto']
        assert not (purchase_rule == 'upto' and (purchase_level is None or purchase_cost is None)) 
        
        #set parameters and initialization
        self.action_type = action_type

        self.purchase_rule = purchase_rule
        self.purchase_level = purchase_level
        self.purchase_cost = purchase_cost

        self.time_limit = time_limit

        self.num_ages = num_ages
        self.num_inv_levels = num_inv_levels # include 0
        self.num_prods = num_prods
        self.target_ages = target_ages
        self.det_demands = np.array(det_demands)

        self.total_demand = sum(self.det_demands)
        self.range_ages = list(range(1, self.num_ages+1))
        self.range_products = list(range(len(target_ages)))

        self.sales_actions, self.sales_blend_actions_counter, self.sales_blend_lens, self.sales_blend_actions = \
            define_action_space(self.det_demands, self.target_ages, self.num_ages, self.num_inv_levels)

        self.n_actions = [len(self.sales_actions), len(self.sales_blend_actions_counter)]

        #ref: https://medium.com/swlh/states-observation-and-action-spaces-in-reinforcement-learning-569a30a8d2a1
        if action_type == 'sales':
            self.max_avail_actions = self.n_actions[0]
        else:
            self.max_avail_actions = self.n_actions[1]

        # define FULL action space: multi-dimensional -> one dimensional
        self.action_space = spaces.Discrete(self.max_avail_actions)
        self.state_mask = None

        # define FULL observation space    
        self.observation_space = \
        spaces.Dict({
            "action_mask": spaces.MultiDiscrete(np.repeat(2, self.max_avail_actions)), # spaces.Box( 0, 1, shape=(self.max_avail_actions,) ),        
            "real_obs": spaces.MultiDiscrete(np.repeat(self.num_inv_levels, self.num_ages)) # spaces.Box(0, self.num_inv_levels-1, shape=(self.num_ages, )) 
        })       


        # reward parameters
        self.holding_cost = holding_cost
        stock_value = [holding_cost*i for i in self.range_ages]

        mkt_max = mkt_max_age * mkt_step
        supermarket_price = [min(mkt_start + (i-1)*mkt_step, mkt_max) for i in self.range_ages]
        self.supermarket_contrib = list(np.array(supermarket_price) - np.array(stock_value))    

        if brd_start is None or brd_exp is None:
            self.brand_contrib = brd_contrib
        else: 
            self.brand_contrib = [int(brd_start*(brd_exp**i)) for i in self.range_products]


        # decay prob
        self.decay_dict = define_decay_rate(self.num_ages, self.range_ages, self.num_inv_levels,
                                            q_weibull, beta_weibull, max_decay)


    ###############################################################
    # reset
    def reset(self):
        self.done = False
        self.iter = 0

        self.obs = np.random.randint(self.num_inv_levels, size=self.num_ages).astype('float')
        self.mask = get_mask(self.sales_blend_actions_counter, self.sales_blend_lens, self.obs, self.range_ages, self.action_type)
        observation = {"action_mask": self.mask, 
                        "real_obs": self.obs}
        
        return observation

    
    ###############################################################
    # helper functions
    def take_blend_action(self, init_inventory, blend_action_enc):
        """
        Full action: sales + blending
        """        
        # read action input
        sales_act, blend_act = self.sales_blend_actions_counter[blend_action_enc]
        inv_counter = Counter(dict(zip(self.range_ages, init_inventory)))

        # check invalid action
        if any(inv_counter[k] < blend_act[k] for k in blend_act):
            print('Found invalid blend action', init_inventory, blend_act, blend_action_enc)
            return init_inventory, -1e15, None # invalid action
        
        # subtract blending qty from inventory 
        new_inventory = inv_counter - blend_act
        new_inventory = [new_inventory[i] for i in self.range_ages]

        # selling brand products
        revenue = sum([i*j for i,j in zip(sales_act, self.brand_contrib)]) # brand contribution already considering cost corresponding to target ages
        added_cost = self.holding_cost*(sum(blend_act.elements()) - sum(i*j for i,j in zip(self.target_ages, sales_act)) ) # added cost due to blending different ages

        # disposal at supermarket
        disposal = new_inventory[-1] * self.supermarket_contrib[-1]
        new_inventory[-1] = 0   

        return new_inventory, revenue - added_cost + disposal, blend_action_enc




    def take_sale_action(self, init_inventory, sales_action_enc, mode='extensive'):
        """
        Extensive mode:
        Blending decision based on sales
        1. Use all stock of last age (which will be disposed if not used)
        2. Closest to target ages
        3. Highest overall decay risk
        
        Simple mode: select from oldest to youngest
        """
        assert mode in ['simple', 'extensive']

        # read sales action input
        sales_act = self.sales_actions[sales_action_enc]
        inv_counter = Counter(dict(zip(self.range_ages, init_inventory)))

        # select corresponding subset of blending actions
        blend_act_start_idx = sum(self.sales_blend_lens[:sales_action_enc])
        blend_act_end_idx = blend_act_start_idx + self.sales_blend_lens[sales_action_enc]
        range_act = zip(range(blend_act_start_idx, blend_act_end_idx), 
                            self.sales_blend_actions_counter[blend_act_start_idx : blend_act_end_idx] )
        
        blend_act = None
        blend_act_enc = None

        if mode == 'simple':
            # oldest stock
            temp = [i for i in self.range_ages for _ in range(int(inv_counter[i]))]
            blend_act = Counter(temp[-sum(sales_act):]) if sum(sales_act) > 0 else Counter({})
            for idx,(_,bl) in list(range_act)[::-1]:
                if bl == blend_act:
                    blend_act_enc = idx
                    break

        else: # mode == 'extensive'
            # use all stock of last age
            last_age_stock = init_inventory[-1]
            if last_age_stock >= sum(sales_act): 
                blend_act = Counter({self.num_ages: sum(sales_act)}) if sum(sales_act) > 0 else Counter({})  # only choose last age to sell
                for idx,(_,bl) in list(range_act)[::-1]:
                    if bl == blend_act:
                        blend_act_enc = idx
                        break
            else:            
                blend_options = [(idx, bl) for idx,(_,bl) in range_act if bl[self.num_ages] == last_age_stock] # sell all last age stocks

                min_sum = 0
                for idx,bl in blend_options:
                    if all(inv_counter[k] >= bl[k] for k in bl): 
                        # start iteration
                        if min_sum == 0: 
                            min_sum = sum(bl.elements())
                            blend_act = bl
                            blend_act_enc = idx

                        # same age sum
                        elif min_sum == sum(bl.elements()): 
                            # highest decay risk
                            # because the decay risk following the truncated discrete Weibull distribution, 
                            # in case of equal sum of age, the one with lower youngest age has higher total decay risk
                            if sorted(bl.elements()) < sorted(blend_act.elements()):
                                blend_act = bl
                                blend_act_enc = idx
                        
                        else: # higher age sum
                            break


        if blend_act is None or blend_act_enc is None: # invalid action
            print('Found invalid sale action', init_inventory, sales_act, sales_action_enc)
            return init_inventory, -1e15, None
        else:
            # subtract blending qty from inventory 
            new_inventory = inv_counter - blend_act
            new_inventory = [new_inventory[i] for i in self.range_ages]

            # selling brand products
            revenue = sum([i*j for i,j in zip(sales_act, self.brand_contrib)]) # brand contribution already considering cost corresponding to target ages
            added_cost = self.holding_cost*(sum(blend_act.elements()) - sum(i*j for i,j in zip(self.target_ages, sales_act)) ) # added cost due to blending different ages

            # disposal at supermarket
            disposal = new_inventory[-1] * self.supermarket_contrib[-1]
            new_inventory[-1] = 0  

            return new_inventory, revenue - added_cost + disposal, blend_act_enc  



    def take_decay_action(self, init_inventory):
        decay_qty = []
        new_inventory = copy.deepcopy(init_inventory)

        # random decay of stock of each age class
        for i in self.range_ages[:-1]: # do not consider decay of last-age
            random_number = np.random.uniform(0,1)

            # select corresponding decay quantity
            inv = new_inventory[i]
            if inv > 0:
                for j in range(int(inv+1)):
                    if random_number < self.decay_dict[inv][i-1, j] :
                        new_inventory[i] -= j
                        decay_qty.append(j)
                        break
            else:
                decay_qty.append(0)

        # profit contribution from selling off decay inventory
        decay = sum([i*j for i,j in zip(decay_qty, self.supermarket_contrib[:-1])])

        return new_inventory, decay, decay_qty



    ###############################################################
    # make transition, return next state, reward & done
    def step(self, 
             action_encoded,
             lod=None,
             mode='extensive',
             mask_preload=False):
        if self.done:
            reward = 0
            return {}, reward, self.done, {}

        action_encoded = int(action_encoded)

        # blend & sell    
        if self.action_type == 'sales' or lod == 0:
            new_inventory, profit, expanded_act = self.take_sale_action(self.obs, action_encoded, mode)
        else:
            new_inventory, profit, expanded_act = self.take_blend_action(self.obs, action_encoded)

        # decay during storing
        new_inventory, decay, decay_qty = self.take_decay_action(new_inventory)
      
        # update state
        if self.purchase_rule == 'fixed': 
            # purchase fixed amount = max inventory level
            purchase_qty = self.num_inv_levels-1
        else: 
            # purchase amount = S (order-up-to level) - current total inventory
            purchase_qty = max( min(self.purchase_level - sum(new_inventory[:-1]), self.num_inv_levels-1), 0)
        self.obs = np.array([purchase_qty] + list(new_inventory[:-1])).astype('float')
        
        # update mask
        if mask_preload and self.state_mask is not None:
            self.mask = self.state_mask[self.obs_dict_rev[tuple(self.obs)], :]
        else:
            self.mask = get_mask(self.sales_blend_actions_counter, self.sales_blend_lens, self.obs, self.range_ages, self.action_type)
        
        # update observation
        observation = {"action_mask": self.mask, 
                       "real_obs": np.array(self.obs)}

        # final reward
        reward = profit + decay - purchase_qty*self.purchase_cost

        self.iter += 1
        if self.iter >= self.time_limit:
            self.done = True


        return observation, reward, self.done, expanded_act



    # ###############################################################
    # def update_state(self, inventory_enc, mode='enc'):
    #     assert (mode == 'obs' and self.observation_space['real_obs'].contains(inventory_enc)) or \
    #                 (mode == 'enc' and inventory_enc in range(len(self.obs_dict)))

    #     new_inventory = self.obs_dict[inventory_enc] if mode == 'enc' else inventory_enc
    #     self.obs = np.array(new_inventory)
    #     self.mask = get_mask(self.sales_blend_actions_counter, self.sales_blend_lens, self.obs, self.range_ages, self.action_type)
    #     observation = {"action_mask": self.mask, 
    #                     "real_obs": np.array(self.obs)}
    #     return observation




#########################################################################################################
    # !!! ONLY FOR VALUE ITERATION #####################################
    # pre-calculate states and values to run multi-processing

    # define transition probability and related values
    def define_transition(self):
        decay_val = [] # decay profit including purchasing cost depended on total inventory level
        decay_prob_flatten = [] # decay probability
        decay_after_flatten = [] # final state after decay and purchasing
        decay_len = [] # number of available final states for each before decay state

        # transform cdf to pmf
        self.decay_dict_prob = dict()
        for inv_lv in self.decay_dict:
            tmp = copy.deepcopy(self.decay_dict[inv_lv])
            for i in range(1, inv_lv+1):
                tmp[:,i] = self.decay_dict[inv_lv][:,i] - self.decay_dict[inv_lv][:,i-1]
            self.decay_dict_prob.update({inv_lv: tmp})  

        # prev_state = after taking blending action ==> last age stock = 0 due to outdate => [num_ages-1 stock] 
        # next_state = before purchasing (~ final state) ==> [num_ages-1 stock] 
        all_obs = itertools.product(*[range(self.num_inv_levels) for _ in range(self.num_ages-1)]) # without the last ages stock      
        
        for prev_state in all_obs:
            poss_next_state = itertools.product(*[range(prev_state[i] + 1) for i in range(self.num_ages-1)])
            
            decay = 0
            l = 0
            for next_state in poss_next_state:
                decay_qty = [prev_state[i] - next_state[i] for i in range(self.num_ages-1)]

                # transition probability
                prob = np.product([self.decay_dict_prob[prev_state[i]][i, decay_qty[i]] if prev_state[i] > 0 else 1 \
                                                for i in range(self.num_ages-1)])
                decay_prob_flatten.append(prob)

                # final state
                if self.purchase_rule == 'fixed': 
                    # purchase fixed amount = max inventory
                    purchase_qty = self.num_inv_levels-1
                else: 
                    # purchase qty = S (order-up-to level) - current total inventory
                    purchase_qty = max( min(self.purchase_level - sum(next_state), self.num_inv_levels-1), 0)
                decay_after_flatten.append(self.obs_dict_rev[(purchase_qty, *next_state)]) 
                
                # reward
                decay_re = sum([i*j for i,j in zip(decay_qty, self.supermarket_contrib[:-1])])
                purchase_re = purchase_qty*self.purchase_cost
                decay += prob*(decay_re - purchase_re)

                # len
                l += 1

            decay_val.append(decay)
            decay_len.append(l)

            print(prev_state, 'done', purchase_re)
        
        # save data
        np.save("database/decay_val", decay_val)
        np.save("database/decay_len", decay_len)  
        np.save("database/decay_prob_flatten", decay_prob_flatten) 
        np.save("database/decay_after_flatten", decay_after_flatten) 

        return decay_val, decay_len, decay_prob_flatten, decay_after_flatten


    # get sales profit from each action
    def define_action_profit(self):
        action_profit = []
        for sales_act, blend_act in self.sales_blend_actions_counter:
            # selling brand products
            revenue = sum([i*j for i,j in zip(sales_act, self.brand_contrib)]) # brand contribution already considering cost corresponding to target ages
            added_cost = self.holding_cost*(sum(blend_act.elements()) - sum(i*j for i,j in zip(self.target_ages, sales_act)) ) # added cost due to blending different ages
            
            action_profit.append(revenue - added_cost) 

        action_profit = np.array(action_profit)
        np.save("database/action_profit", action_profit)
        return action_profit


    # get mask and profit after sales, blending and outdating for each prev state and action
    def define_state_mask(self):
        start = time.time()
        state_mask = []
        state_profit = []

        bounds = range(50000, 850000+1, 50000) # divide data into partial files to match with RAM storage
        bounds_idx = 0

        for i,inv in self.obs_dict.items():
            mask_temp = []
            val_temp = []

            for sales_act, blend_act in self.sales_blend_actions_counter:               
                if any(inv[k-1] < blend_act[k] for k in blend_act):
                    mask = 0                        # INVALID action: mask = 0
                    disposal = -1e5
                else: 
                    mask = 1                        # VALID action: mask = 1 
                    new_inventory = [inv[k-1] - blend_act[k] for k in self.range_ages]          # subtract blending qty from inventory 
                    disposal = new_inventory[-1] * self.supermarket_contrib[-1]                 # disposal at supermarket of outdating stock
                mask_temp.append(mask)
                val_temp.append(disposal)

            if i == bounds[bounds_idx]:
                np.save("database/state_mask_{}".format(bounds[bounds_idx]), np.array(state_mask))
                np.save("database/state_profit_{}".format(bounds[bounds_idx]), np.array(state_profit))
                print(bounds[bounds_idx], 'saved')
                state_mask = []
                state_profit = []
                bounds_idx += 1              

            val_temp = (np.array(val_temp) + self.action_profit)*np.array(mask_temp).tolist()
            state_mask.append(mask_temp)
            state_profit.append(val_temp)
            print(i, 'done')

        
        np.save("database/state_mask_{}".format(bounds[bounds_idx]), np.array(state_mask))
        np.save("database/state_profit_{}".format(bounds[bounds_idx]), np.array(state_profit))
        print('saving array done after', time.time()-start)
        return None, None # state_mask, state_profit


    # get state encode after blending and outdating for each prev state and action
    def define_state_next_state(self):
        start = time.time()
        state_next_state = []

        bounds = range(50000, 850000+1, 50000) # divide data into partial files to match with RAM storage
        bounds_idx = 0

        for i,inv in self.obs_dict.items():
            temp = []

            for j, (sales_act, blend_act) in enumerate(self.sales_blend_actions_counter):
                if any(inv[k-1] < blend_act[k] for k in blend_act): 
                    next_state_enc = -1
                else:                  
                    new_inventory = [inv[k-1] - blend_act[k] for k in self.range_ages]          # subtract blending qty from inventory 
                    new_inventory[-1] = 0
                    next_state_enc = self.obs_dict_rev[tuple(new_inventory)]
                temp.append(next_state_enc)

            if i == bounds[bounds_idx]:
                np.save("database/state_next_state_{}".format(bounds[bounds_idx]), np.array(state_next_state))
                print(bounds[bounds_idx], 'saved')
                state_next_state = []
                bounds_idx += 1  

            state_next_state.append(temp)
            print(i, 'done')

        np.save("database/state_next_state_{}".format(bounds[bounds_idx]), np.array(state_next_state))
        print('saving array done after', time.time()-start)

        return None # state_next_state



    # run all define functions and save the results in order to run value iteration
    def prepare_env(self):
        self.obs_dict = dict(enumerate(itertools.product(*[range(self.num_inv_levels) for _ in self.range_ages])))
        self.obs_dict_rev = dict(map(reversed, self.obs_dict.items()))


        start = time.time()
        transition = self.define_transition()
        print('transition done', time.time()-start)
        print(self.purchase_level)

        
        start = time.time()
        self.action_profit = self.define_action_profit()
        print('action profit done', time.time()-start)

        start = time.time() # run time: 28228 => actual: 2.5h
        self.state_mask, self.state_profit = self.define_state_mask()
        print('state mask done', time.time()-start)

        start = time.time()
        self.state_next_state = self.define_state_next_state()
        print('state next state done', time.time()-start)
        
        return None

    
    def load_data(self):
        self.obs_dict = dict(enumerate(itertools.product(*[range(self.num_inv_levels) for _ in self.range_ages])))
        self.obs_dict_rev = dict(map(reversed, self.obs_dict.items()))

        # m = np.load("database/state_mask.npy") 
        # self.state_mask = (m == 0).astype('int')
        
        return None
