import numpy as np
import pandas as pd
import json, ast, os 
import matplotlib.pyplot as plt
import seaborn as sns

fdir = "final results"

# Optimal policy analysis #####################################################################
fname = os.path.join(fdir,"value iteration","results.csv")
data = pd.read_csv(fname, index_col=0)

f = lambda x: ast.literal_eval(x.replace(' ', ', '))
invs = np.array(list(map(f, data['Current inventory'])))
invs_sum = np.sum(invs, axis=1)
invs_sumage = np.sum(invs * np.array(range(1,8)), axis=1)

sales = np.array(list(map(ast.literal_eval, data['Sales qty'])))

sns.heatmap(pd.crosstab(invs_sum[10:].astype('int'), sales[10:,0]))
plt.xlabel('Sales quantity of the younger product')
plt.ylabel('Total inventory level on hand')
plt.savefig('plots/underfulfillment.png')
plt.show()

sns.heatmap(pd.crosstab(invs_sumage[10:].astype('int'), sales[10:,0]))
plt.xlabel('Sales quantity of the younger product')
plt.ylabel('Total sum of age of the current inventory')
plt.savefig('plots/underfulfillment_sumage.png')
plt.show()


blending = np.array(list(map(ast.literal_eval, data['Blending'])))
min_age = []
max_age = []
for b in blending:
    check_min = True
    check_max = True
    i = 1
    while (check_min or check_max) and (i<8):
        if check_min and b[i-1] > 0 :
            min_age.append(i)
            check_min = False
        if check_max and b[-i] > 0:
            max_age.append(8-i)
            check_max = False
        if i == 7 and (check_max or check_min):
            min_age.append(0)
            max_age.append(0)
        i+=1
spread = np.array(max_age) - np.array(min_age)
sns.histplot(spread)
plt.xlabel('Age spread of the blend')
plt.ylabel('Count')
plt.savefig('plots/spread.png')
plt.show()


# GAS vs DQN vs Value iter ###############################################################
# training return
gas_df = []
foldername = os.path.join(fdir, "GAS+DQN main results", "GAS")
for i in range(5):
    with open(os.path.join(foldername, "test{}.json".format(i))) as f:
        data = json.load(f)['returns']
        gas_df.append([0]+list(list(zip(*data))[1]))

gas_smooth = []
for i in range(5):
    gas_smooth.append([np.mean(gas_df[i][:j][-20:]) for j in range(len(gas_df[i]))])

temp = [(np.mean(i), np.max(i), np.min(i)) for i in zip(*gas_smooth)]
gas_mean, gas_max, gas_min = zip(*temp)


full_df = []
foldername = os.path.join(fdir, "GAS+DQN main results", "DQN")
for i in range(5):
    with open(os.path.join(foldername, "test_full_{}.json".format(i))) as f:
        data = json.load(f)['returns']
        full_df.append([0]+list(list(zip(*data))[1]))

full_smooth = []
for i in range(5):
    full_smooth.append([np.mean(full_df[i][:j][-20:]) for j in range(len(full_df[i]))])

temp = [(np.mean(i), np.max(i), np.min(i)) for i in zip(*full_smooth)]
full_mean, full_max, full_min = zip(*temp)


x = list(range(0,100000, 1000))
plt.plot(x, gas_mean, color="orange", lw=1, label='GAS-DQN') 
plt.fill_between(x, gas_min, gas_max,
                 color='orange',       # The outline color
                 alpha=0.4)          # Transparency of the 
plt.plot(x, full_mean, color="green", lw=1, label='DQN') 
plt.fill_between(x, full_min, full_max,
                 color='green',       # The outline color
                 alpha=0.4)          # Transparency of the 
plt.xlabel('Epoch')
plt.ylabel('Smoothed avg. return')
plt.legend()
plt.savefig('plots/training_return.png')
plt.show()


############################
# running time
x = ['GAS-DQN (5 runs)', 'DQN (5 runs)', 'Value iteration']
y = [8130, 6375, 31526]
sns.barplot(data=pd.DataFrame({'Algorithm':x, 'Running time':y}), 
            x='Algorithm', y='Running time', palette=['orange','green','blue'])
plt.ylabel('Total running time (s)')
plt.xlabel('')
plt.savefig('plots/running_time.png')
plt.show()


######################################
# simulation result
gas_df = pd.DataFrame([], columns=['Step', 'Current inventory', 'Action', 'Sales qty', 
                                'Blending','Reward', 'Next inventory', 'Training'])
foldername = os.path.join(fdir, "GAS+DQN main results", "GAS")
for i in range(5):
    fname = os.path.join(foldername, "results_{}.csv".format(i))
    data = pd.read_csv(fname, index_col=0)
    data['Training'] = i
    gas_df = gas_df.append(data)
x_gas = gas_df.groupby('Training')[['Reward']].aggregate('mean').to_numpy().flatten()

dqn_df = pd.DataFrame([], columns=['Step', 'Current inventory', 'Action', 'Sales qty', 
                                'Blending','Reward', 'Next inventory', 'Training'])
foldername = os.path.join(fdir, "GAS+DQN main results", "DQN")
for i in range(5):
    fname = os.path.join(foldername, "results_full_model_{}.csv".format(i))
    data = pd.read_csv(fname, index_col=0)
    data['Training'] = i
    dqn_df = dqn_df.append(data)
x_dqn = dqn_df.groupby('Training')[['Reward']].aggregate('mean').to_numpy().flatten()

fname = os.path.join(fdir,"value iteration","results.csv")
data = pd.read_csv(fname, index_col=0)
line = np.mean(data.loc[:100000, 'Reward'])

sns.boxplot(x='variable', y='value', 
            data=pd.melt(pd.DataFrame({'GAS-DQN (5 runs)':x_gas,'DQN (5 runs)':x_dqn})),
            palette=['orange','green'])
plt.xlabel('')
plt.ylabel('Expected profit per period')
left, right = plt.xlim()
plt.hlines(y=line,xmin=left,xmax=right, color="blue", linestyles='dashed', label='Value iteration')
plt.legend(loc=3)
plt.savefig('plots/simulation_result.png')
plt.show()


# GAS policy analysis ###########################################################################
f = lambda x: ast.literal_eval(x.replace(' ', ', '))
gas_df['Current inventory'] = gas_df['Current inventory'].apply(f)
gas_df['Current total inventory'] = gas_df['Current inventory'].apply(sum)
gas_df.groupby('Training')[['Current total inventory']].aggregate('mean').to_numpy().flatten()

gas_df.groupby('Training')[['Reward']].aggregate('mean').to_numpy().flatten()

gas_df['Blending'] = gas_df['Blending'].apply(ast.literal_eval)
min_age = []
max_age = []
for b in gas_df['Blending'].to_numpy():
    check_min = True
    check_max = True
    i = 1
    while (check_min or check_max) and (i<8):
        if check_min and b[i-1] > 0 :
            min_age.append(i)
            check_min = False
        if check_max and b[-i] > 0:
            max_age.append(8-i)
            check_max = False
        if i == 7 and (check_max or check_min):
            min_age.append(0)
            max_age.append(0)
        i+=1
spread = np.array(max_age) - np.array(min_age)
gas_df['Spread'] = spread
gas_df.groupby('Training')[['Spread']].aggregate('mean').to_numpy().flatten()

outdate = gas_df['Current inventory'].apply(lambda x: x[-1]).to_numpy() - gas_df['Blending'].apply(lambda x: x[-1]).to_numpy()

gas_df['Sales qty'] = gas_df['Sales qty'].apply(ast.literal_eval)
gas_df['Sales2']=gas_df['Sales qty'].apply(lambda x: x[-1]).to_numpy()
gas_df.groupby('Training')[['Sales2']].aggregate('mean').to_numpy().flatten()
gas_df['Sales1']=gas_df['Sales qty'].apply(lambda x: x[0]).to_numpy()
gas_df.groupby('Training')[['Sales1']].aggregate('mean').to_numpy().flatten()

gas_df.loc[gas_df['Sales1']<3,:].groupby('Training')[['Sales1']].aggregate('count').to_numpy().flatten()





# Simple mapping ##############################################################################
# simulation result
gas_df = pd.DataFrame([], columns=['Step', 'Current inventory', 'Action', 'Sales qty', 
                                'Blending','Reward', 'Next inventory', 'Training'])
foldername = os.path.join(fdir, "GAS+DQN main results", "GAS")
for i in range(5):
    fname = os.path.join(foldername, "results_{}.csv".format(i))
    data = pd.read_csv(fname, index_col=0)
    data['Training'] = i
    gas_df = gas_df.append(data)
x_gas = gas_df.groupby('Training')[['Reward']].aggregate('mean').to_numpy().flatten()

x_gas2_df = pd.DataFrame([], columns=['Step', 'Current inventory', 'Action', 'Sales qty', 
                                'Blending','Reward', 'Next inventory', 'Training'])
foldername = os.path.join(fdir, "simple mapping")
for i in range(5):
    fname = os.path.join(foldername, "results_{}.csv".format(i))
    data = pd.read_csv(fname, index_col=0)
    data['Training'] = i
    x_gas2_df = x_gas2_df.append(data)
x_gas2 = x_gas2_df.groupby('Training')[['Reward']].aggregate('mean').to_numpy().flatten()

fname = os.path.join(fdir,"value iteration","results.csv")
data = pd.read_csv(fname, index_col=0)
line = np.mean(data.loc[:100000, 'Reward'])

sns.boxplot(x='variable', y='value', 
            data=pd.melt(pd.DataFrame({'Mapping function 1':x_gas,'Mapping function 2':x_gas2})),
            palette=['orange','green'])
plt.xlabel('')
plt.ylabel('Expected profit per period')
plt.ylim((1500,2900))
left, right = plt.xlim()
plt.hlines(y=line,xmin=left,xmax=right, color="blue", linestyles='dashed', label='Value iteration')
plt.legend(loc=3)
plt.savefig('plots/simple_result.png')
plt.show()


##################################################
# training return
gas_df = []
foldername = os.path.join(fdir, "GAS+DQN main results", "GAS")
for i in range(5):
    with open(os.path.join(foldername, "test{}.json".format(i))) as f:
        data = json.load(f)['returns']
        gas_df.append([0]+list(list(zip(*data))[1]))

gas_smooth = []
for i in range(5):
    gas_smooth.append([np.mean(gas_df[i][:j][-10:]) for j in range(len(gas_df[i]))])

temp = [(np.mean(i), np.max(i), np.min(i)) for i in zip(*gas_smooth)]
gas_mean, gas_max, gas_min = zip(*temp)


simple_df = []
foldername = os.path.join(fdir, "simple mapping")
for i in range(5):
    with open(os.path.join(foldername, "test{}.json".format(i))) as f:
        data = json.load(f)['returns']
        simple_df.append([0]+list(list(zip(*data))[1]))

simple_smooth = []
for i in range(5):
    simple_smooth.append([np.mean(simple_df[i][:j][-10:]) for j in range(len(simple_df[i]))])

temp = [(np.mean(i), np.max(i), np.min(i)) for i in zip(*simple_smooth)]
simple_mean, simple_max, simple_min = zip(*temp)

x = list(range(0,100000, 1000))
# plt.plot(x, gas_mean, color="orange", lw=1.5, label='Mapping function 1') 
plt.fill_between(x, gas_min, gas_max,
                 color='orange',       # The outline color
                 alpha=0.6,
                 label='Range of smoothed episode return \nwith mapping function 1 from 5 runs')          # Transparency 
# plt.plot(x, simple_mean, color="green", lw=1.5, label='Mapping function 2') 
plt.fill_between(x, simple_min, simple_max,
                 color='green',       # The outline color
                 alpha=0.6,
                 label='Range of smoothed episode return \nwith mapping function 2 from 5 runs')          # Transparency
bounds = plt.ylim()
plt.vlines(x=20000, ymin=bounds[0], ymax=bounds[1], color="blue", linestyles='dashed', label='End of exploration in \nrestricted sub-space $A_0$')
plt.xlabel('Period')
plt.ylabel('Smoothed episode return')
plt.legend(loc=4)
plt.savefig('plots/simple_return_with_range.png')
plt.show()



# scalability #############################################################################
# size of problem
df = pd.DataFrame(
    [["inv=7", "|S|", 7**7],
     ["inv=7", "|A|", 3456], 
     ["inv=7", "Running time", 8130],
     ["inv=10", "|S|", 10**7],
     ["inv=10", "|A|", 15000], 
     ["inv=10", "Running time", 31033]],
     columns=['inv_max', 'parameter', 'value']
)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle('Comparison of two settings')
# State
sns.barplot(ax=axes[0], data=df.loc[df['parameter']=='|S|',], x='inv_max', y='value')
axes[0].set_title("Size of state space")

# Action
sns.barplot(ax=axes[1], data=df.loc[df['parameter']=='|A|',], x='inv_max', y='value')
axes[1].set_title("Size of action space")

# Running time
sns.barplot(ax=axes[2], data=df.loc[df['parameter']=='Running time',], x='inv_max', y='value')
axes[2].set_title("Running time")


########################################################
# simulation result
gas_10 = pd.DataFrame([], columns=['Step', 'Current inventory', 'Action', 'Sales qty', 
                                'Blending','Reward', 'Next inventory', 'Training'])
foldername = os.path.join(fdir, "num_age=10")
for i in range(5):
    fname = os.path.join(foldername, "results_{}.csv".format(i))
    data = pd.read_csv(fname, index_col=0)
    data['Training'] = i
    gas_10 = gas_10.append(data)
x_gas = gas_10.groupby('Training')[['Reward']].aggregate('mean').to_numpy().flatten()

fname = os.path.join(fdir,"num_age=10","greedy_results.csv")
data = pd.read_csv(fname, index_col=0)
line = np.mean(data.loc[:10000, 'Reward'])

sns.barplot(data=pd.DataFrame({'Training':range(1,6),'Value':x_gas}), x='Training', y='Value', color='orange',label='GAS-DQN')
plt.ylabel('Expected profit per period')
plt.xlabel('Runs')
left, right = plt.xlim()
plt.hlines(y=line,xmin=left,xmax=right, color="red", linestyles='dashed', 
            label='Greedy heuristic')
plt.legend(loc=3)
plt.savefig('plots/scale_result.png')
plt.show()


# profit ratio changed ##########################################################################
# optimal policy analysis
fname = os.path.join(fdir,"profits=500,600","results.csv")
data = pd.read_csv(fname, index_col=0)

f = lambda x: ast.literal_eval(x.replace(' ', ', '))
invs = np.array(list(map(f, data['Current inventory'])))
invs_sum = np.sum(invs, axis=1)
invs_sumage = np.sum(invs * np.array(range(1,8)), axis=1)

sales = np.array(list(map(ast.literal_eval, data['Sales qty'])))

sns.heatmap(pd.crosstab(invs_sum[10:].astype('int'), sales[10:,0]))
plt.xlabel('Sales quantity of the younger product')
plt.ylabel('Total inventory level on hand')
# plt.savefig('underfulfillment.png')
plt.show()

sns.heatmap(pd.crosstab(invs_sumage[10:].astype('int'), sales[10:,0]))
plt.xlabel('Sales quantity of the younger product')
plt.ylabel('Total sum of age of the current inventory')
# plt.savefig('underfulfillment_sumage.png')
plt.show()


#######################################################
# simulation result
gas_profits = pd.DataFrame([], columns=['Step', 'Current inventory', 'Action', 'Sales qty', 
                                'Blending','Reward', 'Next inventory', 'Training'])
foldername = os.path.join(fdir, "profits=500,600")
for i in range(5):
    fname = os.path.join(foldername, "results_{}.csv".format(i))
    data = pd.read_csv(fname, index_col=0)
    data['Training'] = i
    gas_profits = gas_profits.append(data)
x_gas = gas_profits.groupby('Training')[['Reward']].aggregate('mean').to_numpy().flatten()

fname = os.path.join(fdir,"profits=500,600","results.csv")
data = pd.read_csv(fname, index_col=0)
line = np.mean(data.loc[:10000, 'Reward'])


sns.barplot(data=pd.DataFrame({'Training':range(1,6),'Value':x_gas}), x='Training', y='Value', color='orange',label='GAS-DQN')
plt.xlabel('Runs')
plt.ylabel('Expected profit per period')
left, right = plt.xlim()
plt.hlines(y=line,xmin=left,xmax=right, color="blue", linestyles='dashed', 
            label='Value iteration')
plt.legend(loc=3)
plt.savefig('plots/profits_results.png')
plt.show()