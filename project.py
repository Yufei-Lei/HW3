import d3rlpy

from d3rlpy.datasets import get_atari
from d3rlpy.datasets import get_cartpole
dataset, env = d3rlpy.datasets.get_pybullet('hopper-bullet-mixed-v0')


from sklearn.model_selection import train_test_split

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)
from d3rlpy.algos import CQL
cql = CQL(use_gpu=True)
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import continuous_action_diff_scorer
from d3rlpy.metrics.scorer import value_estimation_std_scorer
from d3rlpy.metrics.scorer import true_q
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
# set environment in scorer function
#evaluate_scorer = evaluate_on_environment(env)

# evaluate algorithm on the environment
#rewards = evaluate_scorer(dqn)

cql.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs=100,
            scorers={
                'environment': evaluate_on_environment(env),
                'true_q':true_q,
                'estimate_q': initial_state_value_estimation_scorer
            })

from d3rlpy.ope import FQE

# off-policy evaluation algorithm

fqe = FQE(algo=cql,
          n_epochs=200,
          q_func_factory='qr',
          learning_rate=1e-4,
          use_gpu=True,
          encoder_params={'hidden_units': [1024, 1024, 1024, 1024]})

# metrics to evaluate with
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer
from d3rlpy.metrics.scorer import true_q
# train estimators to evaluate the trained policy

fqe.fit(dataset.episodes,
        eval_episodes=dataset.episodes,
        n_epochs= 100,
        scorers={
           'init_value': initial_state_value_estimation_scorer,
           'soft_opc': soft_opc_scorer(return_threshold=600),
            'True_q':true_q


        })
