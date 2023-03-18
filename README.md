# reinforcement learning based on CQL
## installation
```
   $ git clone https://github.com/Yufei-Lei/HW3.git
   $ cd d3rlpy
   $ pip install Cython numpy # if you have not installed them.
   $ pip install -e .
```

## get started
run the project.py file to train the model and generate evaluation scores
```
   cd d3rlpy
   python project.py
```
run the plot.py file and input the path in step 1 to see the result
```
   python plot.py
```
result sample 

![44383e44ee53a8c90f8ae9013cdfd85](https://user-images.githubusercontent.com/87921304/144772461-3c43b796-64d5-4797-be71-55a48e8e82ea.png)

## True Q 
True Q is the ground truth label when we use neural network to fit Q values.

![6b3225baedb6ae7dd4e26f305505f64](https://user-images.githubusercontent.com/87921304/145109851-58c033b0-a287-4a94-bb80-447bccb3c70e.png)

```
   def true_q(algo: AlgoProtocol, episodes: List[Episode]) -> float:
        for episode in episodes:
            for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
                


                # estimate values for next observations
                next_actions = algo.predict([batch.next_observations[0]])
                next_values = algo.predict_value(
                    [batch.next_observations[0]], next_actions
                )

                # calculate true q
                mask = (1.0 - np.asarray(batch.terminals)).reshape(-1)
                rewards = np.asarray(batch.next_rewards).reshape(-1)
                if algo.reward_scaler:
                    rewards = algo.reward_scaler.transform_numpy(rewards)
                y = rewards[0] + algo.gamma * cast(np.ndarray, next_values) * mask


        return float(np.mean(y))
```
## citation 
https://github.com/takuseno/d3rlpy.git
