# MPC-RL

Experiments with model-predictive control and reinforcement learning on the
[Push-T](https://github.com/real-stanford/diffusion_policy) Gymnasium environment.

The immediate goal is to understand and visualize value functions before
building a stronger controller or reinforcement-learning algorithm.

## Current Status

The project currently contains:

- A minimal Push-T environment example in `gym_env_skeleton.py`.
- NumPy Monte Carlo Q baselines in `value_function_baseline.py`, including a
  linear model and an RBF kernel-ridge model.
- Structured state-space exploration and a goal-centered value plot in
  `plot_goal_value.py`.
- A first model-free fitted Q-iteration experiment in
  `fitted_q_iteration.py`, inspired by the linked continuous FVI work.
- A generated example plot in `goal_value_heatmap.png`.

The estimators learn a continuous-action approximation of
$Q^\pi(s,a)$ from random-policy rollouts. It estimates $V(s)$ by sampling
candidate actions and taking the largest predicted Q-value:

$$
V(s) \approx \max_{a \in \mathcal{A}_{sampled}} Q^\pi(s,a).
$$

This is a diagnostic baseline, not yet an estimate of the optimal value
function $V^*(s)$.

The default RBF model uses random centers from the training data. The local
comparison script additionally supports goal-weighted centers: 75% are sampled
near the plotted state `[256, 400, 256, 256, pi/4]` and 25% remain globally
distributed. This focuses capacity on the goal slice without completely
discarding coverage elsewhere.

## Fitted Value Iteration Direction

The project that inspired the next stage is
[`milutter/value_iteration`](https://github.com/milutter/value_iteration),
which implements continuous and robust fitted value iteration. Its cFVI/rFVI
updates exploit known control-affine dynamics and analytic action updates.
Push-T uses contact-rich Pymunk dynamics through a black-box Gymnasium step,
so those analytic updates cannot be transferred directly.

`fitted_q_iteration.py` adapts the Bellman-iteration idea in a model-free way:

$$
y_k = r + \gamma (1-d)\max_{a'} Q_k(s',a')
$$

It uses the existing RBF Q model and samples continuous candidate actions for
the inner maximum. The current script is a first baseline; it uses random
replay data and should be evaluated with the held-out validation workflow
before increasing the number of iterations or moving to a learned policy.

## Environment

Push-T is registered by `gym-pusht` as:

```python
gym.make("gym_pusht/PushT-v0", obs_type="state")
```

For state observations:

```text
state = [pusher_x, pusher_y, object_x, object_y, object_angle]
```

The state has shape `(5,)`. Positions are in a roughly `512 x 512` workspace
and the angle is in radians.

The action space is:

```text
Box(0.0, 512.0, shape=(2,), dtype=float32)
```

An action is the desired 2D pusher position. It is not a force or velocity
command.

The environment defines the target object pose as:

```text
[256, 256, pi / 4]
```

## Setup

Python 3.12 is recommended. The current `gym-pusht` dependency chain is not
compatible with Python 3.14, and Push-T currently requires Pymunk 6.x APIs.
The dependency constraint `pymunk<7` is therefore intentional.

On Windows, install Python 3.12 if necessary:

```powershell
winget install --id Python.Python.3.12 --scope user
```

Create and activate the project environment:

```powershell
cd C:\Users\hanne\Workspace\mpc-rl
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The virtual environment is ignored by Git.

## Run The Examples

Run a single random Push-T episode:

```powershell
python gym_env_skeleton.py
```

Fit the basic Q estimator from random-policy episodes:

```powershell
python value_function_baseline.py
```

Run the initial fitted-Q iteration baseline:

```powershell
python fitted_q_iteration.py
```

Generate the goal-centered value heatmap:

```powershell
python plot_goal_value.py
```

Validate the random-policy estimator on fresh held-out rollouts:

```powershell
python validate_value_function.py
```

This reports Q/V MAE, RMSE, and correlation, and saves
`value_validation.png`. The validation uses prescribed first actions and
random continuation, so it evaluates the same $Q^\pi$ target used for
training.

The first compact validation run used 2,160 training transitions and 16
held-out states. It produced Q correlation `0.1627` and V correlation
`0.4126`, so the current lightweight RBF model should be treated as a
prototype rather than a reliable value estimator. The validation machinery is
working; the next experiment should improve coverage and tune the RBF model
before drawing conclusions from the heatmap.

The plotting script uses the larger RBF experiment configuration and saves
`goal_value_heatmap_rbf_long.png`. It uses 900 structured reset states, two
episodes per reset state, 150 steps per episode, 256 RBF centers, up to 50,000
fit samples, 1,024 candidate actions, and a 61 x 61 evaluation grid. The
earlier lightweight RBF result is retained in `goal_value_heatmap_rbf.png`,
and the linear result is retained in `goal_value_heatmap.png`.

The final four-seed random-policy run also produces
`goal_value_heatmap_rbf_final_random.png`, which includes a mean surface and
an across-seed standard-deviation panel. Its standalone mean-only reference
is `goal_value_heatmap_rbf_final_random_mean.png`, intended for direct
comparison with FQI/FVI plots.

The plotting script uses Gymnasium's `reset_to_state` option to deliberately
initialize episodes over a structured grid of pusher positions, object
positions, and object angles. This is more useful for a state-space
visualization than relying only on random episodes from the default initial
state distribution.

## Interpreting The Plot

`plot_goal_value.py` varies the object's x/y position around the target while
holding the pusher position and object angle fixed. The full state is five
dimensional, so this is only a 2D slice of the value function.

The heatmaps are expected to be imperfect. A random-policy Monte Carlo
dataset, even with an RBF kernel model, does not provide a reliable estimate
of $V^*(s)$. In particular, the value surface may not peak at the goal. That
is a useful diagnostic result, not evidence that the optimal controller has
been learned.

## Planned Work

Possible next steps, in roughly increasing complexity:

1. Improve the state features, including periodic encoding of the object angle.
2. Compare the learned values against short simulated rollouts from fixed
   states.
3. Tune the RBF bandwidth and center selection, or compare against local
    regression and a neural Q approximator.
4. Implement fitted Q-iteration or another off-policy value-learning method.
5. Add a policy that uses the estimated Q function to choose actions.
6. Connect the value estimator to an MPC planner.

## GitHub Notes

Before publishing, review generated artifacts and decide whether to commit
`goal_value_heatmap.png`. The virtual environment and Python bytecode are
already excluded by `.gitignore`.

No secrets or machine-specific paths should be added to the repository.