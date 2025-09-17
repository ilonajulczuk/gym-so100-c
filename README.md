# gym-so100-c

Installation

```
conda create -y -n so100_310 python=3.10 && conda activate so100_310
conda install pytorch
pip install stable-baselines3
pip install poetry
poetry install --all-extras
```

## Simulation Environment and Task design

This repo is using the gym interface. 

I wanted to create a simulation env with a good physics engine and use it with a variety of methods. I decided to structure it as a `gym` and to use MuJoCo as a physics engine. Because of that adapting `gym-aloha` (gym environment of a similar type of robot using MuJoCo) was a great fit.

###Gym interface

OpenAI Gym (and its maintained fork [Gymnasium](https://gymnasium.farama.org/)) defines a standard interface for reinforcement learning environments.  
The two core methods every Gym env implements are:

- `reset()` → returns the initial observation for a new episode.
- `step(action)` → applies an action, advances the simulation by one step, and returns:

```python
  obs, reward, terminated, truncated, info
```

In addition, environments define:

* **Observation space** — what the agent sees (e.g., joint angles, object positions).
* **Action space** — what the agent can control (e.g., target joint positions, gripper commands).
* **Reward function** — the numeric feedback signal for learning.

This standardization means the **same environment** can be used with:

* RL libraries like Stable-Baselines3 (`SAC`, `PPO`, `HER`…)
* Imitation learning libraries like `imitation`
* Custom training loops or evaluation pipelines

My `gym-so100-c` environment follows this interface closely, which makes it easy to switch between algorithms, compare results, and plug into tools like `lerobot`.


### Background: ALOHA & gym-aloha

ALOHA (A Low-cost Open Hardware Arm) is a dual-arm teleoperation platform used in the ACT paper and follow-up work.  
The open-source [gym-aloha](https://github.com/huggingface/gym-aloha) package replicates this setup in MuJoCo and provides ready-to-use Gym environments for imitation and reinforcement learning.

**gym-aloha structure:**
- **Environment core:** [`env.py`](https://github.com/huggingface/gym-aloha/blob/main/gym_aloha/env.py) — loads the MuJoCo scene, handles `reset`/`step`, defines obs/action spaces, and integrates teleop/recording.
- **Task layers:**
  - **Joint-space tasks:** [`tasks/sim.py`](https://github.com/huggingface/gym-aloha/blob/main/gym_aloha/tasks/sim.py) — actions are target joint positions written to `data.ctrl`; MuJoCo’s position actuators move joints to match.
  - **End-effector (mocap) tasks:** [`tasks/sim_end_effector.py`](https://github.com/huggingface/gym-aloha/blob/main/gym_aloha/tasks/sim_end_effector.py) — actions are desired gripper poses applied to a mocap body; MuJoCo’s constraint solver adjusts joints to match.

| Aspect | Joint-space control | End-effector (mocap) control |
|---|---|---|
| **Action** | Target joint positions → `data.ctrl` | Target gripper pose (position + orientation) via mocap |
| **Control loop** | Actuators drive joints toward commanded positions | Constraints drive joints to match mocap pose |
| **Abstraction** | Low-level, robot-specific | Higher-level, task-centric |
| **Kinematics** | Direct mapping; no IK | Implicit IK via MuJoCo constraints |
| **Policy learning** | Exposes full dynamics | Operates in Cartesian space |
| **Real-world transfer** | Straightforward if HW supports pos. control | Needs IK/operational-space control on robot |

---

### Tasks in ALOHA

- **Insertion** — bimanual peg-in-hole task.
- **Cube transfer** — pass a cube from one arm to the other.

Each ALOHA task includes a **scripted policy** using **inverse kinematics (IK) + noise** to generate synthetic demonstrations.  
This is great for imitation learning because it can quickly produce large, diverse datasets without human teleop.

---

### My adaptation: gym-so100-c

I adapted gym-aloha for a **single SO101 arm** (5-DoF + gripper) to match my hardware target.

**Changes:**
- Replaced dual 6-DoF arms with one SO101.
- Removed dual-arm logic and simplified obs/action spaces.
- Ported only the **joint-space** control mode (clearer transfer to my target hardware).
- Added an **experimental end-effector teleop scene** (mocap-based), though not yet used for training.

**Task implemented:**
- **Bin-a-cube:**  
  - Cube starts at a random position on the table.  
  - Goal is to place it inside a fixed bin.  
  - Sparse reward for success, with optional shaping for approach, grasp, and alignment.

**Demonstrations:**
- Collected via **teleoperation** (keyboard or 8BitDo controller).
- Learned about IK + noise scripted demos from the [`gym_so100`](https://github.com/xuaner233/gym-so100/tree/main/gym_so100) repo — a nifty approach I haven’t implemented yet but plan to.

---

### MuJoCo scene

The scene is defined in [`assets/so100_transfer_cube.xml`](https://github.com/ilonajulczuk/gym-so100-c/blob/main/gym_so100/assets/so100_transfer_cube.xml):

- **Robot:** SO101 single arm (5-DoF + gripper).
- **Actuators:** position actuators, one per joint; actions = target joint positions.
- **Workspace:** table, free-moving cube, goal bin.
- **Sites:** for gripper tip, cube, and bin goal — used for shaping and success checks.
- **Cameras:** optional top/front views for logging or vision policies.


### Reward shaping

* Distance-to-goal shaping from cube to bin.
* Sparse success reward + intermediate bonuses for approach/grasp.
* Penalties for unstable motions or constraint violations.


### Goal-based environment support

In addition to the standard Gym API, `gym-so100-c` also implements the [GoalEnv interface](https://gymnasium.farama.org/api/env/#goal-based-environments), which is commonly used with algorithms like **Hindsight Experience Replay (HER)**.

Implementation is in the `env.py`.


A GoalEnv returns observations in a dictionary format with three keys:
- `observation` — the regular observation (e.g., joint positions, velocities, object positions)
- `achieved_goal` — the goal currently achieved in the environment (e.g., cube position)
- `desired_goal` — the target goal for this episode (e.g., bin location)

Example `reset()` output:
```python
{
    "observation": np.array([...]),
    "achieved_goal": np.array([...]),
    "desired_goal": np.array([...]),
}
```

This structure allows algorithms to:

* Compute rewards based on the distance between `achieved_goal` and `desired_goal`.
* Relabel experiences after the fact (as in HER), which can massively improve sample efficiency for sparse-reward tasks.

I use this GoalEnv setup for **SAC + HER** experiments in the bin-a-cube task, where success is naturally expressed as “cube inside bin.”
It makes it easy to switch between plain SAC and HER without rewriting the environment logic.

## Training integrations & scripts