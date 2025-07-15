from gymnasium.envs.registration import register


register(
    id="gym_so100/SO100TouchCube-v0",
    entry_point="gym_so100.env:SO100Env",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "so100_pixels_agent_pos", "task": "so100_touch_cube"},
)

register(
    id="gym_so100/SO100TouchCubeSparse-v0",
    entry_point="gym_so100.env:SO100Env",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "so100_pixels_agent_pos", "task": "so100_touch_cube_sparse"},
)

register(
    id="gym_so100/SO100CubeToBin-v0",
    entry_point="gym_so100.env:SO100Env",
    max_episode_steps=700,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "so100_pixels_agent_pos", "task": "so100_cube_to_bin"},
)