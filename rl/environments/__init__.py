from gymnasium.envs.registration import register

from rl.environments.multi_objective_feature_selection_env import MOFeatureSelectionEnv

register(
    id="mo-feature-selection-env-v0",
    entry_point="rl.environments.multi_objective_feature_selection_env:MOFeatureSelectionEnv",
    max_episode_steps=100,
)
