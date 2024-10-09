import sys
import gym
from gymnasium.spaces import Box, Discrete
import numpy as np
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from jqmcvi.base import dunn_fast
import math

def get_vector_score(df_feat_all, y_pred):
    dbs = -davies_bouldin_score(df_feat_all, y_pred)
    chs = calinski_harabasz_score(df_feat_all, y_pred)
    dunn = dunn_fast(df_feat_all, y_pred)
    sils = (silhouette_score(df_feat_all, y_pred) + 1) / 2
    scores = np.array([dbs, chs, dunn, sils])
    return scores

def obtain_score(df_feat_all, y_pred, list_eval, selected_features):
    scoreTotal = []
    for norm_score in list_eval:
        scoreTotal.append(norm_score.normalize(df_feat_all, y_pred)[1])
    if math.inf in scoreTotal or -math.inf in scoreTotal:
        raise ValueError
    else:
        return np.array(scoreTotal)


class MOFeatureSelectionEnv(gym.Env):

    def __init__(self, df_features, n_features, clustering_model, list_eval) -> None:
        self.observation_space = Box(0, 1, shape=(len(df_features.columns),), dtype=np.float32)
        self.action_space = Discrete(len(df_features.columns))
        self.list_eval = list_eval
        self.list_eval_reset = list_eval
        self.reward_space = Box(low=-1000, high=1000, shape=(len(self.list_eval),))
        self.reward_dim = len(self.list_eval)
        self.all_features = df_features.copy()
        self.current_state = np.zeros(len(self.all_features.columns))
        self.n_features = n_features
        self.clustering_model = clustering_model
        self.previous_vector_score = np.array([0 for x in range(len(list_eval))])
        self.steps_taken = 0
        self.y_pred = None

    
    def _get_obs(self) -> np.array:
        return self.current_state
    
    def _get_features_selected(self):
        # return [feature for i, feature in enumerate(self.all_features.columns) if self.current_state[i]]
        return self.features_selected

    def _get_info(self) -> dict:
        return {
            'legal_actions': [action for action in range(self.action_space.n) if not self.current_state[action]],
            'y_pred': self.y_pred,
            'features_selected': self._get_features_selected(),
        }
    
    def reset(self, seed=None, options=None):
        self.features_selected = []
        self.current_state = np.zeros(len(self.all_features.columns))
        self.list_eval = self.list_eval_reset
        self.previous_vector_score = np.array([0 for x in range(len(self.list_eval))])
        self.scores_received = []
        observation = self._get_obs()
        self.steps_taken = 0
        info = self._get_info()
        return observation, info
    
    def _get_reward(self):
        selected_features = [feature for i, feature in enumerate(self.all_features.columns) if self.current_state[i]]
        y_pred = self.clustering_model.fit_predict(self.all_features[selected_features])
        self.y_pred = y_pred
        try:
            vector_score = obtain_score(self.all_features, y_pred, self.list_eval, selected_features)
            self.scores_received.append(vector_score)
            # vector_score = get_vector_score(self.all_features, y_pred)
            gain = vector_score - self.previous_vector_score
            self.previous_vector_score = vector_score
            return gain
        except:
            self.scores_received.append(np.nan)
            return self.previous_vector_score
    
    def step(self, action):
        self.current_state[action] = 1
        self.features_selected.append(self.all_features.columns[action])

        # Refactor later
        self.features_selected = list(set(self.features_selected))

        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()

        # This is not good because it must stop when performance is stable.
        if len(self.current_state[self.current_state == 1]) < self.n_features and self.steps_taken < self.n_features:
            terminated = False
        else:
            terminated = True

        self.steps_taken += 1

        return observation, reward, terminated, False, info
    
    def render(self):
        pass

    def _render_frame(self):
        pass
    
    def close(self):
        pass
