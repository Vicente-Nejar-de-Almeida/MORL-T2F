import math
import gym
import numpy as np
from early_stopping.constants import IMPROVE, SAFE


def obtain_score(df_feat_all, y_pred, normalized_scorers):
    score_total = []
    real_scores = {}

    for normalized_score_function in normalized_scorers:
        real_score, norm_score = normalized_score_function.get_normalized_value(df_feat_all, y_pred)
        
        score_total.append(norm_score)
        '''
        if normalized_score_function.maximize:
            score_total.append(norm_score)
        else:
            score_total.append(-norm_score)
        '''
        real_scores[normalized_score_function.name] = real_score
    
    score = np.mean(score_total)
    return score, real_scores


class NewFeatureSelectionEnvironment(gym.Env):

    def __init__(self, df_features, n_features, clustering_model, early_stopping, normalized_scorers) -> None:
        self.observation_space = gym.spaces.Box(0, 1, shape=(len(df_features.columns),), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(df_features.columns))
        self.all_features = df_features.copy()
        self.current_state = np.zeros(len(self.all_features.columns))
        self.n_features = n_features
        self.clustering_model = clustering_model
        self.early_stopping = early_stopping
        self.normalized_scores = normalized_scorers

        self.previous_score = 0
        self.real_scores = {}
        self.best_features = None
    
    def _get_obs(self) -> np.array:
        return self.current_state

    def _get_info(self) -> dict:
        return {
            'legal_actions': [action for action in range(self.action_space.n) if not self.current_state[action]],
            'composite_score': self.previous_score,
            'real_scores': self.real_scores,
        }
    
    def reset(self):
        self.current_state = np.zeros(len(self.all_features.columns))
        self.early_stopping.reset()
        observation = self._get_obs()
        return observation
    
    def _get_reward(self, action):
        selected_features = self.all_features.iloc[:, self.current_state.astype(bool)]
        y_pred = self.clustering_model.fit_predict(selected_features)
        
        try:
            """
            Test to see if this works well, otherwise we have two current options:
            (1) Use only the features selected by RL to compute metrics
            (2) Use raw time series to compute metrics
            """
            score, self.real_scores = obtain_score(self.all_features, y_pred, self.normalized_scores)
        except ValueError:
            # happens when all labels have same value, thus no real "clustering" has occurred
            score = -1
            self.real_scores = None

        gain = score - self.previous_score
        self.previous_score = score

        return gain, selected_features, score
    
    def step(self, action):
        print(f'Action: {action}')
        self.current_state[action] = 1
        reward, selected_features, score = self._get_reward(action)
        observation = self._get_obs()
        info = self._get_info()

        check_continuation = self.early_stopping.update_metric(score)
        if check_continuation == IMPROVE:
            self.best_features = selected_features
            done = False
        elif check_continuation == SAFE:
            done = False
        else:
            done = True
        
        if 0 not in observation:
            done = True

        return observation, reward, done, info
    
    def render(self):
        pass

    def _render_frame(self):
        pass
    
    def close(self):
        pass

    def action_masks(self):
        mask = [False if feat else True for feat in self._get_obs()]
        return mask
