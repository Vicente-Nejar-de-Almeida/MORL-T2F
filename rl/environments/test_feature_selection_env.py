import pandas as pd
import gym
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler

from early_stopping.constants import IMPROVE, SAFE


def obtain_score(df_feat_all, y_pred, normalized_scorers):
    score_total = []
    real_scores = {}
    normalized_scores = {}

    for normalized_score_function in normalized_scorers:
        real_score, norm_score = normalized_score_function.normalize(df_feat_all, y_pred)
        
        # score_total.append(norm_score)
        
        if normalized_score_function.maximize:
            score_total.append(norm_score)
        else:
            score_total.append(-norm_score)
        
        real_scores[normalized_score_function.name] = real_score
        normalized_scores[normalized_score_function.name] = norm_score
    
    score = np.mean(score_total)
    '''
    if pd.isna(score):
        score = 1
    '''
    return score, real_scores, normalized_scores


class TestFeatureSelectionEnv(gym.Env):

    def __init__(self, df_features, n_features, clustering_model, early_stopping, normalized_scorers) -> None:
        # self.observation_space = gym.spaces.Box(0, 1, shape=(len(df_features.columns),), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(
            {
                f'f{i}': gym.spaces.Box(0, 1, shape=(len(df_features),), dtype=float) for i in range(n_features)
            }
        )
        self.action_space = gym.spaces.Discrete(2)
        self.all_features = df_features.copy()

        self.normalized_features = df_features.copy()
        scaler = MinMaxScaler()
        self.normalized_features[self.normalized_features.columns] = scaler.fit_transform(self.normalized_features)

        self.n_features = n_features
        self.clustering_model = clustering_model
        self.early_stopping = early_stopping
        self.normalized_scorers = normalized_scorers
        self.best_features = None
    
    def _get_obs(self) -> np.array:
        obs = {}
        for i in range(self.n_features):
            if len(self.features_selected) > i:
                obs[f'f{i}'] = self.normalized_features[self.features_selected[i]].values
            elif len(self.features_selected) == i:
                obs[f'f{i}'] = self.normalized_features[self.normalized_features.columns[self.current_feature_index]].values
            else:
                obs[f'f{i}'] = np.zeros(len(self.normalized_features), dtype=np.float32)
        # print(obs)
        return obs

    def _get_info(self) -> dict:
        return {
            'real_scores': self.real_scores,
            'normalized_scores': self.normalized_scores,
            'features_selected': self.features_selected,
        }
    
    def reset(self):
        self.current_feature_index = 0
        self.features_selected = []
        self.early_stopping.reset()
        observation = self._get_obs()

        self.previous_score = 0
        self.real_scores = {score_function.name: [] for score_function in self.normalized_scorers}
        self.normalized_scores = {score_function.name + '_norm': [] for score_function in self.normalized_scorers}
        self.scores_received = []

        # info = self._get_info()
        return observation
    
    def _get_reward(self, action):
        # selected_features = self.all_features.iloc[:, self.current_state.astype(bool)]
        if len(self.features_selected) == 0:
            return 0, 0
        features = self.all_features[self.features_selected]
        y_pred = self.clustering_model.fit_predict(features)
        
        try:
            """
            Test to see if this works well, otherwise we have two current options:
            (1) Use only the features selected by RL to compute metrics
            (2) Use raw time series to compute metrics
            """
            score, real_scores, normalized_scores = obtain_score(self.all_features, y_pred, self.normalized_scorers)
        except ValueError:
            # happens when all labels have same value, thus no real "clustering" has occurred
            score = -1
            real_scores = {score_function.name: None for score_function in self.normalized_scorers}
            normalized_scores = {score_function.name: None for score_function in self.normalized_scorers}
        
        for score_name in [score_function.name for score_function in self.normalized_scorers]:
            self.real_scores[score_name].append(real_scores[score_name])
            self.normalized_scores[score_name + '_norm'].append(normalized_scores[score_name])
        
        self.scores_received.append(score)

        gain = score - self.previous_score
        self.previous_score = score

        return gain, score
    
    def step(self, action):
        # print(f'Action: {action}')
        if action == 0:
            self.features_selected.append(self.normalized_features.columns[self.current_feature_index])
        self.current_feature_index += 1

        reward, score = self._get_reward(action)
        observation = self._get_obs()
        info = self._get_info()

        check_continuation = self.early_stopping.update_metric(score)
        if check_continuation == IMPROVE:
            self.best_features = copy.deepcopy(self.features_selected)
            done = False
        elif check_continuation == SAFE:
            done = False
        else:
            done = True
        
        if len(self.features_selected) < 5:
            done = False
        
        if (len(self.features_selected) - 1 >= self.n_features) or (self.current_feature_index == len(self.all_features.columns) - 1):
            done = True

        return observation, reward, done, info
    
    def render(self):
        pass

    def _render_frame(self):
        pass
    
    def close(self):
        pass
