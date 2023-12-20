import sys
import gym
import numpy as np
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from jqmcvi.base import dunn_fast


def normalized_DBS(df_feat_all, y_pred):
    upper_limit = 100
    dbs = davies_bouldin_score(df_feat_all, y_pred)
    if dbs > upper_limit:
        dbs = upper_limit
    score = ((dbs - upper_limit)/(0-upper_limit))
    # print('DBS:', str(dbs), '| Normalized:', str(score))
    return dbs, score


def normalized_CHS(df_feat_all, y_pred):
    upper_limit = 1000
    chs = calinski_harabasz_score(df_feat_all, y_pred)
    if chs > upper_limit:
        chs = upper_limit
    score = chs/upper_limit
    # print('CHS:', str(chs), '| Normalized:', str(score))
    return chs, score


def normalized_DuS(df_feat_all, y_pred):
    upper_limit = 1
    dus = dunn_fast(df_feat_all, y_pred)
    if dus > upper_limit:
        dus = upper_limit
    score = dus/upper_limit
    # print('DuS:', str(dus), '| Normalized:', str(score))
    return dus, score


def normalized_SilS(df_feat_all, y_pred): 
    sils = silhouette_score(df_feat_all, y_pred)
    score = (sils + 1)/(2)
    # print('SilS:', str(sils), '| Normalized:', str(score))
    return sils, score


def obtain_score(df_feat_all, y_pred):
    dbs, scoreDBS = normalized_DBS(df_feat_all, y_pred)
    chs, scoreCHS = normalized_CHS(df_feat_all, y_pred)
    dus, scoreDuS = normalized_DuS(df_feat_all, y_pred)
    sils, scoreSilS = normalized_SilS(df_feat_all, y_pred)
    scores = [scoreDBS, scoreCHS, scoreDuS, scoreSilS]
    score = np.mean(scores)
    return score, {
        'total_score': score,
        'dbs': dbs,
        'scoreDBS': scoreDBS,
        'chs': chs,
        'scoreCHS': scoreCHS,
        'dus': dus,
        'scoreDuS': scoreDuS,
        'sils': sils,
        'scoreSilS': scoreSilS,
    }


class FeatureSelectionEnvironment(gym.Env):

    def __init__(self, df_features, n_features, clustering_model) -> None:
        self.observation_space = gym.spaces.Box(0, 1, shape=(len(df_features.columns),), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(df_features.columns))
        
        self.all_features = df_features.copy()
        self.current_state = np.zeros(len(self.all_features.columns))
        self.n_features = n_features
        self.clustering_model = clustering_model
        
        self.past_reward = 0
        self.score_history = []

    
    def _get_obs(self) -> np.array:
        return self.current_state

    def _get_info(self) -> dict:
        return {
            'legal_actions': [action for action in range(self.action_space.n) if not self.current_state[action]],
            'score_history': self.score_history,
        }
    
    def reset(self, seed=None, options=None):
        
        self.past_reward = 0
        self.score_history = []

        self.current_state = np.zeros(len(self.all_features.columns))
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def _get_reward(self, action):
        selected_features = [feature for i, feature in enumerate(self.all_features.columns) if self.current_state[i]]
        y_pred = self.clustering_model.fit_predict(self.all_features[selected_features])
        
        try:
            """
            Test to see if this works well, otherwise we have two current options:
            (1) Use only the features selected by RL to compute metrics
            (2) Use raw time series to compute metrics
            """
            score, detailed_scores = obtain_score(self.all_features, y_pred)
        except ValueError:
            # happens when all labels have same value, thus no real "clustering" has occurred
            score = -1
            detailed_scores = {
                'total_score': np.nan,
                'dbs': np.nan,
                'scoreDBS': np.nan,
                'chs': np.nan,
                'scoreCHS': np.nan,
                'dus': np.nan,
                'scoreDuS': np.nan,
                'sils': np.nan,
                'scoreSilS': np.nan,
            }
        
        gain = score - self.past_reward
        self.past_reward = score

        self.score_history.append(detailed_scores)
        
        return gain
    
    def step(self, action):
        self.current_state[action] = 1
        reward = self._get_reward(action)
        observation = self._get_obs()
        info = self._get_info()
        
        if len(self.current_state[self.current_state == 1]) < self.n_features:
            terminated = False
        else:
            terminated = True

        return observation, reward, terminated, False, info
    
    def render(self):
        pass

    def _render_frame(self):
        pass
    
    def close(self):
        pass
