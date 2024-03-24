import pandas as pd
import gym
import numpy as np
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


class FeatureSelectionEnv(gym.Env):

    def __init__(self, df_features, n_features, clustering_model, early_stopping, normalized_scorers, fixed_features=[]) -> None:
        self.observation_space = gym.spaces.Box(0, 1, shape=(len(df_features.columns),), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(df_features.columns))
        self.all_features = df_features.copy()
        self.n_features = n_features
        self.clustering_model = clustering_model
        self.early_stopping = early_stopping
        self.normalized_scorers = normalized_scorers
        self.best_features = None
        self.fixed_features = fixed_features
    
    def _get_obs(self) -> np.array:
        return self.current_state
    
    def _get_features_selected(self):
        # return [feature for i, feature in enumerate(self.all_features.columns) if self.current_state[i]]
        return self.features_selected

    def _get_info(self) -> dict:
        return {
            'real_scores': self.real_scores,
            'normalized_scores': self.normalized_scores,
            'features_selected': self._get_features_selected(),
        }
    
    def reset(self):
        self.features_selected = []
        self.current_state = np.zeros(len(self.all_features.columns))
        self.early_stopping.reset()
        observation = self._get_obs()

        self.real_scores = {score_function.name: [] for score_function in self.normalized_scorers}
        self.normalized_scores = {score_function.name + '_norm': [] for score_function in self.normalized_scorers}
        self.scores_received = []
        self.previous_score = 0
        
        if len(self.fixed_features) > 0:
            feature_names = list(self.all_features.columns)
            for feature in self.fixed_features:
                i = feature_names.index(feature)
                self.current_state[i] = 1
                self.features_selected.append(feature)
            _, _, _ = self._get_reward(same_action_selected=False)  # in order to update previous score

        # info = self._get_info()
        return observation
    
    def _get_reward(self, same_action_selected):
        # selected_features = self.all_features.iloc[:, self.current_state.astype(bool)]
        selected_features = self.all_features[self._get_features_selected()]

        if same_action_selected:
            return 0, selected_features, self.previous_score

        y_pred = self.clustering_model.fit_predict(selected_features)
        
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

        return gain, selected_features, score
    
    def step(self, action):
        same_action_selected = bool(self.current_state[action])
        self.current_state[action] = 1
        self.features_selected.append(self.all_features.columns[action])
        reward, selected_features, score = self._get_reward(same_action_selected)
        observation = self._get_obs()
        info = self._get_info()

        n_features_selected = len(self._get_features_selected())
        if n_features_selected < self.n_features:
            check_continuation = self.early_stopping.update_metric(score)
            if check_continuation == IMPROVE:
                self.best_features = selected_features
                done = False
            elif check_continuation == SAFE:
                done = False
            else:
                done = True
        else:
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
