import os
import pickle

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from jqmcvi.base import dunn_fast

from t2f.model.clustering import ClusterWrapper
from t2f.data.dataset import read_ucr_datasets
from t2f.selection.selection import cleaning

from normalization import ZScoreNormalization
from rl.agents.knn_td_agent import KNNTDAgent
from rl.agents.true_online_sarsa.true_online_sarsa import TrueOnlineSarsaLambda



"""Available datasets"""
data_path = 'data/'
datasets = {}
for dataset_file in os.listdir(data_path):
    if 'pkl' in dataset_file:
        dataset_name = dataset_file[:dataset_file.find('_feats')]
        datasets[dataset_name] = dataset_file


"""Available RL agents"""
rl_agents = {
    'kNN TD': {
        'parameters': {
            'k': ('k', 1, 12, 3),
            'alpha': ('α', 0.0, 1.0, 0.0001),
            'initial_epsilon': ('initial ε', 0.0, 1.0, 1.0),
            'min_epsilon': ('final ε', 0.0, 1.0, 0.05),
            'decay_episodes': ('number of episodes for ε decay', 1, 50, 20),
        }
    },
    'True Online Sarsa(λ)': {
        'parameters': {
            'lamb': ('λ', 0.0, 1.0, 0.9),
            'alpha': ('α', 0.0, 1.0, 0.001),
            'fourier_order': ('fourier order', 1, 12, 3),
            'initial_epsilon': ('initial ε', 0.0, 1.0, 1.0),
            'min_epsilon': ('final ε', 0.0, 1.0, 0.05),
            'decay_episodes': ('number of episodes for ε decay', 1, 50, 20),
        }
    }
}


"""Normalized metrics"""
silhouette_norm = ZScoreNormalization(
    score_function=silhouette_score,
    name='silhouette',
)

calinski_norm = ZScoreNormalization(
    score_function=calinski_harabasz_score,
    name='calinski_harabasz',
)

davies_bouldin_norm = ZScoreNormalization(
    score_function=davies_bouldin_score,
    maximize=False,  # minimum score is zero, with lower values indicating better clustering
    name='davies_bouldin',
)

dunn_index_norm = ZScoreNormalization(
    score_function=dunn_fast,
    name='dunn_index',
)

normalized_scorers = [
    silhouette_norm,
    calinski_norm,
    davies_bouldin_norm,
    dunn_index_norm
]


def open_dataset(dataset):
    model_type = 'Hierarchical'
    transform_type = 'minmax'

    ts_list, y_true = read_ucr_datasets(nameDataset=dataset, extract_path=data_path)
    n_clusters = len(set(y_true))  # Get number of clusters to find

    # Create cluster model
    model = ClusterWrapper(n_clusters=n_clusters, model_type=model_type, transform_type=transform_type)

    with open(data_path + datasets[dataset], 'rb') as pickle_file:
         df_all_feats = pickle.load(pickle_file)
    df_all_feats = cleaning(df_all_feats)

    return df_all_feats, y_true, model


def get_agent(selected_agent, parameters, env, obs):
    if selected_agent == 'kNN TD':
        agent = KNNTDAgent(
            starting_state=tuple(obs),
            state_space=env.observation_space,
            action_space=env.action_space,
            gamma=1.0,
            **parameters
        )
    elif selected_agent == 'True Online Sarsa(λ)':
        agent = TrueOnlineSarsaLambda(
            state_space=env.observation_space,
            action_space=env.action_space,
            gamma=1.0,
            **parameters
        )
    return agent


def get_action(agent, obs, action_masks, episode):
    if isinstance(agent, KNNTDAgent):
        action = agent.act(action_masks=action_masks, episode=episode)
    elif isinstance(agent, TrueOnlineSarsaLambda):
        action = agent.act(obs=tuple(obs), action_masks=action_masks, episode=episode)
    return action


def learn(agent, state, action, reward, next_state, done, action_masks):
    if isinstance(agent, KNNTDAgent):
        agent.learn(tuple(next_state), reward, done)
    elif isinstance(agent, TrueOnlineSarsaLambda):
        agent.learn(
            state=tuple(state),
            action=action,
            reward=reward,
            next_state=tuple(next_state),
            done=done,
            action_masks=action_masks
        )
