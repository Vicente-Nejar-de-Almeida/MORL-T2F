# Application of Time2Feat on real dataset
import pickle

import numpy as np
import pandas as pd
from t2f.extraction.extractor import feature_extraction
from t2f.utils.importance_old import feature_selection
from t2f.model.clustering import ClusterWrapper
from t2f.data.dataset import read_ucr_datasets
from t2f.selection.selection import cleaning
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_mutual_info_score
import os

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from jqmcvi.base import dunn_fast
import NormalizedScore as ns


from rl.environments.multi_objective_feature_selection_env import MultiObjectiveFeatureSelectionEnvironment
from rl.agents.ql_agent import QLAgent

from morl_baselines.multi_policy.envelope.envelope import Envelope


if __name__ == '__main__':
    listNameDataset = ["BasicMotions"]
    transform_type = 'minmax'
    model_type = 'Hierarchical'
    train_size = 0.3
    batch_size = 500
    p = 8

    silhouette_norm = ns.NormalizedScore(silhouette_score)
    calinski_norm = ns.NormalizedScore(calinski_harabasz_score)
    norm_list = [silhouette_norm, calinski_norm]
    for nameDataset in listNameDataset:
        # Read original dataset
        # print('Read ucr datasets: ', files)
        ts_list, y_true = read_ucr_datasets(nameDataset=nameDataset)
        n_clusters = len(set(y_true))  # Get number of clusters to find

        # Create cluster model
        model = ClusterWrapper(n_clusters=n_clusters,
                               model_type=model_type, transform_type=transform_type)
        print('Dataset shape: {}, Num of clusters: {}'.format(
            ts_list.shape, n_clusters))

        # This side is dedicated for the Semi-Supervised
        labels = {}
        if train_size > 0:
            # Extract a subset of labelled mts to train the semi-supervised model
            idx_train, _, y_train, _ = train_test_split(
                np.arange(len(ts_list)), y_true, train_size=train_size)
            labels = {i: j for i, j in zip(idx_train, y_train)}
            print('Number of Labels: {}'.format(len(labels)))

        if not os.path.isfile("data//"+nameDataset+"_feats.pkl"):
            df_all_feats = feature_extraction(ts_list, batch_size, p)
            file = open("data//"+nameDataset+"_feats.pkl", 'wb')
            pickle.dump(df_all_feats, file)

        with open("data//"+nameDataset+"_feats.pkl", 'rb') as pickle_file:
            df_all_feats = pickle.load(pickle_file)

        df_all_feats = cleaning(df_all_feats)
        total_number_of_features = len(df_all_feats.columns)
        print('Total number of features:', total_number_of_features)

        episodes = 800
        n_features = round(total_number_of_features * 0.15)
        # n_features = 130
        env = MultiObjectiveFeatureSelectionEnvironment(
            df_features=df_all_feats,
            n_features=n_features,
            clustering_model=model,
            list_eval=norm_list
        )

        results = {e: {
            'reward': [],
            'total_score': [],
            'dbs': [],
            'scoreDBS': [],
            'chs': [],
            'scoreCHS': [],
            'dus': [],
            'scoreDuS': [],
            'sils': [],
            'scoreSilS': [],
        } for e in range(episodes)}

        AMIVal = []
        y_pred = y_true

        agent = Envelope(
            env,
            max_grad_norm=0.1,
            learning_rate=3e-4,
            gamma=0.98,
            batch_size=64,
            net_arch=[256, 256, 256, 256],
            buffer_size=int(2e6),
            initial_epsilon=1.0,
            final_epsilon=0.05,
            epsilon_decay_steps=50000,
            initial_homotopy_lambda=0.0,
            final_homotopy_lambda=1.0,
            homotopy_decay_steps=10000,
            learning_starts=100,
            envelope=True,
            gradient_updates=1,
            target_net_update_freq=1000,  # 1000,  # 500 reduce by gradient updates
            tau=1,
            log=False,
        )

        agent.train(
            total_timesteps=4000,
            total_episodes=episodes,
            weight=None,
            eval_env=env,
            ref_point=np.array([0, 0, 0, -200.0]),
            # known_pareto_front=env.unwrapped.pareto_front(gamma=0.98),
            num_eval_weights_for_front=5,
            eval_freq=50,
            reset_num_timesteps=False,
            reset_learning_starts=False,
        )

        print('Training finished')

        obs, info = env.reset()
        done = False
        while not done:
            action = agent.act(obs, w=np.array(1/len(norm_list) for x in range(len(norm_list))))
            print(f'Action: {action}')
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        y_pred = info['y_pred']
        print('Adjusted mutual info score:', adjusted_mutual_info_score(y_pred, y_true))
