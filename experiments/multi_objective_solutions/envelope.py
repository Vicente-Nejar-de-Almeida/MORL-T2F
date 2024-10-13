import pickle
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

import sys
sys.path.append("../..")

from t2f.extraction.extractor import feature_extraction
from t2f.model.clustering import ClusterWrapper
from t2f.data.dataset import read_ucr_datasets
from t2f.selection.selection import cleaning
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_mutual_info_score
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from jqmcvi.base import dunn_fast

from normalization import ZScoreNormalization, MinMaxNormalization

import gymnasium as gym
import mo_gymnasium as mo_gym

from mo_gymnasium.utils import MORecordEpisodeStatistics
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS
from morl_baselines.multi_policy.envelope.envelope import Envelope

from rl.environments.multi_objective_feature_selection_env import MOFeatureSelectionEnv

GAMMA = 0.98

dataset_names = ['BasicMotions', 'Libras', 'ERing', 'RacketSports']
transform_type = 'minmax'
model_type = 'Hierarchical'
train_size = 0.3
batch_size = 500
p = os.cpu_count()


for nameDataset in dataset_names:
    extract_time = 0
    ts_list, y_true = read_ucr_datasets(nameDataset=nameDataset)
    n_clusters = len(set(y_true))  # Get number of clusters to find

    # Create cluster model
    model = ClusterWrapper(n_clusters=n_clusters, model_type=model_type, transform_type=transform_type)
    print('Dataset shape: {}, Num of clusters: {}'.format(ts_list.shape, n_clusters))

    # This side is dedicated for the Semi-Supervised
    labels = {}
    if train_size > 0:
        # Extract a subset of labelled mts to train the semi-supervised model
        idx_train, _, y_train, _ = train_test_split(
            np.arange(len(ts_list)), y_true, train_size=train_size)
        labels = {i: j for i, j in zip(idx_train, y_train)}
        # print('Number of Labels: {}'.format(len(labels)))

    if not os.path.isfile("../../data/"+nameDataset+"_feats.pkl"):
        time_start = time.time()
        df_all_feats = feature_extraction(ts_list, batch_size, p)
        extract_time = time.time() - time_start
        file = open("../../data/"+nameDataset+"_feats.pkl", 'wb')
        pickle.dump(df_all_feats, file)

    with open("../../data/"+nameDataset+"_feats.pkl", 'rb') as pickle_file:
        df_all_feats = pickle.load(pickle_file)

    df_all_feats = cleaning(df_all_feats)
    # print(len(df_all_feats))
    total_number_of_features = [25]
    episodes_total = [100]

    for episodes in episodes_total:
        print('Total number of features:', total_number_of_features)
        for n_features in total_number_of_features:
            if n_features > len(df_all_feats.columns):
                n_features = len(df_all_feats.columns) - 1
            
            start_time = time.time()

            #print('Before:')
            #print(df_all_feats)
            df_all_feats = df_all_feats[list(reversed(df_all_feats.columns))]
            #print('After:')
            #print(df_all_feats)

            silhouette_norm = MinMaxNormalization(
                score_function=silhouette_score,
                min_val=-1,
                max_val=1,
                name='silhouette',
            )

            
            calinski_norm = MinMaxNormalization(
                score_function=calinski_harabasz_score,
                min_val=0,
                max_val=25,
                name='calinski_harabasz',
            )

            davies_bouldin_norm = MinMaxNormalization(
                score_function=davies_bouldin_score,
                min_val=0,
                max_val=12,
                maximize=False,  # minimum score is zero, with lower values indicating better clustering
                name='davies_bouldin',
            )

            dunn_index_norm = MinMaxNormalization(
                score_function=dunn_fast,
                min_val=0,
                max_val=1,
                name='dunn_index',
            )
            

            normalized_scorers = [
                silhouette_norm,
                calinski_norm,
                davies_bouldin_norm,
                dunn_index_norm
            ]

            """
            env = MOFeatureSelectionEnv(
                df_features=df_all_feats,
                n_features=n_features,
                clustering_model=model,
                list_eval=normalized_scorers,
            )
            """
            env = mo_gym.make(
                "mo-feature-selection-env-v0",
                df_features=df_all_feats,
                n_features=n_features,
                clustering_model=model,
                list_eval=normalized_scorers
            )

            obs, info = env.reset()
            done = False

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
                project_name="MORL-Baselines",
                experiment_name="Envelope",
            )

            agent.train(
                total_timesteps=100000,
                total_episodes=None,
                weight=None,
                eval_env=env,
                ref_point=np.array([0, 0, 0, 0]),
                known_pareto_front=None,
                num_eval_weights_for_front=100,
                eval_freq=1000,
                reset_num_timesteps=False,
                reset_learning_starts=False,
            )

            y_pred = y_true

            results = None
            ami_results = None
            AMI_values = []

            obs, info = env.reset()

            # for episode in tqdm(range(episodes)):
            for w in [
                torch.tensor(np.array([0.25, 0.25, 0.25, 0.25]), dtype=torch.float32),
                torch.tensor(np.array([0.52, 0.16, 0.16, 0.16]), dtype=torch.float32),
                torch.tensor(np.array([0.16, 0.52, 0.16, 0.16]), dtype=torch.float32),
                torch.tensor(np.array([0.16, 0.16, 0.52, 0.16]), dtype=torch.float32),
                torch.tensor(np.array([0.16, 0.16, 0.16, 0.52]), dtype=torch.float32),
                torch.tensor(np.array([0.4, 0.4, 0.1, 0.1]), dtype=torch.float32),
                torch.tensor(np.array([0.4, 0.1, 0.4, 0.1]), dtype=torch.float32),
                torch.tensor(np.array([0.4, 0.1, 0.1, 0.4]), dtype=torch.float32),
                torch.tensor(np.array([0.1, 0.4, 0.4, 0.1]), dtype=torch.float32),
                torch.tensor(np.array([0.1, 0.4, 0.1, 0.4]), dtype=torch.float32),
                torch.tensor(np.array([0.1, 0.1, 0.4, 0.4]), dtype=torch.float32),
            ]:
                # print(f'Episode: {episode}')
                rewards = []
                features = []
                while not done:
                    action = agent.act(obs=torch.tensor(obs, dtype=torch.float32), w=w)
                    # print('Action:', action)
                    obs, reward, done, _, info = env.step(action)
                    features_selected = info['features_selected']
                    features.append(' | '.join(features_selected))
                    rewards.append(reward)

                new_results = pd.DataFrame({
                    'w': [str(w) for _ in rewards],
                    'rewards': rewards,
                    'features': features,
                    'normalized_scores': env.scores_received,
                    #**env.real_scores,
                    #**env.normalized_scores,
                })
                if results is None:
                    results = new_results.copy()
                else:
                    results = pd.concat([results, new_results])

                y_pred = model.fit_predict(df_all_feats[features_selected])
                AMI = adjusted_mutual_info_score(y_pred, y_true)
                AMI_values.append(AMI)

                new_ami_results = pd.DataFrame({
                    'w': str(w),
                    'AMI': [AMI]
                })
                if ami_results is None:
                    ami_results = new_ami_results.copy()
                else:
                    ami_results = pd.concat([ami_results, new_ami_results])
                


                # print(env.current_state)
                obs, info = env.reset()
                done = False
            results.to_csv(f'results/morl_envelope_stepwise_results_{nameDataset}.csv', index=False)
            ami_results.to_csv(f'results/morl_envelope_ami_results_{nameDataset}.csv', index=False)
            
            finTime = (time.time() - start_time) + extract_time
            # results = [nameDataset, n_features, AMI, finTime]
            # writer.writerow(results)
            print(f"{nameDataset}, has obtained a value of AMI equal to {max(AMI_values)} with {len(features_selected)} features with time {finTime}")
            # print("The Dataset %s has obtained ")
            # print("AMI: ", AMIVal[len(AMIVal) - 1])
            print('**********************')
