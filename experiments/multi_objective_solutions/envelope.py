import pickle
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

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

from normalization import MinMaxNormalization
import gymnasium as gym
import mo_gymnasium as mo_gym

from mo_gymnasium.utils import MORecordEpisodeStatistics
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS
from morl_baselines.multi_policy.envelope.envelope import Envelope

from rl.environments.multi_objective_feature_selection_env import MOFeatureSelectionEnv

GAMMA = 0.98

dataset_names = ['BasicMotions']
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

    if not os.path.isfile("../../data/"+nameDataset+"_feats.pkl"):
        time_start = time.time()
        df_all_feats = feature_extraction(ts_list, batch_size, p)
        extract_time = time.time() - time_start
        file = open("../../data/"+nameDataset+"_feats.pkl", 'wb')
        pickle.dump(df_all_feats, file)

    with open("../../data/"+nameDataset+"_feats.pkl", 'rb') as pickle_file:
        df_all_feats = pickle.load(pickle_file)

    df_all_feats = cleaning(df_all_feats)

    # Creation Metrics
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


    total_number_of_features = [25]
    episodes_total = [5,10]
    ami_results = None
    for episodes in episodes_total:
        print('Total number of features:', total_number_of_features)

        for n_features in total_number_of_features:
            if n_features > len(df_all_feats.columns):
                original_features = n_features
                n_features = len(df_all_feats.columns) - 1
            else:
                original_features = n_features

            env = mo_gym.make(
                "mo-feature-selection-env-v0",
                df_features=df_all_feats,
                n_features=n_features,
                clustering_model=model,
                list_eval=normalized_scorers
            )

            obs = env.reset()
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
                total_timesteps=100,
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

            df_all_feats = df_all_feats[list(reversed(df_all_feats.columns))]
            y_pred = y_true
            list_AMI = []
            list_feat = []
            list_time = []
            obs = env.reset()

            for w in [
                [0.25, 0.25, 0.25, 0.25],
                [0.52, 0.16, 0.16, 0.16],
                [0.16, 0.52, 0.16, 0.16],
                [0.16, 0.16, 0.52, 0.16],
                [0.16, 0.16, 0.16, 0.52],
                [0.4, 0.4, 0.1, 0.1],
                [0.4, 0.1, 0.4, 0.1],
                [0.4, 0.1, 0.1, 0.4],
                [0.1, 0.4, 0.4, 0.1],
                [0.1, 0.4, 0.1, 0.4],
                [0.1, 0.1, 0.4, 0.4],
            ]:
                for episode in tqdm(range(episodes)):
                    start_time_ep = time.time()
                    rewards = []
                    features = []
                    while not done:
                        action = agent.act(obs=obs, w=w)
                        # print('Action:', action)
                        obs, reward, done, _, info = env.step(action)
                        features_selected = info['features_selected']
                        features.append(' | '.join(features_selected))
                        rewards.append(reward)

                    y_pred = model.fit_predict(df_all_feats[features_selected])
                    AMI = adjusted_mutual_info_score(y_pred, y_true)
                    list_AMI.append(AMI)
                    list_feat.append(len(features_selected))
                    list_time.append(time.time() - start_time_ep)
                    # print(env.current_state)
                    obs = env.reset()
                    done = False
                best_index = np.argmax(list_AMI)
                new_ami_results = pd.DataFrame({
                    'w': [str(w)],
                    'episode': [episodes],
                    'feat_max': [original_features],
                    'features_average': [np.average(list_feat)],
                    'AMI_average': [np.average(list_AMI)],
                    'features_std': [np.std(list_feat)],
                    'AMI_std': [np.std(list_AMI)],
                    'best_AMI': [list_AMI[best_index]],
                    'best_feat': [list_feat[best_index]],
                    'time_average': [np.average(list_time)]
                })
                if ami_results is None:
                    ami_results = new_ami_results.copy()
                else:
                    ami_results = pd.concat([ami_results, new_ami_results])


    if not os.path.exists(f'results/{nameDataset}'):
        os.mkdir(f'results/{nameDataset}')
    ami_results.to_csv(f'results/{nameDataset}/morl_envelope_ami_results_{nameDataset}.csv', index=False)