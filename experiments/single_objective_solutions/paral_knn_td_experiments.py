import pickle
import time
import numpy as np
import pandas as pd

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
from multiprocessing import Pool
from sb3_contrib.common.maskable.utils import get_action_masks

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from jqmcvi.base import dunn_fast

from normalization import ZScoreNormalization, MinMaxNormalization
from early_stopping import PlateauEarlyStopping

from rl.environments.feature_selection_env import FeatureSelectionEnv
from rl.agents.ql_agent import QLAgent
from rl.agents.knn_td_agent import KNNTDAgent

from tqdm import tqdm

def run_experiment(params):
    k, alpha, decay_episodes, df_all_feats, n_features,original_features, model, normalized_scorers, y_true, nameDataset, episodes_total = params
    results = None
    list_AMI = []
    list_feat = []
    list_time = []
    for episodes in episodes_total:
        env = FeatureSelectionEnv(
            df_features=df_all_feats,
            n_features=n_features,
            clustering_model=model,
            normalized_scorers=normalized_scorers,
            early_stopping=PlateauEarlyStopping(
                patience=20,
                plateau_patience=20,
                threshold=0.12
            )
        )
        obs = env.reset()
        done = False
        agent = KNNTDAgent(
            starting_state=tuple(obs),
            state_space=env.observation_space,
            action_space=env.action_space,
            k=k,
            alpha=alpha,
            gamma=1.0,
            initial_epsilon=1.0,
            min_epsilon=0.05,
            decay_episodes=decay_episodes
        )
        for episode in tqdm(range(episodes)):
            start_time_ep = time.time()
            rewards = []
            features = []
            while not done:
                action_masks = get_action_masks(env)
                action = agent.act(action_masks=action_masks, episode=episode)
                next_obs, reward, done, info = env.step(action)
                features_selected = info['features_selected']
                features.append(' | '.join(features_selected))
                agent.learn(tuple(next_obs), reward, done)
                rewards.append(reward)

            y_pred = model.fit_predict(df_all_feats[features_selected])
            AMI = adjusted_mutual_info_score(y_pred, y_true)
            list_AMI.append(AMI)
            list_feat.append(len(features_selected))
            list_time.append(time.time() - start_time_ep)
            obs = env.reset()
            done = False
        best_index = np.argmax(list_AMI)
        new_results = pd.DataFrame({
            'k': [k],
            'alpha': [alpha],
            'decay_episodes': [decay_episodes],
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
        new_results.to_csv(f'results/{nameDataset}/knn_results_{nameDataset}.csv', mode='a',index=False,header=False)



if __name__ == '__main__':
    dataset_names = ["Libras"]
    # dataset_names = ["ERing"]
    transform_type = 'minmax'
    model_type = 'Hierarchical'
    train_size = 0.3
    batch_size = 500
    p = 1# os.cpu_count()

    silhouette_norm = ZScoreNormalization(score_function=silhouette_score, name='silhouette')
    calinski_norm = ZScoreNormalization(score_function=calinski_harabasz_score, name='calinski_harabasz')
    davies_bouldin_norm = ZScoreNormalization(score_function=davies_bouldin_score, maximize=False,
                                              name='davies_bouldin')
    dunn_index_norm = ZScoreNormalization(score_function=dunn_fast, name='dunn_index')

    normalized_scorers = [silhouette_norm, calinski_norm, davies_bouldin_norm, dunn_index_norm]

    for nameDataset in dataset_names:
        print(f'Dataset: {nameDataset}')
        extract_time = 0
        ts_list, y_true = read_ucr_datasets(nameDataset=nameDataset)
        n_clusters = len(set(y_true))  # Get number of clusters to find
        new_results = pd.DataFrame({
            'k':['k'] ,
            'alpha':['alpha'] ,
            'decay_episodes':['decay_episodes'] ,
            'episode':['episode'] ,
            'feat_max':['feat_max'] ,
            'features_average':['features_average'] ,
            'AMI_average':['AMI_average'] ,
            'features_std':['features_std'] ,
            'AMI_std':['AMI_std'] ,
            'best_AMI':['best_AMI'] ,
            'best_feat':['best_feat'] ,
            'time_average':['time_average'] ,
        })
        if not os.path.exists(f'results/{nameDataset}'):
            os.mkdir(f'results/{nameDataset}')
        new_results.to_csv(f'results/{nameDataset}/knn_results_{nameDataset}.csv', mode='a',index=False,header=False)


        # Create cluster model
        model = ClusterWrapper(n_clusters=n_clusters, model_type=model_type, transform_type=transform_type)
        print('Dataset shape: {}, Num of clusters: {}'.format(ts_list.shape, n_clusters))

        if not os.path.isfile("../../data/" + nameDataset + "_feats.pkl"):
            time_start = time.time()
            df_all_feats = feature_extraction(ts_list, batch_size, p)
            extract_time = time.time() - time_start
            file = open("../../data/" + nameDataset + "_feats.pkl", 'wb')
            pickle.dump(df_all_feats, file)

        with open("../../data/" + nameDataset + "_feats.pkl", 'rb') as pickle_file:
            df_all_feats = pickle.load(pickle_file)

        df_all_feats = cleaning(df_all_feats)
        print('Number of features:', len(df_all_feats.columns))

        total_number_of_features = [20]
        episodes_total = [5]

        print('Total number of features:', total_number_of_features)
        for n_features in total_number_of_features:
            original_features = n_features
            if n_features > len(df_all_feats.columns):
                n_features = len(df_all_feats.columns) - 1

        params_list = []
        for k in [1, 3, 9]:
            for alpha in [0.1, 0.001, 0.0001]:
                for decay_episodes in [10, 20, 30]:
                    params_list.append((k, alpha, decay_episodes, df_all_feats, n_features,original_features, model, normalized_scorers,
                                        y_true, nameDataset, episodes_total))

        with Pool(processes=os.cpu_count()) as pool:
            pool.map(run_experiment, params_list)