# Application of Time2Feat on real dataset
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

from sb3_contrib.common.maskable.utils import get_action_masks

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from jqmcvi.base import dunn_fast

from normalization import ZScoreNormalization, MinMaxNormalization
from early_stopping import PlateauEarlyStopping

from rl.environments.feature_selection_env import FeatureSelectionEnv
from rl.agents.ql_agent import QLAgent

from tqdm import tqdm


if __name__ == '__main__':
    # dataset_names = ["ERing", "RacketSports"]
    dataset_names = ["ERing"]
    transform_type = 'minmax'
    model_type = 'Hierarchical'
    train_size = 0.3
    batch_size = 500
    p = os.cpu_count()

    # silhouette_norm = ZScoreNormalization(silhouette_score, name='silhouette')
    # calinski_norm = ZScoreNormalization(calinski_harabasz_score, name='calinski_harabasz')

    """
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
    """

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
        # silhouette_norm,
        calinski_norm,
        # davies_bouldin_norm,
        # dunn_index_norm
    ]

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
        print(len(df_all_feats))
        total_number_of_features = [25]
        episodes_total = [100]

        for episodes in episodes_total:
            print('Total number of features:', total_number_of_features)
            for n_features in total_number_of_features:
                if n_features > len(df_all_feats.columns):
                    n_features = len(df_all_feats.columns) - 1
                
                start_time = time.time()

                print('Before:')
                print(df_all_feats)
                df_all_feats = df_all_feats[list(reversed(df_all_feats.columns))]
                print('After:')
                print(df_all_feats)

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

                agent = QLAgent(
                    starting_state=tuple(obs),
                    state_space=env.observation_space,
                    action_space=env.action_space
                )

                y_pred = y_true

                results = None

                for episode in tqdm(range(episodes)):
                    # print(f'Episode: {episode}')
                    rewards = []
                    features = []
                    while not done:
                        action_masks = get_action_masks(env)
                        action = agent.act(action_masks=action_masks, episode=episode)
                        # print('Action:', action)
                        next_obs, reward, done, info = env.step(action)
                        features_selected = info['features_selected']
                        features.append(' | '.join(features_selected))
                        agent.learn(tuple(next_obs), reward, done)
                        rewards.append(reward)

                    """
                    for detailed_scores in info['score_history']:
                        for k, v in detailed_scores.items():
                            results[episode][k].append(v)
                    """
                    new_results = pd.DataFrame({
                        'episode': episode,
                        'rewards': rewards,
                        'features': features,
                        'scores': env.scores_received,
                        **env.real_scores,
                        **env.normalized_scores,
                    })
                    if results is None:
                        results = new_results.copy()
                    else:
                        results = pd.concat([results, new_results])


                    print(env.current_state)
                    obs = env.reset()
                    done = False
                # results.to_csv(f'results/calinksi_600episodes_test_results_{nameDataset}.csv', index=False)
                y_pred = model.fit_predict(df_all_feats[features_selected])
                AMI = adjusted_mutual_info_score(y_pred, y_true)
                finTime = (time.time() - start_time) + extract_time
                # results = [nameDataset, n_features, AMI, finTime]
                # writer.writerow(results)
                print(f"{nameDataset}, has obtained a value of AMI equal to {AMI} with {len(features_selected)} features with time {finTime}")
                # print("The Dataset %s has obtained ")
                # print("AMI: ", AMIVal[len(AMIVal) - 1])
                print('**********************')
