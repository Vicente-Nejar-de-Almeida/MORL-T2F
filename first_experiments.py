# Application of Time2Feat on real dataset
import pickle
import csv
import time
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
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from jqmcvi.base import dunn_fast
import NormalizedScore as ns

from scipy.stats import pearsonr
from early_stopping_class import CustomEarlyStopping

from rl.environments.feature_selection_env import FeatureSelectionEnvironment
from rl.agents.ql_agent import QLAgent

from tqdm import tqdm


if __name__ == '__main__':
    listNameDataset = ["ERing"]
    transform_type = 'minmax'
    model_type = 'Hierarchical'
    train_size = 0.3
    batch_size = 500
    p = os.cpu_count()
    f = open('resultsT2RL.csv', 'w+', newline='')
    writer = csv.writer(f)
    header = ["Dataset", "n_feats", "AMI", "Time"]
    writer.writerow(header)

    silhouette_norm = ns.NormalizedScore(silhouette_score)
    calinski_norm = ns.NormalizedScore(calinski_harabasz_score)

    norm_list = [silhouette_norm, calinski_norm]

    for nameDataset in listNameDataset:
        extractTime = 0
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
            # print('Number of Labels: {}'.format(len(labels)))

        if not os.path.isfile("data//"+nameDataset+"_feats.pkl"):
            timeStart = time.time()
            df_all_feats = feature_extraction(ts_list, batch_size, p)
            extractTime = time.time() - timeStart
            file = open("data//"+nameDataset+"_feats.pkl", 'wb')
            pickle.dump(df_all_feats, file)

        with open("data//"+nameDataset+"_feats.pkl", 'rb') as pickle_file:
            df_all_feats = pickle.load(pickle_file)

        df_all_feats = cleaning(df_all_feats)
        print(len(df_all_feats))
        total_number_of_features = [50]
        episodes_total = [200]


        for episodes in episodes_total:
            print('Total number of features:', total_number_of_features)
            for n_features in total_number_of_features:
                if n_features > len(df_all_feats.columns):
                    n_features = len(df_all_feats.columns) - 1
                startTime = time.time()
                # n_features = 130
                env = FeatureSelectionEnvironment(
                    df_features=df_all_feats,
                    n_features=n_features,
                    clustering_model=model,
                    list_eval=norm_list
                )

                obs, info = env.reset(norm_list)
                done = False

                agent = QLAgent(starting_state=tuple(
                    obs), state_space=env.observation_space, action_space=env.action_space)

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

                results_total = {val: {'total_score':[],'AMI':[]} for val in range(episodes)}

                AMIVal = []
                y_pred = y_true

                for episode in tqdm(range(episodes)):
                    # print(f'Episode {episode}')
                    while not done:
                        action = agent.act(info['legal_actions'])
                        # print('Action:', action)
                        next_obs, reward, terminated, truncated, info, features_selected = env.step(action)
                        agent.learn(tuple(next_obs), reward, done)
                        if len(features_selected) > n_features:
                            done = True
                        else:
                            done = terminated or truncated


                    for detailed_scores in info['score_history']:
                        for k, v in detailed_scores.items():
                            results[episode][k].append(v)

                    obs, info = env.reset(norm_list)
                    done = False
                y_pred = model.fit_predict(df_all_feats[features_selected])
                AMI = adjusted_mutual_info_score(y_pred, y_true)
                finTime = (time.time() - startTime) + extractTime
                results = [nameDataset, n_features, AMI, finTime]
                writer.writerow(results)
                print(f"{nameDataset}, has obtained a value of AMI equal to {AMI} with {len(env.best_conf_feat)} features with time {finTime}")
                # print("The Dataset %s has obtained ")
                # print("AMI: ", AMIVal[len(AMIVal) - 1])
                print('**********************')
