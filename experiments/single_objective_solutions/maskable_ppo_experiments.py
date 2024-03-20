# Application of Time2Feat on real dataset
import pickle
import time
import numpy as np

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

from rl.environments.feature_selection_env import FeatureSelectionEnv

from normalization.z_score_normalization import ZScoreNormalization
from early_stopping.plateau_early_stopping import PlateauEarlyStopping

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__ == '__main__':
    listNameDataset = ["ERing"]
    transform_type = 'minmax'
    model_type = 'Hierarchical'
    train_size = 0.3
    batch_size = 500
    p = os.cpu_count()
    header = ["Dataset", "n_feats", "AMI", "Time"]

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
        dunn_index_norm,
    ]

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

        if not os.path.isfile("..//..//data//"+nameDataset+"_feats.pkl"):
            timeStart = time.time()
            df_all_feats = feature_extraction(ts_list, batch_size, p)
            extractTime = time.time() - timeStart
            file = open("..//..//data//"+nameDataset+"_feats.pkl", 'wb')
            pickle.dump(df_all_feats, file)

        with open("..//..//data//"+nameDataset+"_feats.pkl", 'rb') as pickle_file:
            df_all_feats = pickle.load(pickle_file)

        df_all_feats = cleaning(df_all_feats)
        print(len(df_all_feats))

        n_features = 50

        def make_env():
            return FeatureSelectionEnv(
                df_features=df_all_feats,
                n_features=n_features,
                clustering_model=model,
                early_stopping=PlateauEarlyStopping(patience=20, plateau_patience=20),
                normalized_scorers=normalized_scorers
            )
        
        env = DummyVecEnv([make_env])
        agent = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=0.001,
            n_steps=10,
            batch_size=5,
            gamma=1,
            verbose=1
        )
        agent.learn(total_timesteps=10000, log_interval=1)

        env = make_env()
        obs = env.reset()
        done = False
        while not done:
            action_masks = get_action_masks(env)
            action, _states = agent.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, done, info = env.step(action)
            print(f'Action: {action}, Reward: {reward}')
        
        # print(obs)
        # print(df_all_feats)
        # print(df_all_feats.iloc[:, obs])
        # y_pred = model.fit_predict(df_all_feats.iloc[:, obs])
        y_pred = model.fit_predict(env.best_features)
        AMI = adjusted_mutual_info_score(y_pred, y_true)
        print(f"{nameDataset}, has obtained a value of AMI equal to {AMI} with {len(env.best_features.columns)} features")
