# Application of Time2Feat on real dataset
import numpy as np
import pandas as pd
from t2f.extraction.extractor import feature_extraction
from t2f.utils.importance_old import feature_selection
from t2f.model.clustering import ClusterWrapper
from t2f.data.dataset import read_ucr_datasets
from t2f.selection.selection import cleaning
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_mutual_info_score

from rl.environments.feature_selection_env import FeatureSelectionEnvironment
from rl.agents.ql_agent import QLAgent

from tqdm import tqdm


if __name__ == '__main__':
    files = ['data/BasicMotions/BasicMotions_TRAIN.txt',
             'data/BasicMotions/BasicMotions_TEST.txt']
    transform_type = 'minmax'
    model_type = 'Hierarchical'
    train_size = 0.3
    batch_size = 500
    p = 4

    # Read original dataset
    print('Read ucr datasets: ', files)
    ts_list, y_true = read_ucr_datasets(paths=files)
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

    df_all_feats = feature_extraction(ts_list, batch_size, p)

    df_all_feats = cleaning(df_all_feats)

    total_number_of_features = len(df_all_feats.columns)
    print('Total number of features:', total_number_of_features)

    episodes = 30
    n_features = round(total_number_of_features * 0.05)

    env = FeatureSelectionEnvironment(
        df_features=df_all_feats,
        n_features=n_features,
        clustering_model=model,
    )

    obs, info = env.reset()
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

    for episode in tqdm(range(episodes)):
        # print(f'Episode {episode}')
        while not done:
            action = agent.act(info['legal_actions'])
            # print('Action:', action)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # print(reward)
            results[episode]['reward'].append(reward)  # save reward

            agent.learn(tuple(next_obs), reward, done)
            done = terminated or truncated
        
        for detailed_scores in info['score_history']:
            for k, v in detailed_scores.items():
                results[episode][k].append(v)
        
        obs, info = env.reset()
        done = False

    for episode in range(episodes):
        results_df = pd.DataFrame(results[episode])
        results_df.to_csv(f'results/results_episode{episode}.csv')
