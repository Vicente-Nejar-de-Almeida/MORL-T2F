import math
import copy
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns

from sb3_contrib.common.maskable.utils import get_action_masks

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

from early_stopping import PlateauEarlyStopping
from rl.environments.feature_selection_env import FeatureSelectionEnv

from utils import *


COLS_IN_PLOT = 2


def cluster_label_to_string(label):
    if isinstance(label, str) and re.search('[a-zA-Z]', label):
        return label.capitalize()
    else:
        return f'Cluster {label}'


# Tab 1
def reset_local_info_tab1():
    st.session_state.tab1_local_metrics = {k: [np.nan] for k in metric_names.keys()}
    st.session_state.tab1_local_rewards = []


def update_global_info_tab1():
    for k, v in st.session_state.tab1_local_metrics.items():
        st.session_state.tab1_global_metrics[k].append(v[-1])
    st.session_state.tab1_global_rewards.append(sum(st.session_state.tab1_local_rewards))


def reset_tab1():
    st.session_state.tab1_running = False
    reset_local_info_tab1()
    st.session_state.tab1_local_features = []
    st.session_state.tab1_global_metrics = {k: [] for k in metric_names.keys()}
    st.session_state.tab1_global_rewards = []
    st.session_state.tab1_episode = 0
    st.session_state.tab1_step = 0


# Tab 2
def reset_local_info_tab2():
    if ('fixed_features' in st.session_state) and len(st.session_state.fixed_features) > 0:
        dummy_env = FeatureSelectionEnv(
            df_features=st.session_state.df_all_feats,
            n_features=math.ceil(len(st.session_state.df_all_feats.columns) * 0.75),
            clustering_model=st.session_state.model,
            normalized_scorers=normalized_scorers,
            early_stopping=PlateauEarlyStopping(
                patience=st.session_state.patience,
                plateau_patience=st.session_state.plateau_patience,
                threshold=st.session_state.threshold
            ),
            fixed_features=st.session_state.fixed_features,
        )
        obs = dummy_env.reset()
        info = dummy_env._get_info()
        real_scores = info['real_scores']
        st.session_state.tab2_local_metrics = {k: [real_scores[k][-1]] for k in metric_names.keys()}
    else:
        st.session_state.tab2_local_metrics = {k: [np.nan] for k in metric_names.keys()}
    st.session_state.tab2_local_rewards = []


def update_global_info_tab2():
    for k, v in st.session_state.tab2_local_metrics.items():
        st.session_state.tab2_global_metrics[k].append(v[-1])
    st.session_state.tab2_global_rewards.append(sum(st.session_state.tab2_local_rewards))


def reset_tab2():
    st.session_state.tab2_running = False
    reset_local_info_tab2()
    if ('fixed_features' in st.session_state) and len(st.session_state.fixed_features) > 0:
        st.session_state.tab2_local_features = copy.deepcopy(st.session_state.fixed_features)
    else:
        st.session_state.tab2_local_features = []
    st.session_state.tab2_global_metrics = {k: [] for k in metric_names.keys()}
    st.session_state.tab2_global_rewards = []
    st.session_state.tab2_episode = 0
    st.session_state.tab2_step = 0


def reset_tabs():
    st.session_state.fixed_features = []
    reset_tab1()
    reset_tab2()



st.set_page_config(
    page_title='Time2ReinforcedFeat',
    page_icon='📈',
)

customized_button = st.markdown('''
    <style >
    .stDownloadButton, div.stButton {text-align:center}
        }
    </style>''', unsafe_allow_html=True
)

with st.sidebar:

    selected_dataset = st.selectbox(
        'Select a dataset',
        tuple(sorted(dataset for dataset in datasets.keys())),
        on_change=reset_tabs,
    )

    if ('dataset' not in st.session_state) or (st.session_state.dataset != selected_dataset):
        st.session_state.dataset = selected_dataset
        st.session_state.ts_list, st.session_state.y_true, st.session_state.df_all_feats, st.session_state.model = open_dataset(selected_dataset)
        possible_labels = sorted(set(st.session_state.y_true))
        st.session_state.y_true_colors = {label: sns.color_palette("husl", n_colors=len(possible_labels)).as_hex()[i] for i, label in enumerate(possible_labels)}

    selected_agent = st.selectbox(
        'Select an RL agent',
        tuple(agent for agent in rl_agents.keys()),
        on_change=reset_tabs,
    )

    episodes_for_training = st.slider('Number of episodes to train the RL agent', 1, 100, 30)

    with st.expander('RL Hyperparameters', expanded=False):
        parameters = {}
        for parameter, details in rl_agents[selected_agent]['parameters'].items():
            parameter_label = 'Select a value for ' + details[0]
            parameter_min_value = details[1]
            parameter_max_value = details[2]
            parameter_default_value = details[3]
            parameter_information = details[4]
            if parameter != 'alpha':
                parameters[parameter] = st.slider(
                    parameter_label,
                    parameter_min_value,
                    parameter_max_value,
                    parameter_default_value,
                    help=parameter_information
                )
            else:
                parameters[parameter] = st.number_input(
                    label=parameter_label,
                    min_value=parameter_min_value,
                    step=0.00001,
                    max_value=parameter_max_value,
                    value=parameter_default_value,
                    format='%f',
                    help=parameter_information
                )
    
    with st.expander('Early Stopping', expanded=False):
        st.session_state.patience = st.slider(
            'Patience',
            1,
            50,
            20,
        )

        st.session_state.plateau_patience = st.slider(
            'Plateau Patience',
            1,
            50,
            20,
        )

        st.session_state.threshold = st.slider(
            'Threshold',
            0.0,
            1.0,
            0.03,
        )
    

st.markdown('# RL-T2F')
st.markdown('#### Reinforcement Learning-Based Feature Selection for Multivariate Time Series Clustering')


def display_metric(local_metrics):
        if len(local_metrics) > 0:
            metric = local_metrics[-1]
        else:
            metric = np.nan
        
        if metric is None or math.isnan(metric):
            return '-'
        else:
            return round(metric, 4)


tab1, tab2, tab3 = st.tabs(["Automatic Feature Selection", "Fixed Features at Initialization", "Ground Truth"])


with tab1:
    if 'tab1_running' not in st.session_state:
        reset_tab1()


    def start_tab1():
        reset_tab1()
        st.session_state.tab1_running = True
        st.session_state.tab1_env = FeatureSelectionEnv(
            df_features=st.session_state.df_all_feats,
            n_features=math.ceil(len(st.session_state.df_all_feats.columns) * 0.75),
            clustering_model=st.session_state.model,
            normalized_scorers=normalized_scorers,
            early_stopping=PlateauEarlyStopping(
                patience=st.session_state.patience,
                plateau_patience=st.session_state.plateau_patience,
                threshold=st.session_state.threshold
            ),
        )
        st.session_state.tab1_obs = st.session_state.tab1_env.reset()
        st.session_state.tab1_agent = get_agent(selected_agent, parameters, st.session_state.tab1_env, st.session_state.tab1_obs)
        st.session_state.tab1_done = False

    
    def new_episode_tab1():
        st.session_state.tab1_obs = st.session_state.tab1_env.reset()
        update_global_info_tab1()
        reset_local_info_tab1()
        st.session_state.tab1_episode += 1
        st.session_state.tab1_step  = 0
        st.session_state.tab1_done = False


    def run_step_tab1():
        if not st.session_state.tab1_running:
            start_tab1()
        
        if st.session_state.tab1_done:
            new_episode_tab1()
        
        action_masks = get_action_masks(st.session_state.tab1_env)
        action = get_action(st.session_state.tab1_agent, st.session_state.tab1_obs, action_masks, st.session_state.tab1_episode)
        
        next_obs, reward, st.session_state.tab1_done, info = st.session_state.tab1_env.step(action)
        st.session_state.tab1_local_rewards.append(reward)

        features_selected = info['features_selected']
        st.session_state.tab1_local_features = copy.deepcopy(features_selected)

        real_scores = info['real_scores']
        for k in st.session_state.tab1_local_metrics.keys():
            st.session_state.tab1_local_metrics[k].append(real_scores[k][-1])

        learn(st.session_state.tab1_agent, st.session_state.tab1_obs, action, reward, next_obs, st.session_state.tab1_done, action_masks)
        st.session_state.tab1_obs = next_obs
        st.session_state.tab1_step += 1
        

    def run_episode_tab1():
        if not st.session_state.tab1_running:
            start_tab1()
        if st.session_state.tab1_done:
            new_episode_tab1()
        while not st.session_state.tab1_done:
            run_step_tab1()
    

    def train_agent_tab1():
        if not st.session_state.tab1_running:
            start_tab1()
        if st.session_state.tab1_done:
            new_episode_tab1()
        for i in range(episodes_for_training):
            while not st.session_state.tab1_done:
                run_step_tab1()
            if i < episodes_for_training - 1:
                new_episode_tab1()


    tab1_col1, tab1_col2, tab1_col3, tab1_col4 = st.columns(4)
    tab1_col1.button('Reset', type='primary', on_click=reset_tab1, disabled=(not st.session_state.tab1_running))
    tab1_col2.button('Train agent', on_click=train_agent_tab1)
    tab1_col3.button('Run episode', on_click=run_episode_tab1)
    tab1_col4.button('Run step', on_click=run_step_tab1)

    st.subheader(f'Episode {st.session_state.tab1_episode}, Step {st.session_state.tab1_step}')


    # RL Performance (current episode)
    with st.expander('RL Performance (current episode)', expanded=False):
        tab1_metric_cols = st.columns(len(metric_names))
        for i, k in enumerate(metric_names.keys()):
            tab1_metric_cols[i].metric(metric_names[k], display_metric(st.session_state.tab1_local_metrics[k]))
        
        tab1_current_episode_fig = make_subplots(
            rows=math.ceil(len(metric_names)/COLS_IN_PLOT), cols=COLS_IN_PLOT,
            subplot_titles=tuple(k for k in metric_names.values()))

        tab1_current_episode_fig_row = 1
        tab1_current_episode_fig_col = 1
        for k in metric_names.keys():
            tab1_current_episode_fig.add_trace(
                go.Scatter(
                    x=[i for i in range(len(st.session_state.tab1_local_metrics[k]))],
                    y=st.session_state.tab1_local_metrics[k]
                ),
                row=tab1_current_episode_fig_row,
                col=tab1_current_episode_fig_col
            )

            tab1_current_episode_fig.update_xaxes(
                title_text="Step",
                row=tab1_current_episode_fig_row,
                col=tab1_current_episode_fig_col
            )

            tab1_current_episode_fig_col += 1
            if tab1_current_episode_fig_col > COLS_IN_PLOT:
                tab1_current_episode_fig_col = 1
                tab1_current_episode_fig_row += 1
        
        tab1_current_episode_fig.update_layout(height=500, width=700,
                        title_text="Unsupervised Metrics", showlegend=False)

        st.plotly_chart(tab1_current_episode_fig, theme="streamlit", use_container_width=True)

        tab1_local_rewards_df = pd.DataFrame({'Cumulated Reward': st.session_state.tab1_local_rewards}).cumsum()
        tab1_local_rewards_df['Step'] = [i for i in range(len(st.session_state.tab1_local_rewards))]
        tab1_local_rewards_fig = px.line(tab1_local_rewards_df, x='Step', y='Cumulated Reward', markers=True, title='Cumulated Reward')
        st.plotly_chart(tab1_local_rewards_fig, theme="streamlit", use_container_width=True)
    

    # RL Performance (global)
    with st.expander('RL Performance (global)', expanded=False):
        tab1_global_fig = make_subplots(
            rows=math.ceil(len(metric_names)/COLS_IN_PLOT), cols=COLS_IN_PLOT,
            subplot_titles=tuple(k for k in metric_names.values()))
        
        tab1_global_fig_row = 1
        tab1_global_fig_col = 1

        for k in metric_names.keys():
            tab1_global_fig.add_trace(
                go.Scatter(
                    x=[i for i in range(len(st.session_state.tab1_global_metrics[k]))],
                    y=st.session_state.tab1_global_metrics[k]
                ),
                row=tab1_global_fig_row,
                col=tab1_global_fig_col
            )

            tab1_global_fig.update_xaxes(
                title_text="Episode",
                row=tab1_global_fig_row,
                col=tab1_global_fig_col
            )

            tab1_global_fig_col += 1
            if tab1_global_fig_col > COLS_IN_PLOT:
                tab1_global_fig_col = 1
                tab1_global_fig_row += 1
        
        tab1_global_fig.update_layout(
            height=500, width=700,
            title_text="Unsupervised Metrics", showlegend=False
        )

        st.plotly_chart(tab1_global_fig, theme="streamlit", use_container_width=True)

        tab1_global_rewards_df = pd.DataFrame({'Total Reward per Episode': st.session_state.tab1_global_rewards})
        tab1_global_rewards_df['Episode'] = [i for i in range(len(st.session_state.tab1_global_rewards))]
        tab1_global_rewards_fig = px.line(tab1_global_rewards_df, x='Episode', y='Total Reward per Episode', markers=True, title='Total Reward per Episode')
        st.plotly_chart(tab1_global_rewards_fig, theme="streamlit", use_container_width=True)
    
    # Selected Features
    with st.expander('Selected Features', expanded=False):
        if len(st.session_state.tab1_local_features) > 0:
            st.markdown('**Feature names**')
            st.table(pd.DataFrame({'Feature': st.session_state.tab1_local_features}))
            features_df = st.session_state.df_all_feats[st.session_state.tab1_local_features]
            st.markdown('**Feature values**')
            st.dataframe(features_df)
        else:
            st.text('No feature selected')
    
    # Clusters
    with st.expander('Clusters', expanded=False):
        if len(st.session_state.tab1_local_features) > 0:
            tab1_y_pred = st.session_state.model.fit_predict(st.session_state.df_all_feats[st.session_state.tab1_local_features])
            AMI = adjusted_mutual_info_score(tab1_y_pred, st.session_state.y_true)
            st.metric('AMI', round(AMI, 4))

            tab1_selected_signal = st.selectbox(
                'Select the dimension of the time series',
                tuple(f'Signal {i+1}' for i in range(len(st.session_state.ts_list[0][0])))
            )
            signal = int(tab1_selected_signal[len('Signal '):]) - 1

            cluster_labels = list(sorted(set(st.session_state.y_true)))

            contingency = contingency_matrix(st.session_state.y_true, tab1_y_pred)
            already_assigned = []
            pred_to_true = {}

            real_labels = cluster_labels
            prediction_labels = list(sorted(set(tab1_y_pred)))

            for pred_label in prediction_labels:
                c = contingency[:, pred_label]
                m = np.zeros(c.size, dtype=bool)
                m[already_assigned] = True
                a = np.ma.array(c, mask=m)
                true_label_index = np.argmax(a)
                true_label = real_labels[true_label_index]
                pred_to_true[pred_label] = true_label
                already_assigned.append(true_label_index)

            tab1_clusters_fig = make_subplots(
                rows=math.ceil(len(cluster_labels)/COLS_IN_PLOT), cols=COLS_IN_PLOT,
                subplot_titles=tuple(cluster_label_to_string(label) for label in cluster_labels)
            )
        
            tab1_clusters_fig_row = 1
            tab1_clusters_fig_col = 1

            for label in cluster_labels:
                current_cluster_df = None
                for time_series_id, predicted_cluster_id in enumerate(tab1_y_pred):
                    predicted_label = pred_to_true[predicted_cluster_id]
                    if str(predicted_label) == str(label):
                        real_label = st.session_state.y_true[time_series_id]
                        time_series = [ts[signal] for ts in st.session_state.ts_list[time_series_id]]
                        tab1_clusters_fig.add_trace(
                            go.Scatter(
                                x=[i for i in range(len(time_series))],
                                y=time_series,
                                line={'color': st.session_state.y_true_colors[real_label]},
                                name=cluster_label_to_string(real_label)
                            ),
                            row=tab1_clusters_fig_row,
                            col=tab1_clusters_fig_col,
                        )

                tab1_clusters_fig.update_xaxes(
                    # title_text=f'Time',
                    row=tab1_clusters_fig_row,
                    col=tab1_clusters_fig_col
                )

                tab1_clusters_fig_col += 1
                if tab1_clusters_fig_col > COLS_IN_PLOT:
                    tab1_clusters_fig_col = 1
                    tab1_clusters_fig_row += 1
            
            tab1_clusters_fig.update_layout(
                height=math.ceil(len(cluster_labels)/COLS_IN_PLOT) * 250, width=700,
                title_text="Predicted Clusters", showlegend=False
            )

            st.plotly_chart(tab1_clusters_fig, theme="streamlit", use_container_width=True)

        else:
            st.text('No feature selected')


with tab2:
    if 'tab2_running' not in st.session_state:
        reset_tab2()
        st.session_state.tab2_running = False
    
    if 'change_fixed_features' not in st.session_state:
        st.session_state.change_fixed_features = False
    
    def change_fixed_features():
        if not st.session_state.tab2_running:
            st.session_state.change_fixed_features = True
        

    st.session_state.fixed_features = st.multiselect(
        'Fixed features (optional)',
        st.session_state.df_all_feats.columns,
        [],
        disabled=st.session_state.tab2_running,
        on_change=change_fixed_features,
    )

    if st.session_state.change_fixed_features:
        if ('fixed_features' in st.session_state) and len(st.session_state.fixed_features) > 0:
            dummy_env = FeatureSelectionEnv(
                df_features=st.session_state.df_all_feats,
                n_features=math.ceil(len(st.session_state.df_all_feats.columns) * 0.75),
                clustering_model=st.session_state.model,
                normalized_scorers=normalized_scorers,
                early_stopping=PlateauEarlyStopping(
                    patience=st.session_state.patience,
                    plateau_patience=st.session_state.plateau_patience,
                    threshold=st.session_state.threshold
                ),
                fixed_features=st.session_state.fixed_features,
            )
            obs = dummy_env.reset()
            info = dummy_env._get_info()
            real_scores = info['real_scores']
            st.session_state.tab2_local_metrics = {k: [real_scores[k][-1]] for k in metric_names.keys()}
        else:
            st.session_state.tab2_local_metrics = {k: [np.nan] for k in metric_names.keys()}
        st.session_state.tab2_local_features = st.session_state.fixed_features
        st.session_state.change_fixed_features = False


    def start_tab2():
        reset_tab2()
        st.session_state.tab2_running = True
        st.session_state.tab2_env = FeatureSelectionEnv(
            df_features=st.session_state.df_all_feats,
            n_features=math.ceil(len(st.session_state.df_all_feats.columns) * 0.75),
            clustering_model=st.session_state.model,
            normalized_scorers=normalized_scorers,
            early_stopping=PlateauEarlyStopping(
                patience=st.session_state.patience,
                plateau_patience=st.session_state.plateau_patience,
                threshold=st.session_state.threshold
            ),
            fixed_features=st.session_state.fixed_features,
        )
        st.session_state.tab2_obs = st.session_state.tab2_env.reset()
        st.session_state.tab2_agent = get_agent(selected_agent, parameters, st.session_state.tab2_env, st.session_state.tab2_obs)
        st.session_state.tab2_done = False

    
    def new_episode_tab2():
        st.session_state.tab2_obs = st.session_state.tab2_env.reset()
        update_global_info_tab2()
        reset_local_info_tab2()
        st.session_state.tab2_episode += 1
        st.session_state.tab2_step  = 0
        st.session_state.tab2_done = False


    def run_step_tab2():
        if not st.session_state.tab2_running:
            start_tab2()
        
        if st.session_state.tab2_done:
            new_episode_tab2()
        
        action_masks = get_action_masks(st.session_state.tab2_env)
        action = get_action(st.session_state.tab2_agent, st.session_state.tab2_obs, action_masks, st.session_state.tab2_episode)
        
        next_obs, reward, st.session_state.tab2_done, info = st.session_state.tab2_env.step(action)
        st.session_state.tab2_local_rewards.append(reward)

        features_selected = info['features_selected']
        st.session_state.tab2_local_features = copy.deepcopy(features_selected)

        real_scores = info['real_scores']
        for k in st.session_state.tab2_local_metrics.keys():
            st.session_state.tab2_local_metrics[k].append(real_scores[k][-1])

        learn(st.session_state.tab2_agent, st.session_state.tab2_obs, action, reward, next_obs, st.session_state.tab2_done, action_masks)
        st.session_state.tab2_obs = next_obs
        st.session_state.tab2_step += 1
        

    def run_episode_tab2():
        if not st.session_state.tab2_running:
            start_tab2()
        if st.session_state.tab2_done:
            new_episode_tab2()
        while not st.session_state.tab2_done:
            run_step_tab2()
    

    def train_agent_tab2():
        if not st.session_state.tab2_running:
            start_tab2()
        if st.session_state.tab2_done:
            new_episode_tab2()
        for i in range(episodes_for_training):
            while not st.session_state.tab2_done:
                run_step_tab2()
            if i < episodes_for_training - 1:
                new_episode_tab2()


    tab2_col1, tab2_col2, tab2_col3, tab2_col4 = st.columns(4)
    tab2_col1.button('Reset', type='primary', on_click=reset_tab2, key='Reset 2', disabled=(not st.session_state.tab2_running))
    tab2_col2.button('Train agent', on_click=train_agent_tab2, key='Train agent 2', disabled=(len(st.session_state.fixed_features) == len(st.session_state.df_all_feats.columns)))
    tab2_col3.button('Run episode', on_click=run_episode_tab2, key='Run episode 2', disabled=(len(st.session_state.fixed_features) == len(st.session_state.df_all_feats.columns)))
    tab2_col4.button('Run step', on_click=run_step_tab2, key='Run step 2', disabled=(len(st.session_state.fixed_features) == len(st.session_state.df_all_feats.columns)))

    st.subheader(f'Episode {st.session_state.tab2_episode}, Step {st.session_state.tab2_step}')


    # RL Performance (current episode)
    with st.expander('RL Performance (current episode)', expanded=False):
        tab2_metric_cols = st.columns(len(metric_names))
        for i, k in enumerate(metric_names.keys()):
            tab2_metric_cols[i].metric(metric_names[k], display_metric(st.session_state.tab2_local_metrics[k]))
        
        tab2_current_episode_fig = make_subplots(
            rows=math.ceil(len(metric_names)/COLS_IN_PLOT), cols=COLS_IN_PLOT,
            subplot_titles=tuple(k for k in metric_names.values()))

        tab2_current_episode_fig_row = 1
        tab2_current_episode_fig_col = 1
        for k in metric_names.keys():
            tab2_current_episode_fig.add_trace(
                go.Scatter(
                    x=[i for i in range(len(st.session_state.tab2_local_metrics[k]))],
                    y=st.session_state.tab2_local_metrics[k]
                ),
                row=tab2_current_episode_fig_row,
                col=tab2_current_episode_fig_col
            )

            tab2_current_episode_fig.update_xaxes(
                title_text="Step",
                row=tab2_current_episode_fig_row,
                col=tab2_current_episode_fig_col
            )

            tab2_current_episode_fig_col += 1
            if tab2_current_episode_fig_col > COLS_IN_PLOT:
                tab2_current_episode_fig_col = 1
                tab2_current_episode_fig_row += 1
        
        tab2_current_episode_fig.update_layout(height=500, width=700,
                        title_text="Unsupervised Metrics", showlegend=False)

        st.plotly_chart(tab2_current_episode_fig, theme="streamlit", use_container_width=True)

        tab2_local_rewards_df = pd.DataFrame({'Cumulated Reward': st.session_state.tab2_local_rewards}).cumsum()
        tab2_local_rewards_df['Step'] = [i for i in range(len(st.session_state.tab2_local_rewards))]
        tab2_local_rewards_fig = px.line(tab2_local_rewards_df, x='Step', y='Cumulated Reward', markers=True, title='Cumulated Reward')
        st.plotly_chart(tab2_local_rewards_fig, theme="streamlit", use_container_width=True)
    

    # RL Performance (global)
    with st.expander('RL Performance (global)', expanded=False):
        tab2_global_fig = make_subplots(
            rows=math.ceil(len(metric_names)/COLS_IN_PLOT), cols=COLS_IN_PLOT,
            subplot_titles=tuple(k for k in metric_names.values()))
        
        tab2_global_fig_row = 1
        tab2_global_fig_col = 1

        for k in metric_names.keys():
            tab2_global_fig.add_trace(
                go.Scatter(
                    x=[i for i in range(len(st.session_state.tab2_global_metrics[k]))],
                    y=st.session_state.tab2_global_metrics[k]
                ),
                row=tab2_global_fig_row,
                col=tab2_global_fig_col
            )

            tab2_global_fig.update_xaxes(
                title_text="Episode",
                row=tab2_global_fig_row,
                col=tab2_global_fig_col
            )

            tab2_global_fig_col += 1
            if tab2_global_fig_col > COLS_IN_PLOT:
                tab2_global_fig_col = 1
                tab2_global_fig_row += 1
        
        tab2_global_fig.update_layout(
            height=500, width=700,
            title_text="Unsupervised Metrics", showlegend=False
        )

        st.plotly_chart(tab2_global_fig, theme="streamlit", use_container_width=True)

        tab2_global_rewards_df = pd.DataFrame({'Total Reward per Episode': st.session_state.tab2_global_rewards})
        tab2_global_rewards_df['Episode'] = [i for i in range(len(st.session_state.tab2_global_rewards))]
        tab2_global_rewards_fig = px.line(tab2_global_rewards_df, x='Episode', y='Total Reward per Episode', markers=True, title='Total Reward per Episode')
        st.plotly_chart(tab2_global_rewards_fig, theme="streamlit", use_container_width=True)
    
    # Selected Features
    with st.expander('Selected Features', expanded=False):
        if len(st.session_state.tab2_local_features) > 0:
            st.markdown('**Feature names**')
            def color_fixed_features(x):
                if x in st.session_state.fixed_features:
                    # return f"background: #ff2b2b; color: white;" 
                    return f'background: #FFA07A'
                return ''
            st.table(pd.DataFrame({'Feature': st.session_state.tab2_local_features}).style.applymap(color_fixed_features))
            features_df = st.session_state.df_all_feats[st.session_state.tab2_local_features]
            st.markdown('**Feature values**')
            st.dataframe(features_df)
        else:
            st.text('No feature selected')
    
    # Clusters
    with st.expander('Clusters', expanded=False):
        if len(st.session_state.tab2_local_features) > 0:
            tab2_y_pred = st.session_state.model.fit_predict(st.session_state.df_all_feats[st.session_state.tab2_local_features])
            AMI = adjusted_mutual_info_score(tab2_y_pred, st.session_state.y_true)
            st.metric('AMI', round(AMI, 4))

            tab2_selected_signal = st.selectbox(
                'Select the dimension of the time series',
                tuple(f'Signal {i+1}' for i in range(len(st.session_state.ts_list[0][0]))),
                key = 'Select the dimension of the time series tab2'
            )
            signal = int(tab2_selected_signal[len('Signal '):]) - 1

            cluster_labels = list(sorted(set(st.session_state.y_true)))

            contingency = contingency_matrix(st.session_state.y_true, tab2_y_pred)
            already_assigned = []
            pred_to_true = {}

            real_labels = cluster_labels
            prediction_labels = list(sorted(set(tab2_y_pred)))

            for pred_label in prediction_labels:
                c = contingency[:, pred_label]
                m = np.zeros(c.size, dtype=bool)
                m[already_assigned] = True
                a = np.ma.array(c, mask=m)
                true_label_index = np.argmax(a)
                true_label = real_labels[true_label_index]
                pred_to_true[pred_label] = true_label
                already_assigned.append(true_label_index)

            tab2_clusters_fig = make_subplots(
                rows=math.ceil(len(cluster_labels)/COLS_IN_PLOT), cols=COLS_IN_PLOT,
                subplot_titles=tuple(cluster_label_to_string(label) for label in cluster_labels)
            )
        
            tab2_clusters_fig_row = 1
            tab2_clusters_fig_col = 1

            for label in cluster_labels:
                current_cluster_df = None
                for time_series_id, predicted_cluster_id in enumerate(tab2_y_pred):
                    predicted_label = pred_to_true[predicted_cluster_id]
                    if str(predicted_label) == str(label):
                        real_label = st.session_state.y_true[time_series_id]
                        time_series = [ts[signal] for ts in st.session_state.ts_list[time_series_id]]
                        tab2_clusters_fig.add_trace(
                            go.Scatter(
                                x=[i for i in range(len(time_series))],
                                y=time_series,
                                line={'color': st.session_state.y_true_colors[real_label]},
                                name=cluster_label_to_string(real_label)
                            ),
                            row=tab2_clusters_fig_row,
                            col=tab2_clusters_fig_col,
                        )

                tab2_clusters_fig.update_xaxes(
                    # title_text=f'Time',
                    row=tab2_clusters_fig_row,
                    col=tab2_clusters_fig_col
                )

                tab2_clusters_fig_col += 1
                if tab2_clusters_fig_col > COLS_IN_PLOT:
                    tab2_clusters_fig_col = 1
                    tab2_clusters_fig_row += 1
            
            tab2_clusters_fig.update_layout(
                height=math.ceil(len(cluster_labels)/COLS_IN_PLOT) * 250, width=700,
                title_text="Predicted Clusters", showlegend=False
            )

            st.plotly_chart(tab2_clusters_fig, theme="streamlit", use_container_width=True)

        else:
            st.text('No feature selected')


with tab3:
    tab3_selected_signal = st.selectbox(
        'Select the dimension of the time series',
        tuple(f'Signal {i+1}' for i in range(len(st.session_state.ts_list[0][0]))),
        key = 'Select the dimension of the time series tab3'
    )
    signal = int(tab3_selected_signal[len('Signal '):]) - 1

    cluster_labels = list(sorted(set(st.session_state.y_true)))

    tab3_clusters_fig = make_subplots(
        rows=math.ceil(len(cluster_labels)/COLS_IN_PLOT), cols=COLS_IN_PLOT,
        subplot_titles=tuple(cluster_label_to_string(label) for label in cluster_labels)
    )

    tab3_clusters_fig_row = 1
    tab3_clusters_fig_col = 1

    for label in cluster_labels:
        current_cluster_df = None
        for time_series_id, cluster_label in enumerate(st.session_state.y_true):
            if str(cluster_label) == str(label):
                time_series = [ts[signal] for ts in st.session_state.ts_list[time_series_id]]
                tab3_clusters_fig.add_trace(
                    go.Scatter(
                        x=[i for i in range(len(time_series))],
                        y=time_series,
                        line={'color': st.session_state.y_true_colors[cluster_label]},
                        name=cluster_label_to_string(cluster_label)
                    ),
                    row=tab3_clusters_fig_row,
                    col=tab3_clusters_fig_col,
                )

        tab3_clusters_fig.update_xaxes(
            # title_text=f'Time',
            row=tab3_clusters_fig_row,
            col=tab3_clusters_fig_col
        )

        tab3_clusters_fig_col += 1
        if tab3_clusters_fig_col > COLS_IN_PLOT:
            tab3_clusters_fig_col = 1
            tab3_clusters_fig_row += 1

    tab3_clusters_fig.update_layout(
        height=math.ceil(len(cluster_labels)/COLS_IN_PLOT) * 250, width=700,
        title_text="Predicted Clusters", showlegend=False
    )

    st.plotly_chart(tab3_clusters_fig, theme="streamlit", use_container_width=True)
