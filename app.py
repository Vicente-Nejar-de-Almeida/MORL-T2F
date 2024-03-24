import math
import copy
import streamlit as st
import numpy as np
import pandas as pd

from sb3_contrib.common.maskable.utils import get_action_masks

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import adjusted_mutual_info_score

from early_stopping import PlateauEarlyStopping
from rl.environments.feature_selection_env import FeatureSelectionEnv

from utils import *


if 'reset_rl_state' not in st.session_state:
    st.session_state.reset_rl_state = False


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

selected_dataset = st.sidebar.selectbox(
    'Select a dataset',
    tuple(sorted(dataset for dataset in datasets.keys()))
)

if ('dataset' not in st.session_state) or (st.session_state.dataset != selected_dataset):
    st.session_state.dataset = selected_dataset
    st.session_state.df_all_feats, st.session_state.y_true, st.session_state.model = open_dataset(selected_dataset)

selected_agent = st.sidebar.selectbox(
    'Select an RL agent',
    tuple(agent for agent in rl_agents.keys())
)

st.sidebar.header('Hyperparameters')

parameters = {}
for parameter, details in rl_agents[selected_agent]['parameters'].items():
    parameter_label = 'Select a value for ' + details[0]
    parameter_min_value = details[1]
    parameter_max_value = details[2]
    parameter_default_value = details[3]
    if parameter != 'alpha':
        parameters[parameter] = st.sidebar.slider(
            parameter_label,
            parameter_min_value,
            parameter_max_value,
            parameter_default_value
        )
    else:
        parameters[parameter] = st.sidebar.number_input(
            label=parameter_label,
            min_value=parameter_min_value,
            step=0.00001,
            max_value=parameter_max_value,
            value=parameter_default_value,
            format='%f'
        )

st.title('Time2ReinforcedFeat')

if 'episode' not in st.session_state:
    st.session_state.episode = None


def start():
    st.session_state.episode = 0
    st.session_state.step = 0
    st.session_state.reset_rl_state = True


def reset():
    st.session_state.episode = None
    st.session_state.step = 0


if 'fixed_features' not in st.session_state:
    st.session_state.fixed_features = []


if st.session_state.episode is None:
    st.session_state.fixed_features = st.multiselect(
        'Fixed features (optional)',
        st.session_state.df_all_feats.columns,
        []
    )

    st.header('Early stopping')
    
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

    st.button('Start', type='secondary', on_click=start)
else:
    
    def reset_episode():
        
        if len(st.session_state.local_rewards) > 0:
            st.session_state.global_rewards.append(sum(st.session_state.local_rewards))

        if len(st.session_state.local_silhouette) > 0:
            st.session_state.global_silhouette.append(st.session_state.local_silhouette[-1])
            st.session_state.global_calinski_harabasz.append(st.session_state.local_calinski_harabasz[-1])
            st.session_state.global_davies_bouldin.append(st.session_state.local_davies_bouldin[-1])
            st.session_state.global_dunn_index.append(st.session_state.local_dunn_index[-1])

        st.session_state.episode += 1
        st.session_state.step = 0
        st.session_state.done = False
        st.session_state.obs = st.session_state.env.reset()

        info = st.session_state.env._get_info()
        real_scores = info['real_scores']
        if len(real_scores['silhouette']) > 0:
            st.session_state.local_silhouette = [real_scores['silhouette'][0]]
            st.session_state.local_calinski_harabasz = [real_scores['calinski_harabasz'][0]]
            st.session_state.local_davies_bouldin = [real_scores['davies_bouldin'][-1]]
            st.session_state.local_dunn_index = [real_scores['dunn_index'][-1]]
        else:
            st.session_state.local_silhouette = [np.nan]
            st.session_state.local_calinski_harabasz = [np.nan]
            st.session_state.local_davies_bouldin = [np.nan]
            st.session_state.local_dunn_index = [np.nan]

        st.session_state.local_rewards = []
        st.session_state.local_features = []

    if st.session_state.reset_rl_state:
        st.session_state.run_training = False
        st.session_state.run_episode = False
        st.session_state.next_episode = False
        st.session_state.env = FeatureSelectionEnv(
            df_features=st.session_state.df_all_feats,
            n_features=50,
            clustering_model=st.session_state.model,
            normalized_scorers=normalized_scorers,
            early_stopping=PlateauEarlyStopping(
                patience=st.session_state.patience,
                plateau_patience=st.session_state.plateau_patience,
                threshold=st.session_state.threshold
            ),
            fixed_features=st.session_state.fixed_features,
        )
        st.session_state.obs = st.session_state.env.reset()
        st.session_state.agent = get_agent(selected_agent, parameters, st.session_state.env, st.session_state.obs)
        
        st.session_state.local_silhouette = []
        st.session_state.local_calinski_harabasz = []
        st.session_state.local_davies_bouldin = []
        st.session_state.local_dunn_index = []
        st.session_state.local_rewards = []
        st.session_state.local_features = []
        reset_episode()
        st.session_state.global_rewards = []
        st.session_state.global_silhouette = []
        st.session_state.global_calinski_harabasz = []
        st.session_state.global_davies_bouldin = []
        st.session_state.global_dunn_index = []
        st.session_state.reset_rl_state = False
        st.session_state.disable_buttons = False


    def run_episode():
        st.session_state.disable_buttons = True
        st.session_state.run_training = True
        st.session_state.run_episode = True
    
    def run_step():
        st.session_state.disable_buttons = True
        st.session_state.run_training = True
    
    def next_episode():
        st.session_state.disable_buttons = True
        st.session_state.next_episode = True

    
    def step_rl():
        if st.session_state.done:
            reset_episode()

        action_masks = get_action_masks(st.session_state.env)
        action = get_action(st.session_state.agent, st.session_state.obs, action_masks, st.session_state.episode)
        next_obs, reward, st.session_state.done, info = st.session_state.env.step(action)
        st.session_state.local_rewards.append(reward)
        features_selected = info['features_selected']
        st.session_state.local_features = copy.deepcopy(features_selected)
        real_scores = info['real_scores']
        
        
        st.session_state.local_silhouette.append(real_scores['silhouette'][-1])
        st.session_state.local_calinski_harabasz.append(real_scores['calinski_harabasz'][-1])
        st.session_state.local_davies_bouldin.append(real_scores['davies_bouldin'][-1])
        st.session_state.local_dunn_index.append(real_scores['dunn_index'][-1])

        learn(st.session_state.agent, st.session_state.obs, action, reward, next_obs, st.session_state.done, action_masks)
        st.session_state.obs = next_obs
        st.session_state.step += 1

    button_col1, button_col2, button_col3, button_col4 = st.columns(4)
    button_col1.button('Reset', type='primary', on_click=reset)
    button_col2.button('Run episode', on_click=run_episode, disabled=st.session_state.disable_buttons)
    button_col3.button('Next step', on_click=run_step, disabled=st.session_state.disable_buttons)
    button_col4.button('Next episode', on_click=next_episode, disabled=(st.session_state.step == 0))

    if st.session_state.run_training:
        if st.session_state.run_episode:
            if st.session_state.done:
                reset_episode()
            while not st.session_state.done:
                step_rl()
            st.session_state.run_episode = False
        else:
            step_rl()
        
        st.session_state.run_training = False
        st.session_state.disable_buttons = False
        st.rerun()
    
    if st.session_state.next_episode:
        reset_episode()
        st.session_state.next_episode = False
        st.session_state.disable_buttons = False
        st.rerun()

    st.subheader(f'Episode {st.session_state.episode}, Step {st.session_state.step}')


    def display_metric(metric):
        if math.isnan(metric):
            return '-'
        else:
            return round(metric, 4)

    with st.expander('RL Performance (current episode)', expanded=True):
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric('Silhouette Coefficient', display_metric(st.session_state.local_silhouette[-1]))
        metric_col2.metric('Calinski and Harabasz', display_metric(st.session_state.local_calinski_harabasz[-1]))
        metric_col3.metric('Davies-Bouldin', display_metric(st.session_state.local_davies_bouldin[-1]))
        metric_col4.metric('Dunn index', display_metric(st.session_state.local_dunn_index[-1]))

        current_episode_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Silhouette Coefficient", "Calinski and Harabasz", "Davies-Bouldin", "Dunn index"))

        current_episode_fig.add_trace(go.Scatter(x=[i for i in range(len(st.session_state.local_silhouette))], y=st.session_state.local_silhouette),
                    row=1, col=1)

        current_episode_fig.add_trace(go.Scatter(x=[i for i in range(len(st.session_state.local_calinski_harabasz))], y=st.session_state.local_calinski_harabasz),
                    row=1, col=2)

        current_episode_fig.add_trace(go.Scatter(x=[i for i in range(len(st.session_state.local_davies_bouldin))], y=st.session_state.local_davies_bouldin),
                    row=2, col=1)

        current_episode_fig.add_trace(go.Scatter(x=[i for i in range(len(st.session_state.local_dunn_index))], y=st.session_state.local_dunn_index),
                    row=2, col=2)

        current_episode_fig.update_layout(height=500, width=700,
                        title_text="Unsupervised Metrics", showlegend=False)

        # Update xaxis properties
        current_episode_fig.update_xaxes(title_text="Step", row=1, col=1)
        current_episode_fig.update_xaxes(title_text="Step", row=1, col=2)
        current_episode_fig.update_xaxes(title_text="Step", row=2, col=1)
        current_episode_fig.update_xaxes(title_text="Step", row=2, col=2)

        st.plotly_chart(current_episode_fig, theme="streamlit", use_container_width=True)

        if len(st.session_state.local_rewards) > 0:
            local_rewards_df = pd.DataFrame({'Cumulated Reward': st.session_state.local_rewards}).cumsum()
            local_rewards_df['Step'] = [i for i in range(len(st.session_state.local_rewards))]
            local_rewards_fig = px.line(local_rewards_df, x='Step', y='Cumulated Reward', markers=True, title='Cumulated Reward')
            st.plotly_chart(local_rewards_fig, theme="streamlit", use_container_width=True)
    
    if st.session_state.episode > 1:
        with st.expander('RL Performance (global)', expanded=False):
            global_fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Silhouette Coefficient", "Calinski and Harabasz", "Davies-Bouldin", "Dunn index"))

            global_fig.add_trace(go.Scatter(x=[i for i in range(len(st.session_state.global_silhouette))], y=st.session_state.global_silhouette),
                        row=1, col=1)

            global_fig.add_trace(go.Scatter(x=[i for i in range(len(st.session_state.global_calinski_harabasz))], y=st.session_state.global_calinski_harabasz),
                        row=1, col=2)

            global_fig.add_trace(go.Scatter(x=[i for i in range(len(st.session_state.global_davies_bouldin))], y=st.session_state.global_davies_bouldin),
                        row=2, col=1)

            global_fig.add_trace(go.Scatter(x=[i for i in range(len(st.session_state.global_dunn_index))], y=st.session_state.global_dunn_index),
                        row=2, col=2)

            global_fig.update_layout(height=500, width=700,
                            title_text="Unsupervised Metrics", showlegend=False)

            # Update xaxis properties
            global_fig.update_xaxes(title_text="Episode", row=1, col=1)
            global_fig.update_xaxes(title_text="Episode", row=1, col=2)
            global_fig.update_xaxes(title_text="Episode", row=2, col=1)
            global_fig.update_xaxes(title_text="Episode", row=2, col=2)

            st.plotly_chart(global_fig, theme="streamlit", use_container_width=True)

            if len(st.session_state.global_rewards) > 0:
                global_rewards_df = pd.DataFrame({'Total Reward per Episode': st.session_state.global_rewards})
                global_rewards_df['Episode'] = [i for i in range(len(st.session_state.global_rewards))]
                global_rewards_df = px.line(global_rewards_df, x='Episode', y='Total Reward per Episode', markers=True, title='Total Reward per Episode')
                st.plotly_chart(global_rewards_df, theme="streamlit", use_container_width=True)
    
    
    if len(st.session_state.local_features) > 0:
        with st.expander('Selected Features', expanded=True):
            features_df = pd.DataFrame({'Features': st.session_state.local_features})
            st.table(features_df)
    
    if len(st.session_state.local_features) > 0:
        with st.expander('Clusters', expanded=True):
            y_pred = st.session_state.model.fit_predict(st.session_state.df_all_feats[st.session_state.local_features])
            AMI = adjusted_mutual_info_score(y_pred, st.session_state.y_true)
            st.metric('AMI', round(AMI, 4))
