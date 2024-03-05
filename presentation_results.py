import pandas as pd
import matplotlib.pyplot as plt

from early_stopping.early_stopping_class import CustomEarlyStopping


def show_average_AMI(df):
    df['AMI'] = pd.to_numeric(df['AMI'], errors='coerce')

    # Raggruppa i dati per configurazione (Length e Sample) e calcola la media degli AMI
    # Ora raggruppa i dati per 'Length' e 'Sample' e calcola la media di 'AMI'
    grouped_data = df.groupby(['n_feats'])['AMI'].mean().reset_index()

    grouped_data['Configuration'] = grouped_data['n_feats'].astype(str)

    import matplotlib.pyplot as plt
    # Plot a bar chart for each configuration
    unique_configurations = grouped_data['Configuration'].unique()

    plt.figure(figsize=(12, 8))

    for configuration in unique_configurations:
        subset = grouped_data[grouped_data['Configuration'] == configuration]
        plt.bar(subset['Configuration'], subset['AMI'], label=configuration)

    # Etichette degli assi
    plt.xlabel('Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('AMI')

    # Titolo del grafico
    plt.title('Bar Chart per Configurazione con Valori AMI')

    # Legenda
    plt.legend()
    # Mostra il plot
    plt.show()

def time_average(df):
    df['TimeGraph'] = pd.to_numeric(df['TimeGraph'], errors='coerce')
    df['TimeConsensus'] = pd.to_numeric(df['TimeConsensus'], errors='coerce')
    df['TimeCluster'] = pd.to_numeric(df['TimeCluster'], errors='coerce')

    # Raggruppa i dati per configurazione (Length e Sample) e calcola la media degli AMI
    # Ora raggruppa i dati per 'Length' e 'Sample' e calcola la media di 'AMI'
    import numpy as np
    grouped_data = df.groupby(['Length', 'Sample'])[['TimeGraph', 'TimeConsensus', 'TimeCluster']].mean().reset_index()
    # Supponiamo che 'grouped_data' sia il tuo DataFrame raggruppato
    # Se le colonne 'Length' e 'Sample' sono di tipo numerico, puoi concatenarle per formare una nuova colonna
    grouped_data['Configuration'] = grouped_data['Length'].astype(str) + '_' + grouped_data['Sample'].astype(str)

    # Calcola le posizioni delle barre
    unique_configurations = grouped_data['Configuration'].unique()
    bar_width = 0.25  # Larghezza delle barre
    bar_positions = np.arange(len(unique_configurations))

    # Plot a bar chart for each time (TimeGraph, TimeConsensus, TimeCluster)
    plt.figure(figsize=(12, 8))

    plt.bar(bar_positions - bar_width, grouped_data['TimeGraph'], width=bar_width, color='blue', label='TimeGraph')
    plt.bar(bar_positions, grouped_data['TimeConsensus'], width=bar_width, color='green', label='TimeConsensus')
    plt.bar(bar_positions + bar_width, grouped_data['TimeCluster'], width=bar_width, color='red', label='TimeCluster')

    # Etichette degli assi e delle barre
    plt.xlabel('Configuration')
    plt.ylabel('Time')
    plt.title('Bar Chart')
    plt.xticks(bar_positions, unique_configurations, rotation=45, ha='right')  # Etichette personalizzate
    plt.yscale('log')  # Scala logaritmica sull'asse y

    plt.legend()
    # Mostra il plot
    plt.show()

def length_average_ami(df):
    # Supponiamo che 'df' sia il tuo DataFrame
    df['AMI'] = pd.to_numeric(df['AMI'], errors='coerce')

    # Raggruppa i dati per 'Length' e calcola la media di 'AMI'
    grouped_data = df.groupby('Length')['AMI'].mean().reset_index()

    # Aggiungi una colonna 'Configuration' per la visualizzazione
    grouped_data['Configuration'] = grouped_data['Length'].astype(str)

    # Plot della media di AMI per la stessa 'Length'
    plt.figure(figsize=(10, 6))
    plt.bar(grouped_data['Configuration'], grouped_data['AMI'], color='blue')

    # Etichette degli assi
    plt.xlabel('Length')
    plt.ylabel('Average AMI')
    plt.title('Bar Chart per Media di AMI per la stessa Length')

    # Mostra il plot
    plt.show()

def sample_average_ami(df):
    # Supponiamo che 'df' sia il tuo DataFrame
    df['AMI'] = pd.to_numeric(df['AMI'], errors='coerce')

    # Raggruppa i dati per 'Length' e calcola la media di 'AMI'
    grouped_data = df.groupby('Sample')['AMI'].mean().reset_index()

    # Aggiungi una colonna 'Configuration' per la visualizzazione
    grouped_data['Configuration'] = grouped_data['Sample'].astype(str)

    # Plot della media di AMI per la stessa 'Length'
    plt.figure(figsize=(10, 6))
    plt.bar(grouped_data['Configuration'], grouped_data['AMI'], color='blue')

    # Etichette degli assi
    plt.xlabel('Sample')
    plt.ylabel('Average AMI')
    plt.title('Bar Chart per Media di AMI per la stessa Sample')

    # Mostra il plot
    plt.show()

def main():
    # Leggi il CSV
    df_obt = {"ArticularyWordRecognition": 0.963, "BasicMotions": 1.000,
    "Cricket": 0.946,
    "ERing": 0.921,
    "Epilepsy": 0.792,
    "EthanolConcentration": 0.052,
    "HandMovementDirection": 0.015,
    "Handwriting": 0.161,
    "Libras": 0.716,
    "RacketSports": 0.35,
    "StandWalkJump": 0.048,
    "UWaveGestureLibrary": 0.587}
    df = pd.read_csv('resultsT2RL.csv')
    df['AMI'] = pd.to_numeric(df['AMI'], errors='coerce')

    # Group the DataFrame by 'n_feats'
    grouped_df = df.groupby('n_feats')

    # Iterate over the groups and print the results
    for n_feats, group in grouped_df:
        print(f"n_feats: {n_feats}")
        for dataset, data in group.groupby('Dataset'):
            print(f"  Dataset: {dataset}, AMI: {list(data['AMI'])[0]}")
    # show_average_AMI(df)

if __name__ == '__main__':
    # Example usage:
    custom_early_stopping = CustomEarlyStopping(patience=5, plateau_patience=3)

    # Simulate some iterations with an advancing metric
    metric_values = [0.1, 0.2, 0.15, 0.18, 0.11, 0.08, 0.1, 0.09, 0.07, 0.05]

    for iteration, metric_value in enumerate(metric_values, start=1):
        if custom_early_stopping.update_metric(metric_value):
            # Stop the process
            print(f"Stopping the process at iteration {iteration}")
            break
        else:
            # Your iterative process here
            # ...
            print(f"Iteration {iteration}: Metric value = {metric_value}")