import numpy as np

from PopArt import PopArt
from OnlineDeviationMean import OnlineDeviationMean
import torch
import random
popart_layer = PopArt(input_features=2, output_features=2)

# che_list = [random.uniform(0, 1) for _ in range(20)]
# dhe_list = [random.uniform(10, 20) for _ in range(20)]
# she_list = [random.uniform(0, 0.5) for _ in range(20)]
che_list = [100, 200, 400, 800]
dhe_list = [-7, -6, -5.9, -5]
# she_list = [0.9294905689917834, 0.3359975444038169, 0.5734267341088504, 0.3217149834508586, 0.6715180219638304, 0.9947623736574149, 0.22762368680955902, 0.9893318692225025, 0.24558633481495085, 0.964925772521628,0.9294905689917834, 0.3359975444038169, 0.5734267341088504, 0.3217149834508586, 0.6715180219638304, 29.9947623736574149, 0.22762368680955902, 0.9893318692225025, 0.24558633481495085, 0.964925772521628]

che_list_norm = OnlineDeviationMean()
dhe_list_norm = OnlineDeviationMean()
she_list_norm = OnlineDeviationMean()

for i in range(20):
        # chs_tensor = torch.tensor([chs]).unsqueeze(0)
        # Crea un tensore con le tue metriche
        metriche = torch.tensor([[float(che_list[i]), float(dhe_list[i])]])
        # Normalizza il punteggio CHS usando il modulo PopArt
        chs_normalizzato = popart_layer(metriche)
        # Aggiorna i parametri del modulo PopArt con il punteggio CHS
        popart_layer.update_parameters(metriche)
        if i > 1:
                norm_val = che_list_norm.compute_normalize(che_list[i])
                norm_val_1 = dhe_list_norm.compute_normalize(dhe_list[i])
                #norm_val_2 = she_list_norm.compute_normalize(she_list[i])
                print(che_list[i], dhe_list[i])
                print("Average Mean :", np.mean([norm_val, norm_val_1]))
                print("Average PopArt :", chs_normalizzato["pred"][0].detach().numpy().mean())
        che_list_norm.add_data_point(che_list[i])
        dhe_list_norm.add_data_point(dhe_list[i])
        # she_list_norm.add_data_point(she_list[i])

        print("----------")