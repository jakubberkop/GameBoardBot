import os
from typing import List, Tuple

from sb3_contrib import MaskablePPO
import tqdm

def get_all_models():
    models = []

    for folder in ["ppo_elo", "."]: #, "model_checkpoints", "model_checkpoints_elo"]:
        for file in os.listdir(folder):
            if file.endswith(".zip"):
                models.append(f"{folder}/{file}")


        # for root, dirs, files in os.walk(folder):
        #     for file in files:
        #         if file.endswith(".zip"):
        #             models.append(f"{root}/{file}")

    models.sort(key=lambda x: os.path.getmtime(x))

    return models


def get_all_okay_models() -> List[Tuple[MaskablePPO, str]]:
    models = get_all_models()
    newest_model = MaskablePPO.load(models[-1])
    okay_models: List[Tuple[MaskablePPO, str]] = []

    for model in models:
        try:
            loaded_model = MaskablePPO.load(model)
        except:
            continue

        if loaded_model.observation_space == newest_model.observation_space and loaded_model.action_space == newest_model.action_space:
            okay_models.append((loaded_model, model),)
        # else:
        #     os.remove(model)


    return okay_models