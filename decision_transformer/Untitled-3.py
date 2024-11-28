import datasets
from transformers import DecisionTransformerConfig, DecisionTransformerModel, TrainingArguments, Trainer, TrainerCallback

from game import PlayerDecision, initialize_game_state

import numpy as np
import torch

import pytracy
# pytracy.set_tracing_mode(pytracy.TracingMode.All)

state_size = len(initialize_game_state().to_state_array(0))
action_size = len(PlayerDecision(0).to_state_array())

ds = datasets.load_from_disk("data/dataset3big")
split_ds = ds.train_test_split(test_size=0.1)

# %%
TARGET_RETURN = 20

class MyDataCollator:
    # This class maps "states", "actions", "rewards", "done"
    # into 
    # "states" ,"actions" ,"rewards" ,"returns_to_go" ,"timesteps"

    def __call__(self, features):
        # add the returns to go

        batch_size = len(features)

        s, a, r, d, rtg, timesteps = [], [], [], [], [], []

        for i in range(batch_size):
            s.append(features[i]["states"])
            a.append(features[i]["actions"])
            # r.append(features[i]["rewards"])
            d.append(features[i]["dones"])
            rtg.append([TARGET_RETURN for _ in range(len(features[i]["rewards"]))])
            timesteps.append([i for i in range(len(features[i]["rewards"]))])

        # Change shape from (batch_size, 1, episode_length) to (batch_size, episode_length, 1)
        r = np.array([[features[i]["rewards"]] for i in range(batch_size)], dtype=np.float32).reshape(batch_size, -1, 1)

        # Change shape from (batch_size, episode_length) to (batch_size, episode_length, 1)
        rtg = np.array(rtg, dtype=np.float32).reshape(batch_size, -1, 1)
        d = np.array(d).reshape(batch_size, -1, 1)
        # timesteps = np.array(timesteps).reshape(batch_size, -1, 1).astype(np.int64)
        timesteps = np.array(timesteps).astype(np.int64)

        mask = np.ones_like(timesteps, dtype=np.float32)


        s         = torch.tensor(s)
        a         = torch.tensor(a)
        r         = torch.tensor(r)
        d         = torch.tensor(d)
        rtg       = torch.tensor(rtg)
        timesteps = torch.tensor(timesteps).long()
        mask      = torch.tensor(mask).float()

        # s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        # a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        # r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        # d = torch.from_numpy(np.concatenate(d, axis=0))
        # rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        # timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()

        a = {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }
        
        # # print the shapes and types
        # for k, v in a.items():
        #     print(f"{k:10}: {str(v.shape):10} {v.dtype}")
        # print()

        return a

collator = MyDataCollator()

configuration = DecisionTransformerConfig(
	state_dim=state_size,
	act_dim=action_size,
)

# model = DecisionTransformerModel(configuration)
class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        # loss = torch.mean((action_preds - action_targets) ** 2)
        loss_fun = torch.nn.CrossEntropyLoss()
        loss = loss_fun(action_preds, action_targets)

        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)

model = TrainableDT(configuration)

# Load the model from disk
# model = TrainableDT.from_pretrained("results_dt/decision_transformer1")

training_args = TrainingArguments(
    output_dir="results_dt/",
    learning_rate=2e-5,
    per_device_train_batch_size=1736,
    per_device_eval_batch_size=1024,
    num_train_epochs=4,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_ds["train"],
    eval_dataset=split_ds["test"],
    data_collator=collator,
    # callbacks=[PrintLossCallback()]
)

trainer.train()
# trainer.evaluate()



# # Save model
trainer.save_model("results_dt/decision_transformer1")
