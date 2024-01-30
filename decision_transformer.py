import datasets
from transformers import DecisionTransformerConfig, DecisionTransformerModel, TrainingArguments, Trainer

from game import PlayerDecision, initialize_game_state

# Initializing a DecisionTransformer configuration

state_size = len(initialize_game_state().to_state_array(0))
action_size = len(PlayerDecision(0).to_state_array())

configuration = DecisionTransformerConfig(
	state_dim=state_size,
	act_dim=action_size,
)

# Initializing a model (with random weights) from the configuration

model = DecisionTransformerModel(configuration)



training_args = TrainingArguments(
    output_dir="results_dt/",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
	remove_unused_columns=False
)

ds = datasets.load_from_disk("data/dataset2_T")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    # eval_dataset=dataset["test"],
    # data_collator=data_collator,
)

trainer.train()