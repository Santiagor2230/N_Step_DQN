algo = DeepQLearning(
    'PongNoFrameskip-v4',
    lr=1e-4,
    sigma=0.5,
    hidden_size=256,
    a_last_episode=1_000,
    b_last_episode= 1_000,
    n_steps=8,
    samples_per_epoch = 10_000
)

trainer = Trainer(
    gpus=num_gpus,
    max_epochs=1_200,
    log_every_n_steps=1
)

trainer.fit(algo)