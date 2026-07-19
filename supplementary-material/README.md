# Supplementary Material

This directory contains the supplementary assets for the ISWC 2026 paper
"Neuro-Symbolic Meta-Policies for Temporal Knowledge-Graph Memory under Partial
Observability".

The code that reproduces the reported experiments lives at the [repository
root](../) (see the main [README](../README.md) for entry points and run
commands).

## Contents

- `graph_view_step_099.pdf` and `graph-view_step_099.pdf`: full-size graph view figure
  for the final memory state.
- `memory_state_graphs/`: memory evolution figures across steps 0 to 99.
- `q_values_explore.pdf`, `q_values_forget.pdf`, `q_values_qa.pdf`: diagnostic plots for
  learned meta-policy values.

## Q-Value Plot Notes

- `q_values_qa.pdf`: the QA head shifts from earlier MRU preference toward later MRA preference, consistent with a transition from relying on recently reused facts to relying on newly added facts as exploration reveals fresher evidence.
- `q_values_explore.pdf`: the exploration head follows a rise-then-fall pattern, since expected return increases as memory becomes more informative and then decreases near the end of the episode when fewer future questions remain; the three exploration heuristics often stay close because they share the same BFS-based movement mechanism.
- `q_values_forget.pdf`: forget-head values appear only after long-term memory reaches capacity; the trace typically shifts from earlier FIFO to stronger LFU preference, which is consistent with age-based eviction being adequate at first and frequency statistics becoming more informative later.

## Additional Hyperparameters

The main paper already reports the shared training protocol (200 episodes, batch size 32,
replay buffer equal to the training horizon, 10\% warm start, target update interval 50,
$\epsilon$ decay from 1.0 to 0.01 over the first half of training, learning rate $10^{-4}$,
$\gamma=0.95$, and gradient clipping). For completeness, the experiments use the
following additional settings that are not enumerated in the main paper:

- Learned runs use seeds `0, 5, 10, 15, 20`.
- The reported GNN and sequence-model configurations use embedding dimension `32` and `4` layers.
- For the reported R-GCN configuration, `num_bases=30`.
- All MLP heads use 1 hidden layer; the neuro-symbolic heads use dueling DQN, while the end-to-end neural baselines do not.
- Transformer baselines use `num_heads=4`, with feedforward dimension equal to `4 x embedding_dim`.
- The reported neuro-symbolic runs include both shared-network and separate-network variants, and a combinatorial 27-way action-head variant.
- Fixed implementation defaults include DDQN enabled, gradient clipping value `10.0`, validation interval `1`, plotting interval `20`, and CPU execution in the released scripts.
- GNN/sequence-model dropout is set to `0.0` in the released paper scripts; for StarE-GNN, `triple_qual_weight=0.8`, SiLU is used between layers, and inter-layer dropout is disabled.
