# Main experiment

## Controlled variables 
- **Total number of parameters**  
  match a target param-budget per comparative run  
- **Amount of cells in the LSTM stack (stack depth)**  
  number of cells (and hiden states) a single thinking step uses.
- **Relative density of a single cell**
  use preset density levels D1..Dn.
- **Training regimen & environment** (implicit controls)  
  same training frames, optimizer, LR schedule, train/test level sets, and seeds policy per comparison.


## Independent variable
- **Thinking steps supplied to the agent** — discrete levels: {1, 2, 3, 4, 5}  
  *Operation:* number of internal recurrent/imagined/think iterations the agent is allowed per decision.

## Dependent variables
- **Learning speed** frames (or wall-clock time) to reach target success (steps-to-50% or steps-to-threshold).  
- **Overall performance on Sokoban levels** final success rate %, avg steps-to-solve, reward AUC.  
- **Adaptability / generalization** performance drop on held-out / OOD levels (d%), transfer success.  
- **Stability / variance across seeds** -> std


## Experimental repeats / blocking
- **Repetitions:** 5 seeds per (density, stack size, thinking_steps) condition used to compute mean±std for dependent variables.

## Hypothesis:
H0: Extra thinking steps produce diminishing returns (log-like).

H1: External sparse memory + repeated cells yields near-linear gains with thinking steps (at least if memory size is kept big).

**Decision thresholds (To be further reviewd)** Sublinear if delta performance drops by 20% at each think step increase.
**Metrics idea** Mineral enrichment metrics of chemestry.