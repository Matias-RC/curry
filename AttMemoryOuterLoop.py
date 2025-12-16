import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

# ---------- Minimal stubs for environment / experience manager ----------
class Board:
    """Minimal environment stub so the module type-checks in an editor.
    Replace with real environment API later.
    """
    def __init__(self, state_tensor: torch.Tensor, n_actions: int = 4):
        self._state = state_tensor
        self.n_actions = n_actions

    def current_state_tokens(self) -> torch.Tensor:
        """Return a token / vector representation of the state.
        Here we simply return the stored tensor (already tokenized).
        """
        return self._state

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Dummy step: returns same state, zero reward, not done.
        Replace with real env logic.
        """
        return self._state, 0.0, False, {}


class ExperienceManager:
    """Simple experience manager stub.
    - push_trajectory collects trajectories
    - update_parameters is a placeholder to run optimization
    """
    def __init__(self):
        self._db = []

    def push_trajectory(self, trajectory: List[Tuple], high_level_seq: List[torch.Tensor]) -> None:
        self._db.append((trajectory, high_level_seq))

    def update_parameters(self) -> None:
        # Dummy: in real life compute losses and step optimizers
        pass


# ---------- Data structures: Slot & ScratchPad ----------
class Slot:
    def __init__(self, id_dim: int, body_dim: int, surf_dim: int):
        # short identifier vector (meant to be cheap to read every think-step)
        self.id: torch.Tensor = torch.zeros(id_dim, dtype=torch.float32)
        # long-term content for the slot; updated only by Consolidator
        self.body: torch.Tensor = torch.zeros(body_dim, dtype=torch.float32)
        # ephemeral surface updated every think-step by the Thinker (gated write)
        self.surface: torch.Tensor = torch.zeros(surf_dim, dtype=torch.float32)
        self.meta: Dict = {"writes": 0, "last_write": -1}


class ScratchPad:
    def __init__(self, n_slots: int, id_dim: int, body_dim: int, surf_dim: int):
        self.slots: List[Slot] = [Slot(id_dim, body_dim, surf_dim) for _ in range(n_slots)]
        self.step_counter: int = 0  # global think-step counter; incremented by Agent

    def read_slot(self, idx: int) -> Slot:
        """Return exact slot object (no smoothing)."""
        return self.slots[idx]

    def get_all_ids_tensor(self) -> torch.Tensor:
        """Return stacked ids of all slots: shape (n_slots, id_dim).
        The Thinker will receive this tensor at every think-step (explicit requirement).
        """
        return torch.stack([s.id for s in self.slots], dim=0)

    def gated_write_surface(self, idx: int, surf_delta: torch.Tensor, gate: torch.Tensor) -> None:
        """Gated write to the surface of slot `idx`.
        gate expected to be broadcastable to surface shape and in [0,1].
        """
        s = self.slots[idx]
        # clamp gate to [0,1] to avoid NaNs in editor type-checking
        gate_clamped = torch.clamp(gate, 0.0, 1.0)
        s.surface = (1.0 - gate_clamped) * s.surface + gate_clamped * surf_delta
        s.meta["writes"] += 1
        s.meta["last_write"] = self.step_counter

    def update_slot_body(self, idx: int, new_body: torch.Tensor, body_gate: torch.Tensor) -> None:
        """Called by Consolidator at end of trajectory to update long-term `body`.
        body_gate in [0,1] decides how much to incorporate.
        """
        s = self.slots[idx]
        body_gate_clamped = torch.clamp(body_gate, 0.0, 1.0)
        s.body = (1.0 - body_gate_clamped) * s.body + body_gate_clamped * new_body

    def increment_step(self) -> None:
        self.step_counter += 1

    def reset_surfaces(self) -> None:
        for s in self.slots:
            s.surface.zero_()


# ---------- Core modules (minimal / dummy implementations) ----------
class Thinker(nn.Module):
    """Decoder-style module `f`.

    Important: at every think-step this module receives:
      - board_tokens: vector encoding the board
      - current_slot_id: the id vector of the currently selected slot
      - current_slot_surface: the surface vector of that slot (exact, not smoothed)
      - all_slot_ids: stacked ids of ALL slots (explicitly provided every step)
      - prev_think_history: list of previous synth vectors (short summary vectors)

    Outputs:
      - action_logits: policy over environment actions
      - value: scalar value estimate
      - slot_select_logits: logits for selecting the next slot (discrete hard-sampling)
      - surf_update: proposed surface delta to be gated into the current slot.surface
      - synth_vector: short vector retained in high_level_seq for Consolidator
    """

    def __init__(self, board_dim: int, id_dim: int, surf_dim: int, synth_dim: int, n_actions: int, n_slots: int):
        super().__init__()
        # small MLPs for dummy behavior; replace with attention-decoder later
        input_dim = board_dim + id_dim + surf_dim + (n_slots * id_dim) // 4
        # note: we artificially compress `all_slot_ids` in input_dim to keep the linear small
        self._enc = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self._policy_head = nn.Linear(128, n_actions)
        self._value_head = nn.Linear(128, 1)
        self._slot_select_head = nn.Linear(128, n_slots)
        self._surf_update_head = nn.Linear(128, surf_dim)
        self._synth_head = nn.Linear(128, synth_dim)
        self._gate_head = nn.Linear(128, 1)  # single scalar gate for surf writes

    def forward(
        self,
        board_tokens: torch.Tensor,
        curr_slot_id: torch.Tensor,
        curr_slot_surface: torch.Tensor,
        all_slot_ids: torch.Tensor,
        prev_think_history: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        # Simple flattening / compression of inputs to avoid large linear layers.
        if prev_think_history is None or len(prev_think_history) == 0:
            hist = torch.zeros(0)
        else:
            # concatenate last few synths (clamp to reasonable length); flatten
            hist = torch.cat(prev_think_history[-4:], dim=-1)

        # compress all_slot_ids by mean to keep dimension bounded (dummy logic)
        all_ids_compressed = all_slot_ids.mean(dim=0)

        inp = torch.cat([board_tokens.view(-1), curr_slot_id.view(-1), curr_slot_surface.view(-1), all_ids_compressed.view(-1), hist], dim=0)
        # ensure correct device / shape (add batch dim behaviour is left to caller)
        h = self._enc(inp.unsqueeze(0)).squeeze(0)

        action_logits = self._policy_head(h)
        value = self._value_head(h).squeeze(-1)
        slot_select_logits = self._slot_select_head(h)
        surf_update = self._surf_update_head(h)
        synth_vector = self._synth_head(h)
        gate_logit = self._gate_head(h)

        return {
            "action_logits": action_logits,
            "value": value,
            "slot_select_logits": slot_select_logits,
            "surf_update": surf_update,
            "synth_vector": synth_vector,
            "write_gate_logit": gate_logit,
        }


class Consolidator(nn.Module):
    """Module `g` that consumes the concatenated synth_vectors across the trajectory
    and produces per-slot updates (new body proposals, body gates, optional new ids).

    This dummy implementation aggregates synths (mean) and produces updates for every slot
    using small linear heads. Replace with attention / transformer over synths later.
    """

    def __init__(self, synth_dim: int, body_dim: int, id_dim: int, n_slots: int):
        super().__init__()
        self.n_slots = n_slots
        self._agg = nn.Sequential(nn.Linear(synth_dim, 128), nn.ReLU())
        # produce concatenated updates for all slots in one forward (dummy)
        self._body_head = nn.Linear(128, n_slots * body_dim)
        self._body_gate_head = nn.Linear(128, n_slots)
        self._id_head = nn.Linear(128, n_slots * id_dim)

    def forward(self, synths_sequence: List[torch.Tensor], scratchpad: ScratchPad) -> Dict[int, Dict[str, torch.Tensor]]:
        if len(synths_sequence) == 0:
            # return no updates
            return {}
        stack = torch.stack(synths_sequence, dim=0)
        mean_synth = stack.mean(dim=0)
        h = self._agg(mean_synth)

        body_concat = self._body_head(h)
        body_gate_logits = self._body_gate_head(h)
        id_concat = self._id_head(h)

        # split and return structured dict
        out = {}
        body_dim = scratchpad.slots[0].body.shape[0]
        id_dim = scratchpad.slots[0].id.shape[0]
        for i in range(self.n_slots):
            start_b = i * body_dim
            end_b = start_b + body_dim
            new_body = body_concat[start_b:end_b]
            body_gate = torch.sigmoid(body_gate_logits[i])

            start_id = i * id_dim
            end_id = start_id + id_dim
            new_id = id_concat[start_id:end_id]

            out[i] = {"body_delta": new_body, "body_gate": body_gate, "new_id": new_id}
        return out


# ---------- Agent (inner-loop driver) ----------
class Agent:
    def __init__(
        self,
        thinker: Thinker,
        consolidator: Consolidator,
        scratchpad: ScratchPad,
        thinking_steps: int = 4,
        max_prev_history: int = 8,
    ) -> None:
        self.thinker = thinker
        self.consolidator = consolidator
        self.scratchpad = scratchpad
        self.thinking_steps = thinking_steps
        self.max_prev_history = max_prev_history

        # high-level sequence of synth_vectors collected across states
        self.high_level_seq: List[torch.Tensor] = []
        # previous think history (list of synth vectors)
        self.prev_think_history: List[torch.Tensor] = []

    def initial_slot_choice(self, board_tokens: torch.Tensor) -> int:
        # trivial heuristic: choose slot with fewest writes
        writes = [s.meta["writes"] for s in self.scratchpad.slots]
        return int(min(range(len(writes)), key=writes.__getitem__))

    def inner_think(self, board_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        collected_synths: List[torch.Tensor] = []
        curr_slot_idx = self.initial_slot_choice(board_tokens) # To be implemented -> how do you chose first slot? Maybe since it is a decoder
        # You can treat it as a memoryless think step / maybe I could train the decoder to also be capable without previous memory

        for k in range(self.thinking_steps):
            # increment the shared step counter (used by ScratchPad to mark last_write)
            self.scratchpad.increment_step()

            slot = self.scratchpad.read_slot(curr_slot_idx)
            all_ids = self.scratchpad.get_all_ids_tensor()

            out = self.thinker(
                board_tokens=board_tokens,
                curr_slot_id=slot.id,
                curr_slot_surface=slot.surface,
                all_slot_ids=all_ids,
                prev_think_history=self.prev_think_history,
            )

            # compute write gate and apply to the SAME slot (curr_slot_idx)
            gate = torch.sigmoid(out["write_gate_logit"])  # scalar in [0,1]
            surf_delta = out["surf_update"]
            # dummy broadcasting for gate -> surf shape
            gate_broad = gate
            self.scratchpad.gated_write_surface(curr_slot_idx, surf_delta, gate_broad)

            # collect synth for later Consolidator
            collected_synths.append(out["synth_vector"].detach())
            slot_select_logits = out["slot_select_logits"]
            probs = torch.softmax(slot_select_logits, dim=-1)

            probs = probs + 1e-8
            next_slot_idx = int(torch.multinomial(probs, num_samples=1).item())

            # maintain prev_think_history (bounded)
            self.prev_think_history.append(out["synth_vector"].detach())
            if len(self.prev_think_history) > self.max_prev_history:
                self.prev_think_history.pop(0)

            curr_slot_idx = next_slot_idx

        return out["action_logits"], out["value"], collected_synths

# ---------- Outer loop / Trajectory management (functions) ----------

def inner_loop(agent: Agent, board: Board, scratchpad: ScratchPad, hyperparams: Dict) -> Tuple[bool, List[torch.Tensor], List[Tuple]]:
    """Run a rollout where at each environment state the agent performs an inner thinking loop.

    Returns: done flag, high_level_seq (synths), trajectory list.
    """
    trajectory: List[Tuple] = []
    agent.high_level_seq = []

    done = False
    steps = 0
    while not done and steps < hyperparams.get("max_inner_steps", 50):
        board_tokens = board.current_state_tokens()
        action_logits, value, synths = agent.inner_think(board_tokens)

        # collect synths for trajectory-level consolidation
        agent.high_level_seq.extend(synths)

        # sample an external action from policy
        action_probs = torch.softmax(action_logits, dim=-1)
        action = int(torch.multinomial(action_probs + 1e-8, num_samples=1).item())

        obs, reward, done, info = board.step(action)
        trajectory.append((obs, reward, info))
        steps += 1

    # at end of inner loop, call consolidator to produce per-slot updates
    slot_updates = agent.consolidator(agent.high_level_seq, scratchpad)
    for idx, upd in slot_updates.items():
        scratchpad.update_slot_body(idx, upd["body_delta"], upd["body_gate"])
        if "new_id" in upd:
            scratchpad.slots[idx].id = upd["new_id"].detach()

    return done, agent.high_level_seq, trajectory


def outer_loop(agent: Agent, board: Board, scratchpad: ScratchPad, hyperparams: Dict, experience_manager: ExperienceManager) -> None:
    episode = 0
    while episode < hyperparams.get("max_episodes", 10):
        # optionally reset only surfaces (keep bodies/ids persistent across episodes)
        if hyperparams.get("reset_surfaces_per_episode", True):
            scratchpad.reset_surfaces()

        done, high_level_seq, traj = inner_loop(agent, board, scratchpad, hyperparams)
        experience_manager.push_trajectory(traj, high_level_seq)

        # outer update placeholder: compute gradients / update parameters
        experience_manager.update_parameters()
        episode += 1

# If run as script, create a tiny system to type-check / exercise the API
if __name__ == "__main__":
    board = Board(torch.randn(16))
    n_slots = 6
    sp = ScratchPad(n_slots=n_slots, id_dim=4, body_dim=16, surf_dim=8)
    thinker = Thinker(board_dim=16, id_dim=4, surf_dim=8, synth_dim=12, n_actions=4, n_slots=n_slots)
    consolidator = Consolidator(synth_dim=12, body_dim=16, id_dim=4, n_slots=n_slots)
    agent = Agent(thinker, consolidator, sp, thinking_steps=3)
    em = ExperienceManager()
    hyper = {"max_inner_steps": 5, "max_episodes": 1}
    outer_loop(agent, board, sp, hyper, em)

    print("Dummy run finished. Slots writes:")
    for i, s in enumerate(sp.slots):
        print(i, s.meta)
"""
There are some things you got wrong about the architecture. First of all Thinker at every think step does not output everything you say. instead most things are 
harvested after K thinking steps. 
And relative to other aspects of the architecture I have partially changed my mind the re-writte stuff at scratch pad was a little convoluted I instead want the 
vector decoded at think step which is not a synth vector it will be called think_step the think step is a vector and is decoded by the thinker with  this information (
b_t, previous_think_steps (sliding window of attention), (the body, id, and previously written think steps) of the selected slot at scratchpad, all of the ids of the
scratch pad)
); to be appended to 
the slot and every time the agent acceses this slot in memory it writes a thinkstep there and can read the last think steps writen by itself
"""

"""
Ive changed my mind respect the architecture. the scratch-pad won't be strictly necesary to have for the agent to function, at the end of the day we are still handling
an attention based decoder that has flexibility for input length. Still, since the agent will be allowed for re try of the level the scratchpad will be useful then.
So how do we implement it? well, at any state s without a scratch pad thinkier will take in (B_t, all of the previous think steps (empty if s_0)) and will decode a
think step (a think step is a vector and is just for informative/planning purposes) this think step is incorporated into previous think steps and the thinker decodes
again, this repeats K times. Once K thinking steps have been generated for state s_t an a neural network encoder will generate two vectors: synth and primitive_policy (primitive policy
because we use an MLP to get the action logits and value function from it then synth is kept and the next state of the board is reached after smpling from the logits.
Then the cycle repeats. Alternatively if there is a scratch pad then always the first index at the scratch pad is selected as first to be read, the agent at any given
state s with a scratch pad will take in (B_t, all prev think steps, body of slot, scratch surface) Scratch surface is no longer accurate because I want to assign the
"""