# Architecture Overview: Residual Causal Convolutional Memory

This network is a hybrid architecture designed for Reinforcement Learning in partially observable 2D environments (like Sokoban). It replaces standard LSTM memory with a fixed-size **Recurrent Feature Stack** (Fixed during a  play, but could be adapted in size based on difficulty of level).

### 1. The Core Concept
The network utilizes **Causal Self-Attention** within a recurrent loop.

Instead of compressing the entire history into a single hidden vector (like an RNN) or keeping a static history of raw inputs (like a standard sliding window), this agent maintains a stack of $M$ **Latent Feature Maps**. 

Crucially, this memory is **Recurrent**: The features in the stack are not static snapshots. At every timestep, the Causal Self-Attention mechanism compares the *Current Frame* against the *Memory Stack*, refining and updating the stored features before passing them to the next timestep.

### 2. Data Flow Mechanics

**Step-by-Step for a Single Timestep:**

1.  **Input Injection (Concatenation)**
    * The network takes the **Current Frame Features** ($x_t$) and prepends them to the **Memory Stack** ($H_{t-1}$) from the previous step.
    * *Result:* A sequence structured as `[Current, Memory_1, Memory_2, ... Memory_N]`.

2.  **Causal Self-Attention**
    * The network applies Self-Attention to this combined sequence.
    * **The Causal Mask:** A lower-triangular mask is applied. Because the current frame is at Index 0, this creates a specific information flow:
        * **Index 0 (Current Frame):** Can only see itself. It remains "pure" and uninfluenced by the past.
        * **Index $N$ (Oldest Frame):** Can see indices $0$ through $N$. It can attend to the Current Frame *and* the entire history stack.

3.  **Residual Update (Identity Preservation)**
    * The output of the attention mechanism is added back to the original sequence: 
        $$\text{New Sequence} = \text{Old Sequence} + \text{Attention}(\text{Old Sequence})$$
    * This residual connection ensures the network learns *refinements* to the memory rather than overwriting it, preserving the identity of stable features (like walls) while updating dynamic ones.

4.  **Output & State Management**
    * **Network Output:** The **last element** of the sequence is extracted and sent to the Actor/Critic heads. Because of the masking logic, this element acts as the **Global Accumulator**—it is the only slot containing aggregated information from the entire window.
    * **Next State:** The sequence is sliced to keep the *last* $M$ elements. The slot physically containing the "Current Input" (Index 0) is dropped, but its information has been integrated into the deeper memory slots via the residual attention update.

---

### 3. Common Misunderstandings

#### A. "It's just a sliding window buffer."
* **Misunderstanding:** Thinking the memory holds raw, static copies of the last $M$ inputs (like a FrameStack wrapper).
* **Reality:** It is an **Evolving Latent Memory**. A feature map in the "oldest" slot has been processed and refined $M$ times. It represents the network's *current understanding* of that past moment, not the raw pixel data.

#### B. "The 'Oldest' Frame is the least important."
* **Misunderstanding:** Thinking the last element in the tensor (`[:, -1]`) is just "context from 4 steps ago" and therefore outdated.
* **Reality:** The "Oldest" slot is actually the **Most Important**. Due to the Causal Masking, it functions as the **Summary Token** (similar to the `[CLS]` token in BERT). It is the decision-making bottleneck that has access to the full context window.

#### C. "Is it a Transformer or an RNN?"
* **Misunderstanding:** Classifying it strictly as one or the other.
* **Reality:** It is a **Recurrent Transformer**. It uses Transformer mechanisms (Attention, Residuals) for processing, but RNN mechanics (State passing) for temporal continuity.

#### D. "Why the Residual Connection?"
* **Misunderstanding:** Thinking the residual addition `h + att(h)` is just a standard trick for training stability.
* **Reality:** In this specific recurrent architecture, the residual connection is critical for **Memory Persistence**. Without it, the "Update" step would destructively overwrite the memory features at every timestep, turning the history into noise. The residual forces the network to only modify what needs to change.
