import math
import numpy as np


class Sampler:
    def __init__(
        self,
        pool,
        batch_size,
        exploration_ratio,  # γ
        momentum,           # α
    ):
        self.pool = pool
        self.pool_size = len(pool)
        self.batch_size = batch_size
        self.gamma = exploration_ratio
        self.alpha = momentum

        # ω_i(1) = 1
        self.weights = np.ones(self.pool_size, dtype=np.float64)

        # h_i(1) = 0
        self.edge_of_solvability_measure = np.zeros(self.pool_size, dtype=np.float64)

        self.sample_indexes = None
        self.last_probs = None

    def _compute_sampling_probs(self):
        total_weight = self.weights.sum()

        probs = (
            (1 - self.gamma) * self.weights / total_weight
            + self.gamma / self.pool_size
        )

        # numerical safety
        probs = probs / probs.sum()
        return probs

    def sample_batch(self):
        probs = self._compute_sampling_probs()

        self.sample_indexes = np.random.choice(
            self.pool_size,
            size=self.batch_size,
            replace=False,  
            p=probs,
        )

        self.last_probs = probs[self.sample_indexes]

        return [self.pool[i] for i in self.sample_indexes]

    def update_weights(self, successes):
        assert self.sample_indexes is not None, "Call sample_batch() first"

        successes = np.asarray(successes, dtype=np.float64)

        for j, idx in enumerate(self.sample_indexes):
            success = successes[j]

            # r_j(t) = (h_i(t) - 1_success)^2
            h_i = self.edge_of_solvability_measure[idx]
            r_j = (h_i - success) ** 2

            # θ_j(t) = r_j(t) / p_i(t)
            theta_j = r_j / self.last_probs[j]

            # ω_i(t+1) = ω_i(t) * exp(γ * θ_j(t) / N)
            self.weights[idx] *= math.exp(
                self.gamma * theta_j / self.pool_size
            )

            # h_i(t+1) = α h_i(t) + (1 - α) r_j(t)
            self.edge_of_solvability_measure[idx] = (
                self.alpha * h_i + (1 - self.alpha) * r_j
            )



if __name__ == "__main__":
    pool = [f"task_{i}" for i in range(10)]
    pool_succes = [0 for i in range(10)]
    sampler = Sampler(
        pool=pool,
        batch_size=4,
        exploration_ratio=0.2,
        momentum=0.9,
    )

    for epoch in range(500):
        print(f"Epoch {epoch + 1}")
        batch = sampler.sample_batch()
        print("Sampled batch:", batch)

        # Simulate successes (random for demonstration)
        #successes = np.random.randint(0, 2, size=len(batch)).tolist()
        # simulate task 0 has 50/50 succes rate and rest have 10% succes rate
        successes = []
        for task in batch:
            if task == "task_0":
                success = np.random.choice([0, 1], p=[0.5, 0.5])
            else:
                success = np.random.choice([0, 1], p=[0.9, 0.1])
            successes.append(success)
        
        

        sampler.update_weights(successes)

        for i, task in enumerate(batch):
            idx = pool.index(task)
            pool_succes[idx] += successes[i]
        #print("Total successes per task:", pool_succes)
        print("Total successes per task:", pool_succes)
        print()