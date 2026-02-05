from stable_baselines3.common.callbacks import BaseCallback

class MapleCallback(BaseCallback):
    """
    Custom callback for MAPLE to log meta-learning metrics.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.inner_loop_updates = 0

    def _on_step(self) -> bool:
        # Access the MAPLE model via self.model
        
        # 1. Log Prefix Norms (Are they growing? Exploding? Vanishing?)
        # We check the first environment's prefix as a proxy
        if self.model.dynamic_buffer.current_prefixes is not None:
            actor_prefix, critic_prefix = self.model.dynamic_buffer.current_prefixes
            
            # Log magnitude of the prefixes
            self.logger.record("maple/actor_prefix_norm", actor_prefix[0].norm().item())
            self.logger.record("maple/critic_prefix_norm", critic_prefix[0].norm().item())

        # 2. Log Inner Loop Activity
        # You might want to track how often we are running the inner loop
        # We can check the infos from the last step
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("message") == "restored_from_save":
                self.inner_loop_updates += 1
        
        self.logger.record("maple/inner_loop_count", self.inner_loop_updates)
        return True

    def _on_rollout_end(self) -> None:
        # Optional: Log Consolidator buffer size to ensure we are gathering data
        buffer_len = len(self.model.consolidator_buffer)
        self.logger.record("maple/consolidator_buffer_size", buffer_len)