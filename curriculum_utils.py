from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

class VectorCurriculumCallback(BaseCallback):
    """
    Updates the environment's task distribution based on total timesteps.
    Designed to work with Vectorized Environments (VecEnv) by using env_method.
    """
    def __init__(self, schedule_plan, verbose=0):
        super().__init__(verbose)
        self.schedule_plan = schedule_plan
        # Sort thresholds to ensure we check them in order
        self.thresholds = sorted(schedule_plan.keys())
        self.current_stage_idx = -1

    def _on_step(self) -> bool:
        # Calculate which stage we should be in based on total steps
        next_stage_idx = self.current_stage_idx + 1
        
        # If there are stages left to advance to...
        if next_stage_idx < len(self.thresholds):
            threshold = self.thresholds[next_stage_idx]
            
            # If we passed the timestep threshold for the next stage
            if self.num_timesteps >= threshold:
                new_schedule = self.schedule_plan[threshold]
                self.current_stage_idx = next_stage_idx
                
                if self.verbose > 0:
                    print(f"\n[Curriculum] --------------------------------------------------")
                    print(f"[Curriculum] UPGRADE TRIGGERED at step {self.num_timesteps}")
                    print(f"[Curriculum] Broadcasting new schedule to {self.training_env.num_envs} environments.")
                    print(f"[Curriculum] New Schedule: {new_schedule}")
                    print(f"[Curriculum] --------------------------------------------------")

                # BROADCAST UPDATE:
                # 'update_schedule' must be a method in your SokoRetriesCurriculum class.
                # env_method automatically tunnels through Monitor/DummyVecEnv layers.
                self.training_env.env_method("update_schedule", new_schedule)
                
        return True
