from transformers import TrainerCallback
import logging 

class CustomLogCallback(TrainerCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            log_str = f"Step {state.global_step}: " + ", ".join([f"{k}={v}" for k, v in logs.items()])
            logging.info(log_str)
            