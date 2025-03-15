from transformers import TrainerCallback
import torch
import os
# import wandb

class SubmoduleGradLogger(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        grad_norms = {}
        none_modules = []
        for name, module in model.named_modules():
            params = [p for p in module.parameters() if p.requires_grad]
            none_params = []
            if params:
                norms = []
                for p in params:
                    if p.grad is None:
                        # 记录为 None 或者 0
                        none_params.append(p)
                    else:
                        norms.append(p.grad.detach().norm(2))
                if norms:
                    total_norm = torch.norm(torch.stack(norms), 2)
                    grad_norms[f"{name}_grad_norm"] = total_norm.item()
        
            if none_params:
                none_modules.append(name)
        if none_modules:
            os.makedirs("logs", exist_ok=True)
            with open("logs/none_modules.txt", "a") as f:
                f.write(str(none_modules) + "\n")
        # wandb.log(grad_norms, step=state.global_step)
        return control
