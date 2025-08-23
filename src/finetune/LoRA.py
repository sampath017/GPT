import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.scaling = alpha / r

        # frozen base weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        # LoRA adapters
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A, self.lora_B = None, None

    def forward(self, x):
        result = F.linear(x, self.weight)
        if self.r > 0:
            lora_update = F.linear(self.dropout(
                x), self.lora_A)  # type: ignore
            lora_update = F.linear(lora_update, self.lora_B) * self.scaling  # nopep8 # type: ignore
            result += lora_update
        return result

    # ---- Main utility like get_peft_model ----

    @staticmethod
    def apply_lora(model, r=8, alpha=16, dropout=0.05, target_modules=("attn", "proj")):
        """
        Replaces Linear layers inside target_modules with LoRALinear,
        freezes base params, and returns LoRA-augmented model.
        """
        def _replace(module, name_prefix=""):
            for name, child in list(module.named_children()):
                full_name = f"{name_prefix}.{name}" if name_prefix else name

                if isinstance(child, nn.Linear) and any(t in full_name for t in target_modules):
                    in_features, out_features = child.in_features, child.out_features
                    lora_layer = LoRALinear(
                        in_features, out_features, r=r, alpha=alpha, dropout=dropout)
                    lora_layer.weight.data = child.weight.data.clone()  # copy pretrained weight
                    setattr(module, name, lora_layer)

                else:
                    _replace(child, full_name)

        # replace linear layers
        _replace(model)

        # freeze all except LoRA
        for name, param in model.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                param.requires_grad = False

        return model
