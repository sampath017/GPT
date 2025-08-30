import torch
import torch.nn as nn
import torch.nn.functional as F
import finetune.settings as s


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, scale=16, dropout=0.05):
        super().__init__()
        self.r = r
        self.scale = scale
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.scaling = scale / r

        # Frozen base weight (use register_buffer for non-trainable params)
        self.register_buffer('weight', torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)  # type: ignore

        # LoRA adapters
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
            self.enabled = True  # Flag to enable/disable LoRA
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)
            self.enabled = False

    def forward(self, x):
        result = F.linear(x, self.weight)  # type: ignore

        if self.r > 0 and self.enabled:
            # Apply dropout to input, then multiply by A, then B
            lora_update = F.linear(self.dropout(x), self.lora_A)
            lora_update = F.linear(lora_update, self.lora_B) * self.scaling
            result += lora_update

        return result

    def merge_weights(self):
        """Merge LoRA weights into base weights (for inference)"""
        if self.r > 0 and self.enabled:
            self.weight.data += (self.lora_B @ self.lora_A) * \
                self.scaling  # type: ignore
            self.enabled = False

    def unmerge_weights(self):
        """Unmerge LoRA weights from base weights"""
        if self.r > 0 and not self.enabled:
            self.weight.data -= (self.lora_B @ self.lora_A) * \
                self.scaling  # type: ignore
            self.enabled = True

    @staticmethod
    # ---- Main utility like get_peft_model ----
    def apply_lora(model, r=8, scale=16, dropout=0.05, target_modules=("attn", "proj")):
        """
        Replaces Linear layers inside target_modules with LoRALinear,
        freezes base params, and returns LoRA-augmented model.
        """
        def _replace(module, name_prefix=""):
            for name, child in list(module.named_children()):
                full_name = f"{name_prefix}.{name}" if name_prefix else name

                if isinstance(child, nn.Linear) and any(t in full_name for t in target_modules):
                    in_features = child.in_features
                    out_features = child.out_features

                    # Create LoRA layer with same configuration
                    lora_layer = LoRALinear(
                        in_features, out_features,
                        r=r, scale=scale, dropout=dropout
                    )

                    # Copy pretrained weights
                    lora_layer.weight.data = child.weight.data.clone()

                    setattr(module, name, lora_layer)
                else:
                    _replace(child, full_name)

        # Replace linear layers
        _replace(model)

        # Freeze all except LoRA parameters
        trainable_params = 0
        total_params = 0

        for name, param in model.named_parameters():
            total_params += param.numel()
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False

        return model.to(s.device)

    @staticmethod
    # Optional: Utility functions for LoRA models
    def get_state_dict(model):
        """Return only LoRA weights"""
        return {k: v for k, v in model.state_dict().items()
                if 'lora_' in k}
