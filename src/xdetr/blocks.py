import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

import timm
import torch
from einops import rearrange
from torch import nn


class LoRARule(TypedDict):
    patterns: Sequence[str]
    rank: int
    alpha: NotRequired[float | None]
    p_dropout: NotRequired[float]


@dataclass(frozen=True)
class _ValidatedLoRARule:
    patterns: tuple[str, ...]
    compiled_patterns: tuple[re.Pattern[str], ...]
    rank: int
    alpha: float | None
    p_dropout: float


class LoRALinear(nn.Module):
    """Apply Low Rank Adaptation (LoRA) to a given Linear layer.

    Args:
        inner: The linear layer to adapt.
        r: the rank.
        alpha: Optional value to control adapter magnitude, where `scale = alpha / r`. Defaults to `r`.
        p_dropout: Dropout probability. Defaults to 0.0.
    """

    def __init__(self, inner: nn.Linear, r: int, alpha: float | None = None, p_dropout: float = 0.0):
        super().__init__()

        if alpha is None:
            alpha = r
        if r <= 0:
            raise ValueError(f"r must be greater than 0 (got {r})")
        if alpha <= 0:
            raise ValueError(f"alpha must be greater than 0 (got {alpha})")

        self.inner = inner
        self.lora_a = nn.Linear(inner.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, inner.out_features, bias=False)
        self.dropout = nn.Dropout(p_dropout)

        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        self.inner.requires_grad_(False)
        nn.init.trunc_normal_(self.lora_a.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_inner = self.inner(x)
        y_lora = self.scale * self.lora_b(self.lora_a(self.dropout(x)))
        return y_inner + y_lora


class TIMMBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        model_output_format: Literal["CHW", "HWC"] = "CHW",
        layer_freeze_patterns: Sequence[str] | None = None,
        apply_lora: bool = False,
        lora_rules: Sequence[LoRARule] | None = None,
        **kwargs,
    ):
        super().__init__()

        if layer_freeze_patterns and apply_lora:
            warnings.warn(
                "layer_freeze_patterns and apply_lora are mutually exclusive but both are set - "
                "ignoring layer_freeze_patterns.",
                stacklevel=2,
            )

        if layer_freeze_patterns is None:
            layer_freeze_patterns = []

        match model_output_format:
            case "CHW":
                self.forward = self._chw_forward
            case "HWC":
                self.forward = self._hwc_forward
            case _:
                raise ValueError(f"Unsupported model_output_format: {model_output_format}")

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            **kwargs,
        )
        self.model_output_format = model_output_format
        if apply_lora:
            validated_lora_rules = _validate_lora_rules(lora_rules)
            # First, freeze everything, then inject lora layers.
            for param in self.parameters():
                param.requires_grad = False
            _inject_lora_layers(self, rules=validated_lora_rules)
        else:
            for name, param in self.named_parameters():
                for pattern in layer_freeze_patterns:
                    if re.match(pattern, name):
                        param.requires_grad = False
                        break

    @property
    def channels(self) -> Sequence[int]:
        return self.model.feature_info.channels()

    @property
    def reduction(self) -> Sequence[int]:
        return self.model.feature_info.reduction()

    def _chw_forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_maps = self.model(x)
        return {str(i): feature_map for i, feature_map in enumerate(feature_maps)}

    def _hwc_forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_maps = self.model(x)
        feature_maps = [rearrange(fm, "b h w c -> b c h w") for fm in feature_maps]
        return {str(i): feature_map for i, feature_map in enumerate(feature_maps)}


def _validate_lora_rules(rules: Sequence[LoRARule] | None) -> list[_ValidatedLoRARule]:
    if not rules:
        raise ValueError("LoRA requires non-empty lora_rules.")

    validated_rules: list[_ValidatedLoRARule] = []
    for rule_idx, rule in enumerate(rules):
        patterns = rule.get("patterns")
        if not patterns:
            raise ValueError(f"LoRA rule at index {rule_idx} must include non-empty patterns.")

        compiled_patterns: list[re.Pattern[str]] = []
        normalized_patterns: list[str] = []
        for pattern_idx, pattern in enumerate(patterns):
            if not pattern:
                raise ValueError(f"LoRA rule at index {rule_idx} has an empty pattern at index {pattern_idx}.")
            try:
                compiled_patterns.append(re.compile(pattern))
            except re.error as e:
                raise ValueError(f"LoRA rule at index {rule_idx} has invalid regex pattern '{pattern}': {e}") from e
            normalized_patterns.append(pattern)

        rank = rule["rank"]
        if rank <= 0:
            raise ValueError(f"LoRA rule at index {rule_idx} must have rank > 0 (got {rank}).")

        alpha = rule.get("alpha")
        if alpha is not None and alpha <= 0:
            raise ValueError(f"LoRA rule at index {rule_idx} must have alpha > 0 if set (got {alpha}).")

        p_dropout = rule.get("p_dropout", 0.0)
        if p_dropout < 0.0 or p_dropout >= 1.0:
            raise ValueError(f"LoRA rule at index {rule_idx} must satisfy 0 <= p_dropout < 1 (got {p_dropout}).")

        validated_rules.append(
            _ValidatedLoRARule(
                patterns=tuple(normalized_patterns),
                compiled_patterns=tuple(compiled_patterns),
                rank=rank,
                alpha=alpha,
                p_dropout=p_dropout,
            )
        )

    return validated_rules


def _inject_lora_layers(backbone: nn.Module, rules: Sequence[_ValidatedLoRARule]) -> None:
    num_injected = 0

    def _convert(module: nn.Module, path: str) -> None:
        nonlocal num_injected
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            if isinstance(child, nn.Linear):
                matching_rule_indices = [
                    rule_idx
                    for rule_idx, rule in enumerate(rules)
                    if any(pattern.match(child_path) for pattern in rule.compiled_patterns)
                ]
                if len(matching_rule_indices) > 1:
                    matched_patterns = [rules[idx].patterns for idx in matching_rule_indices]
                    raise ValueError(
                        f"Linear layer '{child_path}' matched multiple LoRA rules at indices "
                        f"{matching_rule_indices}: {matched_patterns}"
                    )
                if len(matching_rule_indices) == 1:
                    matched_rule = rules[matching_rule_indices[0]]
                    setattr(
                        module,
                        name,
                        LoRALinear(
                            child, r=matched_rule.rank, alpha=matched_rule.alpha, p_dropout=matched_rule.p_dropout
                        ),
                    )
                    num_injected += 1
            elif not isinstance(child, LoRALinear):
                _convert(child, child_path)

    _convert(backbone, path="")
    if num_injected == 0:
        raise ValueError("No linear layers matched LoRA rules.")


__all__ = ["LoRALinear", "LoRARule", "TIMMBackbone"]
