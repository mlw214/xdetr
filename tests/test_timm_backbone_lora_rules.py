from __future__ import annotations

import pytest
from torch import nn

from xdetr.blocks import LoRALinear, TIMMBackbone

TIMM_TEST_MODEL = "vit_tiny_patch16_224"
TIMM_TEST_OUT_INDICES = (2, 5, 8, 11)


def _module_by_name(model: nn.Module, name: str) -> nn.Module:
    return dict(model.named_modules())[name]


def test_injects_single_matching_rule():
    model = TIMMBackbone(
        model_name=TIMM_TEST_MODEL,
        pretrained=False,
        out_indices=TIMM_TEST_OUT_INDICES,
        apply_lora=True,
        lora_rules=[{"patterns": [r"^model\.model\.blocks\.0\.attn\.qkv$"], "rank": 4}],
    )

    qkv = _module_by_name(model, "model.model.blocks.0.attn.qkv")
    proj = _module_by_name(model, "model.model.blocks.0.attn.proj")
    assert isinstance(qkv, LoRALinear)
    assert isinstance(proj, nn.Linear)
    assert qkv.r == 4


def test_raises_on_multiple_matching_rules():
    with pytest.raises(ValueError, match="matched multiple LoRA rules"):
        TIMMBackbone(
            model_name=TIMM_TEST_MODEL,
            pretrained=False,
            out_indices=TIMM_TEST_OUT_INDICES,
            apply_lora=True,
            lora_rules=[
                {"patterns": [r"^model\.model\..*attn\.qkv$"], "rank": 4},
                {"patterns": [r"^model\.model\.blocks\.0\.attn\.qkv$"], "rank": 8},
            ],
        )


def test_raises_on_zero_total_matches():
    with pytest.raises(ValueError, match="No linear layers matched LoRA rules"):
        TIMMBackbone(
            model_name=TIMM_TEST_MODEL,
            pretrained=False,
            out_indices=TIMM_TEST_OUT_INDICES,
            apply_lora=True,
            lora_rules=[{"patterns": [r"^model\.model\..*does_not_exist$"], "rank": 4}],
        )


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        ([{"patterns": [r"^model\.model\..*attn\.qkv$"], "rank": 0}], "rank > 0"),
        ([{"patterns": [r"^model\.model\..*attn\.qkv$"], "rank": 4, "alpha": 0.0}], "alpha > 0"),
        ([{"patterns": [r"^model\.model\..*attn\.qkv$"], "rank": 4, "p_dropout": 1.0}], "0 <= p_dropout < 1"),
        ([{"patterns": [r"(invalid"], "rank": 4}], "invalid regex pattern"),
        ([{"patterns": [], "rank": 4}], "non-empty patterns"),
    ],
)
def test_rule_validation_rejects_bad_rank_alpha_dropout(rules: list[dict], message: str):
    with pytest.raises(ValueError, match=message):
        TIMMBackbone(
            model_name=TIMM_TEST_MODEL,
            pretrained=False,
            out_indices=TIMM_TEST_OUT_INDICES,
            apply_lora=True,
            lora_rules=rules,
        )


@pytest.mark.parametrize("rules", [None, []])
def test_apply_lora_requires_non_empty_rules(rules: list[dict] | None):
    with pytest.raises(ValueError, match="non-empty lora_rules"):
        TIMMBackbone(
            model_name=TIMM_TEST_MODEL,
            pretrained=False,
            out_indices=TIMM_TEST_OUT_INDICES,
            apply_lora=True,
            lora_rules=rules,
        )


def test_apply_lora_freezes_base_and_keeps_lora_trainable():
    model = TIMMBackbone(
        model_name=TIMM_TEST_MODEL,
        pretrained=False,
        out_indices=TIMM_TEST_OUT_INDICES,
        apply_lora=True,
        lora_rules=[{"patterns": [r"^model\.model\.blocks\.0\.attn\.qkv$"], "rank": 4}],
    )

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable_names
    assert all(".lora_" in name for name in trainable_names)

    qkv = _module_by_name(model, "model.model.blocks.0.attn.qkv")
    proj = _module_by_name(model, "model.model.blocks.0.attn.proj")
    assert isinstance(qkv, LoRALinear)
    assert isinstance(proj, nn.Linear)
    assert qkv.inner.weight.requires_grad is False
    assert proj.weight.requires_grad is False
