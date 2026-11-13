import pytest

torch = pytest.importorskip("torch")
model_module = pytest.importorskip("duo.model")
torch_model_module = pytest.importorskip("duo.torch_model")
build_joint_layout = model_module.build_joint_layout
DuoModelConfig = torch_model_module.DuoModelConfig
JointMediaTransformer = torch_model_module.JointMediaTransformer


def test_reference_transformer_runs_the_joint_layout():
    layout = build_joint_layout(
        blocks=2,
        reference_tokens=1,
        shared_tokens=1,
        condition_tokens=1,
        media_tokens=2,
    )
    model = JointMediaTransformer(
        DuoModelConfig(input_size=8, hidden_size=16, layers=1, attention_heads=4)
    )
    sequence = layout.allowed.shape[0]
    output = model(
        torch.randn(2, sequence, 8),
        allowed=torch.from_numpy(layout.allowed),
        token_types=torch.zeros(2, sequence, dtype=torch.long),
    )
    assert output.shape == (2, sequence, 8)
