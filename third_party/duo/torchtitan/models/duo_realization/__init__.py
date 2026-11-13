"""Realization namespace; codecs can be imported without trainer dependencies."""


def register():
    from data.dataloader_realization import build_realization_dataloader
    from torchtitan.components.loss_realization import modal_loss
    from torchtitan.components.lr_scheduler import build_lr_schedulers
    from torchtitan.components.optimizer import build_optimizers
    from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

    from .infra.parallelize import parallelize_realization
    from .modules.args import RealizationModelArgs, tiny_args
    from .modules.model import RealizationTransformer

    register_train_spec(
        TrainSpec(
            name="vdm_realization",
            model_cls=RealizationTransformer,
            model_args={"fl2va": RealizationModelArgs(), "tiny": tiny_args()},
            parallelize_fn=parallelize_realization,
            pipelining_fn=None,
            build_optimizers_fn=build_optimizers,
            build_lr_schedulers_fn=build_lr_schedulers,
            build_dataloader_fn=build_realization_dataloader,
            build_tokenizer_fn=None,
            build_loss_fn=lambda _: modal_loss,
        )
    )
