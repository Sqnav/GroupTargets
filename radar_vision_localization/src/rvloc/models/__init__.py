"""Neural arms 2, 3 and 7 of the common 32 MHz comparison."""
from .single_frame import RGBLocalizer, RadarVisionLocalizer, SingleFrameLocalizer
from .world_fusion import WorldFusionLocalizer, WorldModelMemory, MemberState
from .losses import localization_loss
from .checkpoint import save_checkpoint, load_checkpoint


def build_model(group, config=None):
    factories = {2: RGBLocalizer, 3: RadarVisionLocalizer, 7: WorldFusionLocalizer,
                 "2": RGBLocalizer, "3": RadarVisionLocalizer, "7": WorldFusionLocalizer,
                 "rgb": RGBLocalizer, "single_frame": RadarVisionLocalizer,
                 "world_fusion": WorldFusionLocalizer}
    try:
        factory = factories[group]
    except (KeyError, TypeError) as error:
        raise ValueError("Neural model group must be rgb/2, single_frame/3 or world_fusion/7") from error
    return factory(config)


__all__ = ["build_model", "RGBLocalizer", "RadarVisionLocalizer", "SingleFrameLocalizer",
           "WorldFusionLocalizer", "WorldModelMemory", "MemberState", "localization_loss",
           "save_checkpoint", "load_checkpoint"]
