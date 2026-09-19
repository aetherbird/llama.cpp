from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf, logger

from .bailingmoe3 import BailingMoeV3Model
from .qwen3vl import Qwen3VLVisionModel


@ModelBase.register("BailingMoeV3VLForConditionalGeneration")
@ModelBase.example("inclusionAI/Ling-3.0-flash-VL")
class BailingMoeV3VLModel(BailingMoeV3Model):
    model_arch = gguf.MODEL_ARCH.BAILINGMOE3VL

    def index_tensors(self, remote_hf_model_id: str | None = None):
        # hoist text_config before the shared BailingMoeV3 logic runs:
        # ModelBase.__init__ calls this with the raw VL config, where the text
        # dims still live under text_config
        if "text_config" in self.hparams:
            self.hparams = {**self.hparams, **self.hparams["text_config"]}
        # the VL config omits keys flash ships: one shared expert (the shexp
        # tensors are present) and no MTP block
        self.hparams.setdefault("num_shared_experts", 1)
        self.hparams.setdefault("num_nextn_predict_layers", 0)
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        mrope_section = self.hparams.get("mrope_section")
        if mrope_section is None:
            raise ValueError("BailingMoeV3VL requires mrope_section in the config")
        # mrope_section is [t, h, w]; pad to the 4-wide sections array
        self.gguf_writer.add_rope_dimension_sections(list(mrope_section[:3]) + [0])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # Skip vision encoder and projector tensors
        if name.startswith("model.visual.") or name.startswith("linear_proj"):
            return None

        return super().filter_tensors(item)


@ModelBase.register("BailingMoeV3VLForConditionalGeneration")
@ModelBase.example("inclusionAI/Ling-3.0-flash-VL")
class BailingMoeV3VLVisionModel(Qwen3VLVisionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None

        if self.hparams_vision.get("disable_merger_proj") is not True:
            logger.warning("BailingMoeV3VL: expected disable_merger_proj=true, the merger MLP tensors will be mapped anyway")

        # out_hidden_size is the vision encoder output (post spatial merge, pre linear_proj)
        self.image_emb_dim = self.hparams_vision.get("out_hidden_size")
        if self.image_emb_dim is None:
            raise ValueError("BailingMoeV3VL vision config requires out_hidden_size")

    def set_gguf_parameters(self):
        MmprojModel.set_gguf_parameters(self) # skip Qwen3VLVisionModel parameters
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.LING3VL)
        self.gguf_writer.add_vision_use_gelu(True)

        merge_size = self.hparams_vision.get("spatial_merge_size")
        if merge_size is not None:
            self.gguf_writer.add_vision_spatial_merge_size(int(merge_size))

        rms_norm_eps = self.global_config.get("text_config", {}).get("rms_norm_eps", 1e-6)
        self.gguf_writer.add_vision_attention_layernorm_eps(rms_norm_eps)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith("lm_head."):
            return None

        if name.startswith("linear_proj."):
            # top-level projector MLP: linear_proj.0 -> mm.0, linear_proj.2 -> mm.2
            parts = name.split(".")
            if len(parts) != 3:
                raise ValueError(f"Unexpected linear_proj tensor: {name}")
            idx, suffix = int(parts[1]), parts[2]
            name = f"mm.{idx * 2}.{suffix}" if idx == 0 else f"mm.{idx}.{suffix}"
            # the qwen3vl filter keeps only visual.*; skip it for the renamed projector tensors
            return MmprojModel.filter_tensors.__func__(cls, (name, gen))

        if name.startswith("model.visual."):
            name = name.replace("model.visual.", "visual.", 1)

        if not name.startswith("visual."):
            return None

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        assert self.hparams_vision is not None

        if name.startswith("mm.0.") or name.startswith("mm.2."):
            # top-level projector MLP (linear_proj.0 / linear_proj.2, renamed by filter_tensors)
            yield (name, data_torch)
            return

        if name == "visual.merger.norm.weight" or name == "visual.merger.norm.bias":
            # the merger is norm-only for Ling: per-patch LayerNorm before the spatial merge
            new_name = f"mm.input_norm.{name.split('.')[-1]}"
            yield (new_name, data_torch)
            return

        # Ling has no patch bias; the Conv3D split below matches the stock qwen3vl path
        yield from Qwen3VLVisionModel.modify_tensors(self, data_torch, name, bid)
