# Prediction interface for Cog ⚙️
from cog import BasePredictor, Input, Path
import os
import torch
from PIL import Image
from weights_downloader import WeightsDownloader
from clip_interrogator import Config, Interrogator
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
)

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
MODEL_CACHE="model-cache"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        WeightsDownloader.download_if_not_exists(SDXL_URL, MODEL_CACHE)
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.ci = Interrogator(Config(clip_model_name="ViT-bigG-14/laion2b_s39b_b160k"))
        self.compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2] ,
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )
        self.pipe = pipe.to("cuda")

    def predict(
        self,
        image1: Path = Input(description="Input image #1"),
        image2: Path = Input(description="Input image #2"),
        strength1: float = Input(description="Strength of img1", default=1.0, ge=0.1, le=1.5),
        strength2: float = Input(description="Strength of img2", default=1.0, ge=0.1, le=1.5),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=300, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        scheduler: str = Input(
            description="Scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        seed: int = Input(
            description="Seed. Leave blank to randomize it", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        img1 = Image.open(image1)
        img2 = Image.open(image2)
       
        ci1 = self.ci.interrogate(img1)
        ci2 = self.ci.interrogate(img2)
        
        # compel_proc('("a red cat playing with a ball", "jungle").blend(0.7, 0.8)')
        prompt = '("' + ci1 + '", "' + ci2 + '").blend(' + str(strength1) + ', ' + str(strength2) + ')'
        conditioning, pooled = self.compel(prompt)

        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(self.pipe.scheduler.config)

        image = self.pipe(
            prompt_embeds=conditioning, 
            pooled_prompt_embeds=pooled, 
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        ).images[0]

        output_path = "/tmp/output.png"
        image.save(output_path)
        return Path(output_path)
