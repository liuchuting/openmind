# T2I-Adapter

## Overview

[T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.08453) by Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, Xiaohu Qie.

Using the pretrained models we can provide control images (for example, a depth map) to control Stable Diffusion text-to-image generation so that it follows the structure of the depth image and fills in the details.

The abstract of the paper is the following:

*The incredible generative ability of large-scale text-to-image (T2I) models has demonstrated strong power of learning complex structures and meaningful semantics. However, relying solely on text prompts cannot fully take advantage of the knowledge learned by the model, especially when flexible and accurate controlling (e.g., color and structure) is needed. In this paper, we aim to ``dig out" the capabilities that T2I models have implicitly learned, and then explicitly use them to control the generation more granularly. Specifically, we propose to learn simple and lightweight T2I-Adapters to align internal knowledge in T2I models with external control signals, while freezing the original large T2I models. In this way, we can train various adapters according to different conditions, achieving rich control and editing effects in the color and structure of the generation results. Further, the proposed T2I-Adapters have attractive properties of practical value, such as composability and generalization ability. Extensive experiments demonstrate that our T2I-Adapter has promising generation quality and a wide range of applications.*

This model was contributed by the community contributor [HimariO](https://github.com/HimariO) ❤️ .

## How to use

### StableDiffusionAdapterPipeline

```python
from PIL import Image
from mindone.diffusers.utils import load_image
import mindspore
from mindone.diffusers import StableDiffusionAdapterPipeline, T2IAdapter

image = load_image(
    "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/color_ref.png"
)

color_palette = image.resize((8, 8))
color_palette = color_palette.resize((512, 512), resample=Image.Resampling.NEAREST)

adapter = T2IAdapter.from_pretrained("TencentARC/t2iadapter_color_sd14v1", mindspore_dtype=mindspore.float16)
pipe = StableDiffusionAdapterPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    adapter=adapter,
    mindspore_dtype=mindspore.float16,
)

out_image = pipe(
    "At night, glowing cubes in front of the beach",
    image=color_palette,
)[0][0]

out_image.save("At night, glowing cubes in front of the beach.png")
```

<img src=./images/At night, glowing cubes in front of the beach.jpg>

### StableDiffusionXLAdapterPipeline

```python
import mindspore
from mindone.diffusers import T2IAdapter, StableDiffusionXLAdapterPipeline, DDPMScheduler
from mindone.diffusers.utils import load_image

sketch_image = load_image("https://huggingface.co/Adapter/t2iadapter/resolve/main/sketch.png").convert("L")

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

adapter = T2IAdapter.from_pretrained(
    "Adapter/t2iadapter",
    subfolder="sketch_sdxl_1.0",
    mindspore_dtype=mindspore.float16,
    adapter_type="full_adapter_xl",
)
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    model_id, adapter=adapter, mindspore_dtype=mindspore.float16, variant="fp16", scheduler=scheduler
)

sketch_image_out = pipe(
    prompt="a photo of a dog in real world, high quality",
    negative_prompt="extra digit, fewer digits, cropped, worst quality, low quality",
    image=sketch_image,
    guidance_scale=7.5,
)[0][0]

sketch_image_out.save("a photo of a dog in real world, high quality.png")
```

<img src=./images/a photo of a dog in real world, high quality.jpg>