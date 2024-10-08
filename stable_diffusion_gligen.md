# GLIGEN (Grounded Language-to-Image Generation)

## Overview

The GLIGEN model was created by researchers and engineers from [University of Wisconsin-Madison, Columbia University, and Microsoft](https://github.com/gligen/GLIGEN). The [`StableDiffusionGLIGENPipeline`] and [`StableDiffusionGLIGENTextImagePipeline`] can generate photorealistic images conditioned on grounding inputs. Along with text and bounding boxes with [`StableDiffusionGLIGENPipeline`], if input images are given, [`StableDiffusionGLIGENTextImagePipeline`] can insert objects described by text at the region defined by bounding boxes. Otherwise, it'll generate an image described by the caption/prompt and insert objects described by text at the region defined by bounding boxes. It's trained on COCO2014D and COCO2014CD datasets, and the model uses a frozen CLIP ViT-L/14 text encoder to condition itself on grounding inputs.

The abstract from the [paper](https://huggingface.co/papers/2301.07093) is:

*Large-scale text-to-image diffusion models have made amazing advances. However, the status quo is to use text input alone, which can impede controllability. In this work, we propose GLIGEN, Grounded-Language-to-Image Generation, a novel approach that builds upon and extends the functionality of existing pre-trained text-to-image diffusion models by enabling them to also be conditioned on grounding inputs. To preserve the vast concept knowledge of the pre-trained model, we freeze all of its weights and inject the grounding information into new trainable layers via a gated mechanism. Our model achieves open-world grounded text2img generation with caption and bounding box condition inputs, and the grounding ability generalizes well to novel spatial configurations and concepts. GLIGEN’s zeroshot performance on COCO and LVIS outperforms existing supervised layout-to-image baselines by a large margin.*

## How to use

### StableDiffusionGLIGENPipeline

```python
import mindspore
from mindone.diffusers import StableDiffusionGLIGENPipeline
from mindone.diffusers.utils import load_image

# Generate an image described by the prompt and
# insert objects described by text at the region defined by bounding boxes
pipe = StableDiffusionGLIGENPipeline.from_pretrained(
    "masterful/gligen-1-4-generation-text-box", variant="fp16", mindspore_dtype=mindspore.float16
)

prompt = "a waterfall and a modern high speed train running through the tunnel in a beautiful forest  fall foliage"
boxes = [[0.1387, 0.2051, 0.4277, 0.7090], [0.4980, 0.4355, 0.8516, 0.7266]]
phrases = ["a waterfall", "a modern high speed train running through the tunnel"]

images = pipe(
    prompt=prompt,
    gligen_phrases=phrases,
    gligen_boxes=boxes,
    gligen_scheduled_sampling_beta=1,
    output_type="pil",
    num_inference_steps=50,
)[0]

images[0].save("./gligen-1-4-generation-text-box.jpg")
```

<img src=./images/gligen-1-4-generation-text-box.jpg>

### StableDiffusionGLIGENTextImagePipeline

```python
import mindspore
from mindone.diffusers import StableDiffusionGLIGENTextImagePipeline
from mindone.diffusers.utils import load_image

# Insert objects described by image at the region defined by bounding boxes
pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
    "anhnct/Gligen_Inpainting_Text_Image", mindspore_dtype=mindspore.float16
)

input_image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/livingroom_modern.png"
)
prompt = "a backpack"
boxes = [[0.2676, 0.4088, 0.4773, 0.7183]]
phrases = None
gligen_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/backpack.jpeg"
)

images = pipe(
    prompt=prompt,
    gligen_phrases=phrases,
    gligen_inpaint_image=input_image,
    gligen_boxes=boxes,
    gligen_images=[gligen_image],
    gligen_scheduled_sampling_beta=1,
    output_type="pil",
    num_inference_steps=50,
)[0]

images[0].save("./gligen-inpainting-text-image-box.jpg")
```

<img src=./images/gligen-inpainting-text-image-box.jpg>