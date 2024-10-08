# Stable Diffusion 3

## Overview

Stable Diffusion 3 (SD3) was proposed in [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206.pdf) by Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach.

The abstract from the paper is:

*Diffusion models create data from noise by inverting the forward paths of data towards noise and have emerged as a powerful generative modeling technique for high-dimensional, perceptual data such as images and videos. Rectified flow is a recent generative model formulation that connects data and noise in a straight line. Despite its better theoretical properties and conceptual simplicity, it is not yet decisively established as standard practice. In this work, we improve existing noise sampling techniques for training rectified flow models by biasing them towards perceptually relevant scales. Through a large-scale study, we demonstrate the superior performance of this approach compared to established diffusion formulations for high-resolution text-to-image synthesis. Additionally, we present a novel transformer-based architecture for text-to-image generation that uses separate weights for the two modalities and enables a bidirectional flow of information between image and text tokens, improving text comprehension typography, and human preference ratings. We demonstrate that this architecture follows predictable scaling trends and correlates lower validation loss to improved text-to-image synthesis as measured by various metrics and human evaluations.*

## How to use

### StableDiffusion3Img2ImgPipeline

```py
import mindspore

from mindone.diffusers import StableDiffusion3Img2ImgPipeline
from mindone.diffusers.utils import load_image

model_id_or_path = "stabilityai/stable-diffusion-3-medium-diffusers"
pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_id_or_path, mindspore_dtype=mindspore.float16)

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
init_image = load_image(url).resize((512, 512))

prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

images = pipe(prompt=prompt, image=init_image, strength=0.95, guidance_scale=7.5)[0][0]
images.save("cat.jpg")
```

<img src=./images/cat.jpg>