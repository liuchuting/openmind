# Marigold Pipelines for Computer Vision Tasks

## Overview

Marigold was proposed in [Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation](https://huggingface.co/papers/2312.02145), a CVPR 2024 Oral paper by [Bingxin Ke](http://www.kebingxin.com/), [Anton Obukhov](https://www.obukhov.ai/), [Shengyu Huang](https://shengyuh.github.io/), [Nando Metzger](https://nandometzger.github.io/), [Rodrigo Caye Daudt](https://rcdaudt.github.io/), and [Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en).
The idea is to repurpose the rich generative prior of Text-to-Image Latent Diffusion Models (LDMs) for traditional computer vision tasks.
Initially, this idea was explored to fine-tune Stable Diffusion for Monocular Depth Estimation, as shown in the teaser above.
Later,
- [Tianfu Wang](https://tianfwang.github.io/) trained the first Latent Consistency Model (LCM) of Marigold, which unlocked fast single-step inference;
- [Kevin Qu](https://www.linkedin.com/in/kevin-qu-b3417621b/?locale=en_US) extended the approach to Surface Normals Estimation;
- [Anton Obukhov](https://www.obukhov.ai/) contributed the pipelines and documentation into diffusers (enabled and supported by [YiYi Xu](https://yiyixuxu.github.io/) and [Sayak Paul](https://sayak.dev/)).

The abstract from the paper is:

*Monocular depth estimation is a fundamental computer vision task. Recovering 3D depth from a single image is geometrically ill-posed and requires scene understanding, so it is not surprising that the rise of deep learning has led to a breakthrough. The impressive progress of monocular depth estimators has mirrored the growth in model capacity, from relatively modest CNNs to large Transformer architectures. Still, monocular depth estimators tend to struggle when presented with images with unfamiliar content and layout, since their knowledge of the visual world is restricted by the data seen during training, and challenged by zero-shot generalization to new domains. This motivates us to explore whether the extensive priors captured in recent generative diffusion models can enable better, more generalizable depth estimation. We introduce Marigold, a method for affine-invariant monocular depth estimation that is derived from Stable Diffusion and retains its rich prior knowledge. The estimator can be fine-tuned in a couple of days on a single GPU using only synthetic training data. It delivers state-of-the-art performance across a wide range of datasets, including over 20% performance gains in specific cases. Project page: https://marigoldmonodepth.github.io.*

## How to use

### MarigoldDepthPipeline

```python
from mindone import diffusers
import mindspore

pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", mindspore_dtype=mindspore.float16
)

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
depth = pipe(image)

vis = pipe.image_processor.visualize_depth(depth[0])
vis[0].save("einstein_depth.png")

depth_16bit = pipe.image_processor.export_depth_to_16bit_png(depth[0])
depth_16bit[0].save("einstein_depth_16bit.png")
```

<img src=./images/einstein_depth_16bit.png>

### MarigoldNormalsPipeline

```python
from mindone import diffusers
import mindspore

pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
    "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", mindspore_dtype=mindspore.float16
)

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
normals = pipe(image)

vis = pipe.image_processor.visualize_normals(normals[0])
vis[0].save("einstein_normals.png")
```

<img src=./images/einstein_normals.png>