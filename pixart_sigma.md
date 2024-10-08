# PixArt-Σ

## Overview

[PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation](https://huggingface.co/papers/2403.04692) is Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li.

The abstract from the paper is:

*In this paper, we introduce PixArt-Σ, a Diffusion Transformer model (DiT) capable of directly generating images at 4K resolution. PixArt-Σ represents a significant advancement over its predecessor, PixArt-α, offering images of markedly higher fidelity and improved alignment with text prompts. A key feature of PixArt-Σ is its training efficiency. Leveraging the foundational pre-training of PixArt-α, it evolves from the ‘weaker’ baseline to a ‘stronger’ model via incorporating higher quality data, a process we term “weak-to-strong training”. The advancements in PixArt-Σ are twofold: (1) High-Quality Training Data: PixArt-Σ incorporates superior-quality image data, paired with more precise and detailed image captions. (2) Efficient Token Compression: we propose a novel attention module within the DiT framework that compresses both keys and values, significantly improving efficiency and facilitating ultra-high-resolution image generation. Thanks to these improvements, PixArt-Σ achieves superior image quality and user prompt adherence capabilities with significantly smaller model size (0.6B parameters) than existing text-to-image diffusion models, such as SDXL (2.6B parameters) and SD Cascade (5.1B parameters). Moreover, PixArt-Σ’s capability to generate 4K images supports the creation of high-resolution posters and wallpapers, efficiently bolstering the production of highquality visual content in industries such as film and gaming.*

## How to use

### PixArtSigmaPipeline

```py
import mindspore
from mindone.diffusers import PixArtSigmaPipeline

# You can replace the checkpoint id with "PixArt-alpha/PixArt-Sigma-XL-2-512-MS" too.
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", mindspore_dtype=mindspore.float16
)

prompt = "A small cactus with a happy face in the Sahara desert."
image = pipe(prompt)[0][0]
image.save("pixart_sigma.jpg")
```

Here is sample outputs.
<img src=./images/pixart_sigma.jpg>