# AI image detector smoke-test set

This folder contains a small, labeled subset of the public [AIGC Detection Benchmark](https://huggingface.co/datasets/TheKernel01/AIGC-Detection-Benchmark), collected for local image-detector testing.

## Contents

- `ai_generated/`: 25 generated images, five each labeled `DALLE2`, `Midjourney`, `SD15`, `SDXL`, and `Wukong`.
- `real_control/`: 25 real-image controls labeled `Real`.
- `manifest.csv`: expected class, generator label, source dataset row, dimensions, and a stable source link for every file.

The source dataset has 125,026 evaluation rows and labels both real/fake status and generator. This subset is intentionally small; use it for a smoke test, not as a statistically representative benchmark. The generator labels describe the source dataset's model categories; they do not guarantee the exact product version used to create each image.

## Provenance and use

The Hugging Face dataset card declares Apache-2.0 for the dataset. Image-generator terms may also apply. Keep the manifest with the images, and check the relevant source terms before redistributing the images or using them commercially.

Source: [TheKernel01/AIGC-Detection-Benchmark](https://huggingface.co/datasets/TheKernel01/AIGC-Detection-Benchmark)

Original dataset source: [Ekko-zn/AIGCDetectBenchmark](https://github.com/Ekko-zn/AIGCDetectBenchmark)
