# Project Context

## Research topic

Study the impact of neural-network quantization on denoising
within a ray-traced Monte Carlo rendering pipeline.

The project is an empirical/system evaluation, not a new
denoiser or new quantization algorithm.

## Research question

How does reduced numerical precision affect:
- denoising/image reconstruction quality
- inference latency
- GPU utilisation
- memory behaviour
- other relevant GPU execution characteristics

when a neural denoiser is deployed in a ray-traced rendering pipeline?

## Core methodology

Use:
1. A prebuilt open-source Monte Carlo ray tracer.
2. A neural ray-tracing denoiser based on OIDN.
3. Separately generated FP32, FP16 and INT8 model variants.
4. A custom C++ benchmarking/evaluation pipeline.
5. NVIDIA Nsight Systems and Nsight Compute for profiling.
6. PSNR and SSIM, with possible additional image-quality metrics.

The ray tracer itself is NOT being developed.
The denoiser is NOT being developed.
The quantization algorithm is NOT being developed.

The original implementation is the experimental pipeline,
automation, benchmarking and data collection.

## Quantization

Use ONNX Runtime's official quantization tooling.

Python is used once/offline to generate the separately
quantized model artifacts.

Expected artifacts:
- FP32.onnx
- FP16.onnx
- INT8.onnx

The actual experimental pipeline is C++.

## C++ implementation

Environment:
- Windows 11
- Clang
- CMake
- CUDA
- ONNX Runtime GPU/CUDA Execution Provider
- OpenEXR
- NVIDIA Nsight Systems
- NVIDIA Nsight Compute
- RTX 5060

## Current implementation milestone

STEP 1:

Build a minimal C++/Clang + CMake application that:

1. Links against a prebuilt ONNX Runtime GPU distribution.
2. Creates `Ort::Env`.
3. Creates `Ort::SessionOptions`.
4. Attaches the CUDA Execution Provider.
5. Loads a small known-good ONNX model.
6. Runs inference on the RTX 5060.

Do NOT integrate the ray tracer, OIDN, OpenEXR,
quantization or Nsight until this works.

## Intended eventual pipeline

Ray tracer
 -> noisy radiance + auxiliary buffers
 -> C++ pipeline
 -> selected ONNX model
 -> CUDA inference
 -> denoised image
 -> PSNR/SSIM
 -> CSV/results

Nsight is used externally to profile the benchmark executable.

## Literature

Required minimum: 12 citations.

Current bibliography includes:
- Bako et al. (2017), KPCN
- Chaitanya et al. (2017), recurrent denoising autoencoder
- Cook, Porter & Carpenter (1984), Distributed Ray Tracing
- Huo & Yoon (2021), deep learning Monte Carlo denoising survey
- Jacob et al. (2018), integer-only quantization
- Kajiya (1986), Rendering Equation
- Kalantari, Bako & Sen (2015), ML filtering of Monte Carlo noise
- Lin et al. (2020), Detail Preserving Neural Network
- Ronneberger et al. (2015), U-Net
- Schied et al. (2017), SVGF
- Wang et al. (2021), Three-Scale Neural Network
- Yen et al. (2025), ML-based Monte Carlo denoising literature review

## Base paper

University requires a SINGLE recent, relevant,
high-quality publication, preferably IEEE/Springer/
Elsevier/ACM/reputed journal.

Potential base paper currently being considered:
Lin et al. (2020), "A Detail Preserving Neural Network
Model for Monte Carlo Denoising."

OIDN itself is a software project rather than the required
peer-reviewed base paper.

## Literature-gap framing

Do NOT claim that nobody has ever studied quantization
of neural denoisers.

Use the more defensible framing:

Existing work has substantially focused on improving
Monte Carlo denoising architectures and reconstruction
quality, while quantization is an established neural
network deployment optimisation. Comparatively little
work has systematically characterised the effects of
inference precision on the quality, performance and
hardware behaviour of neural Monte Carlo denoisers
within ray-traced rendering pipelines.

## Project constraint

Implementation should be feasible in roughly 10 days,
so existing open-source components are deliberately being
used rather than implementing rendering or denoising
algorithms from scratch.