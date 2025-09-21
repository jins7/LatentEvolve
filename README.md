<div align="center">
<h1>LatentEvolve: Self-Evolving Test-Time Scaling in Latent Space</h1> 
</div>

## 👋 Overview
We introduces **LatentEvolve** a self-evolving latent test-time scaling framework for large language models, inspired by the Complementary Learning Systems theory. It operates through a **dual-phase evolution**: daytime scaling performs fast, instance-specific latent optimization using episodic traces from previously solved problems, while nighttime scaling consolidates these experiences into a latent weaver that distills procedural knowledge for future tasks. This continual cycle allows models to not only adapt on the fly but also progressively evolve their reasoning capabilities without supervision.

![alt text](assets/framework.png)

## 🚀 Quick Start

### Setup and Dependencies
```
conda create -n LatentEvolve python=3.10
conda activate LatentEvolve
pip install -r requirements.txt
```
### Prepare Data
Taking mathematical reasoning and multiple-choice questions as examples, our data is in the data folder.
### How to Run
```
cd src/scripts
bash daytime_generation_examples.sh # Step 1: Get daytime fine-tuning latent data
bash nighttime_model_examples.sh # Step 2: Fine-tune the latent model using above data
bash evolve_generation_examples.sh # Step 3: Generate evolved answer using latent model
```

## 🙏 Acknowledgement
- We sincerely thank [LatentSeek](https://github.com/bigai-nlco/LatentSeek) for providing the basic implementation for obtaining latent data.
