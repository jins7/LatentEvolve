<div align="center">
<h1>LatentEvolve: Self-Evolving Test-Time Scaling in Latent Space</h1> 
</div>

## 🚀 Quick Start

### Setup and Dependencies
```
conda create -n LatentEvolve python=3.10
conda activate LatentEvolve
pip install -r requirements.txt
```
### Prepare Data
Prepare datasets in the `datasets` folder.
### How to Run
```
cd src/scripts
bash daytime_generation_examples.sh # Step 1: Get daytime fine-tuning latent data
bash nighttime_model_examples.sh # Step 2: Fine-tune the latent model using above data
bash evolve_generation_examples.sh # Step 3: Generate evolved answer using latent model
```

## 🙏 Acknowledgement
- We sincerely thank [LatentSeek](https://github.com/bigai-nlco/LatentSeek) for providing the basic implementation for obtaining latent data.
