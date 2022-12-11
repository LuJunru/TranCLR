# Introduction
This is the project repo for our Findings of EMNLP'22 paper: https://arxiv.org/abs/2210.12902. We borrow a large part of codes from the original ESTER dataset repository: https://github.com/PlusLabNLP/ESTER. Pls refer to the official dataset page for dataset details. Only the training and development set are used in our paper.

# Models
## I. Install packages. 
We list the packages in our environment in env.yml file for your reference.

## II. Train and test
### 1. Fine-tuned models.
We provide several fine-tuned models for quick usage.
- Models: https://drive.google.com/drive/folders/1ljx4pgpy0ocyHN6pGBjLTuHaG7Dd0RTu?usp=sharing
- Extractive QA: roberta-large_IO_prefix_transE. Download it to `./output/spanqa/`.
- Generative QA: unifiedqa-t5-large_prefix_transE. Download it to `./output/allenai/`.

### 2. Train from scratch 
Run `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.  
Run `bash ./code/run_span_pred.sh` and `bash ./code/run_ans_generation.sh`.

### 3. Test on dev set
Run `bash ./code/eval_span_pred.sh` and `bash ./code/eval_ans_generation.sh`.
