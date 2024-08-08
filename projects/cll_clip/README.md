# CLL-CLIP

PyTroch implementation of our AAAI'24 paper:
> **Embracing Language Inclusivity and Diversity in CLIP Through Continual Language Learning** <br>
> Bang Yang, Yong Dai, Xuxin Cheng, Yaowei Li, Asif Raza, Yuexian Zou <br>
[![Paper](https://img.shields.io/badge/Paper-AAAI'24-red)](https://ojs.aaai.org/index.php/AAAI/article/view/28466)
[![Paper](https://img.shields.io/badge/Paper-arXiv_(with_appendix)-green)](https://arxiv.org/abs/2401.17186)


> **Abstract:** While vision-language pre-trained models (VL-PTMs) have advanced multimodal research in recent years, their mastery in a few languages like English restricts their applicability in broader communities. To this end, there is an increasing interest in developing multilingual VL models via a joint-learning setup, which, however, could be unrealistic due to expensive costs and data availability. In this work, we propose to extend VL-PTMs' language capacity by continual language learning (CLL), where a model needs to update its linguistic knowledge incrementally without suffering from catastrophic forgetting (CF). We begin our study by introducing a model dubbed CLL-CLIP, which builds upon CLIP, a prevailing VL-PTM that has acquired image-English text alignment. Specifically, CLL-CLIP contains an expandable token embedding layer to handle linguistic differences. It solely trains token embeddings to improve memory stability and is optimized under cross-modal and cross-lingual objectives to learn the alignment between images and multilingual texts. To alleviate CF raised by covariate shift and lexical overlap, we further propose a novel approach that ensures the identical distribution of all token embeddings during initialization and regularizes token embedding learning during training. We construct a CLL benchmark covering 36 languages based on MSCOCO and XM3600 datasets and then evaluate multilingual image-text retrieval performance. Extensive experiments verify the effectiveness of CLL-CLIP and show that our approach can boost CLL-CLIP, e.g., by 6.7\% in text-to-image average Recall@1 on XM3600, and improve various state-of-the-art methods consistently.

## TOC
- [Update Notes](#update-notes)
- [Environment](#environment)
- [Reproduction](#reproduction)
  - [Data](#data)
  - [Methods](#methods)
  - [Experiments](#experiments)
- [Citation](#citation)

## Update Notes
**[2024-08-08]** Release the code and data

## Environment
```bash
# clone the repo
git clone https://github.com/yangbang18/CLFM

# enter the repo
cd CLFM

# create a new environment (recommended)
conda create -n clfm python==3.8 -y
conda activate clfm

# install a proper version of PyTorch
# see https://pytorch.org/get-started/previous-versions/
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# install this repo
pip install -e .

# enter the project directory
cd projects/cll_clip
```


## Reproduction
### Data
1. Ensure that you have enter the project directory (`cd projects/cll_clip`)
1. Download our data following the instructions in [data/README.md](/projects/cll_clip/data/README.md).
2. Run `bash scripts/prepare.sh` to produce additional data required for running.

### Methods
We reproduce several state-of-the-art continual learning and parameter-efficient fine-tuning methods, whose definition and hyper-parameters can be found in [methods.yaml](/projects/cll_clip/methods.yaml).

### Experiments
```shell
bash reproducibility/run_sota.sh
bash reproducibility/run_ablation_study.sh
bash reproducibility/run_analysis_converge.sh
bash reproducibility/run_analysis_attention.sh
bash reproducibility/run_translate_test.sh
bash reproducibility/run_flickr30k.sh
```
Refer to notebooks in the [reproducibility](/projects/cll_clip/reproducibility/) folder to see how we analyze and plot the results.


## Citation

Please consider citing our paper if our code and data are useful to your work, thanks sincerely!

```bibtex
@inproceedings{yang2024embracing,
  title={Embracing Language Inclusivity and Diversity in CLIP through Continual Language Learning},
  author={Yang, Bang and Dai, Yong and Cheng, Xuxin and Li, Yaowei and Raza, Asif and Zou, Yuexian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={6458--6466},
  year={2024}
}
```
