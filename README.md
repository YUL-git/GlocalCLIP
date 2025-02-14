# GlocalCLIP
> by [Jiyul Ham](), [Yonggon Jung](), [Jun-Geol Baek]()

### [paper](https://arxiv.org/abs/2411.06071). We will continue to update the paper and code!

## Introduction
Zero-shot anomaly detection (ZSAD) plays a vital role in identifying anomalous patterns in target datasets without relying on training samples. Recently, pre-trained vision-language models demonstrated strong zero-shot performance across various visual tasks, which accelerate ZSAD. However, their focus on learning class semantics makes their direct application to ZSAD challenging. To address this scenario, we propose GlocalCLIP, which uniquely separates global and local prompts and jointly optimizes them. First, we design object-agnostic semantic prompts to detect fine-grained anomalous patterns. Then, we build a global-local branch, where the global prompt is optimized for detecting normal and anomalous patterns, and the local prompt focuses on anomaly localization. Finally, we introduce glocal contrastive learning to enhance the complementary interaction between global and local prompts, enabling more effective detection of anomalous patterns across diverse domains. The generalization performance of GlocalCLIP was demonstrated on 15 real-world datasets from both the industrial and medical domains, achieving superior performance compared to existing methods.

![overview](https://github.com/YUL-git/GlocalCLIP/blob/main/asset/figure3.png)

## Overview of GlocalCLIP
![overview](https://github.com/YUL-git/GlocalCLIP/blob/main/asset/figure2.png)

## Motivation of GlocalCLIP
<p align="center">
  <img src="https://github.com/YUL-git/GlocalCLIP/blob/main/asset/figure1.png" alt="Fig1" width="70%">
</p>

## Quantitative results
![overview](https://github.com/YUL-git/GlocalCLIP/blob/main/asset/table1_2.png)
<p align="center">
  <img src="https://github.com/YUL-git/GlocalCLIP/raw/main/asset/table2.png" alt="Table 2" width="45%">
  <img src="https://github.com/YUL-git/GlocalCLIP/raw/main/asset/table3.png" alt="Table 3" width="45%">
</p>

## Additional qualitative results of GlocalCLIP
![overview](https://github.com/YUL-git/GlocalCLIP/blob/main/asset/figure5.png)
![overview](https://github.com/YUL-git/GlocalCLIP/blob/main/asset/figure6.png)
![overview](https://github.com/YUL-git/GlocalCLIP/blob/main/asset/figure7.png)
![overview](https://github.com/YUL-git/GlocalCLIP/blob/main/asset/figure8.png)
![overview](https://github.com/YUL-git/GlocalCLIP/blob/main/asset/figure9.png)
![overview](https://github.com/YUL-git/GlocalCLIP/blob/main/asset/figure10.png)
![overview](https://github.com/YUL-git/GlocalCLIP/blob/main/asset/figure11.png)
![overview](https://github.com/YUL-git/GlocalCLIP/blob/main/asset/figure12.png)

## Reproducibility
Implementation environment 
* Ubuntu==22.04.1 LTS
* cuda==12.1.0
* cudnn==8
* python==3.10.15
* pytorch==2.5.1

Create anaconda enviornment
```
conda env create --file GlocalCLIP_env.tml
```

First, download MVTecAD and VisA datasets. and then generate json files.
```
cd generate_dataset_json
python mvtec.py
python visa.py
```

Second, run GlocalCLIP python file.
```
bash train.sh
```

## Acknowledgement
🤗 Thanks for the [OpenCLIP](https://github.com/mlfoundations/open_clip), [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP)
