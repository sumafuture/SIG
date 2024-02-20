
# SIG


----------


This repository contains code for **SIG.**

[SIG: Speaker Identification in Literature via Prompt-Based Generation](https://arxiv.org/abs/2312.14590)

Zhenlin Su, Liyan Xu, Jin Xu, Jiangnan Li, Mingdu Huangfu


___
# 
# Introduction
Speaker identification in literary text aims at identifying the speaker of quotation in narrative genres Our method is identifying the speaker via Prompt-Based Generation: Prompting the generation model like BART to get the generation score to identify the speaker. 

The main idea of SIG is consists of two parts:

 1.  **Prompt designed:** The training input (X) and the training label (Y) are accompanied by the appropriate prompt. Natural language prompt is added after quotation to close to the MLM training method of the pre-trained model. The training tag is preceded by a prefix prompt, such as **"Speaker: Y"**, which allows the model to predict the speaker based on the input and prefix. 
    
 2.  **Classification by Generation:** SIG calculate the generation probability of each candidate speaker to make better use of prior knowledge and limit the range of options for the final answer.
    

![enter description here](./images/workflow.png)

By changing the method of prompt and selection of candidates, SIG can be used for many tasks.



#  Main code Introduction:

 - [config.py](https://github.com/sumafuture/SIG/blob/main/config.py) : Modify parameters here to control training and evaluation, including prompt construction, training parameters, checkpoint and data paths, etc
 -  [train.py](https://github.com/sumafuture/SIG/blob/main/train.py): Code used to train the model
 -  [evaluate.py](https://github.com/sumafuture/SIG/blob/main/evaluate.py): Code used to evaluate the trained model in two ways: direct generation and classification by generation
 - [SIG_test.py](https://github.com/sumafuture/SIG/blob/main/SIG_test_.py): Code used to test the model and to select the appropriate prompt and compare the various methods (Unnecessary)


   
# Requirements:

 - pytorch==1.8.1
 - transformers==4.4.1
 - jieba==0.42.1
 - spacy==3.6.1

# Citation:

``` 
@article{su2023sig,
title={SIG: Speaker Identification in Literature via Prompt-Based Generation},
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Zhenlin Su and Liyan Xu and Jin Xu and Jiangnan Li and Mingdu Huangfu}, 
year={2024}
}
```

# Contact
If you have any problems, raise an issue or contact suzhenlin75@gmail.com