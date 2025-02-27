# German Irony Detector

## Overview
This project implements a machine learning model for detecting irony in German text using fine-tuned BERT by Guhr et al. (2020; https://github.com/oliverguhr/german-sentiment-lib) and the tuning procedure by Lüdke et al. (2021). The datasets used to fine-tune the model are from Schmidt and Harbusch (2023) and from Claude by Anthropic (2025).

## Features
- BERT-based irony classification
- Support for multiple datasets
- Detailed evaluation metrics
- Customizable training pipeline

## Data
The data can be found in the data-folder. It consists of four files:
- irony_dataset.csv: Including the texts and labels from Claude (completely artifically generated)
- subtle_irony_dataset.csv: Including texts using subtle irony and labels from Claude (completely artifically generated)
- train.txt and train_labels.txt: The reduced training dataset by Schmidt and Harbusch (2023) consisting of 200 texts and labels.
- test.txt and test_labels.txt: The reduced testing dataset by Schmidt and Harbusch (2023) consisting of 100 texts and labels.

## Cross-Validation Results

The results using a 5-fold Cross Validation are the following:

| Metric     | Mean    | Standard Deviation |
|:----------:|:-------:|:------------------:|
| Accuracy   | 0.9651  | 0.0154             |
| Precision  | 0.9731  | 0.0117             |
| Recall     | 0.9632  | 0.0198             |
| F1-Score   | 0.9681  | 0.0141             |

For the hold-out dataset by Schmidt and Harbusch (2023) (test.txt and test_labels.txt) the results are as follows:

| Class        | Precision | Recall | F1-Score    |
|:------------:|:---------:|:------:|:-----------:|
| Not Ironic   | 0.8000    | 0.9600 | 0.8727      |
| Ironic       | 0.9500    | 0.7600 | 0.8444      |

## Installation
```bash
git clone https://github.com/yourusername/german-irony-detector.git
cd german-irony-detector
pip install -r requirements.txt
```

The full model can be found on Huggingface: https://huggingface.co/JacobSKN/german-irony-detector

## Further steps
- Train on more samples
- Including positive/negative irony as prediction

## Comments/Questions
If you have comments or question, please send an email to jacob.schildknecht@zew.de