# German Irony Detector

## Overview
This project implements a machine learning model for detecting irony in German text using fine-tuned BERT by Guhr et al. (2020) and the tuning procedure by Lüdke et al. (2021). The datasets used to fine-tune the model are from Schmidt and Harbusch (2023) and from Claude by Anthropic (2025).

## Features
- BERT-based irony classification
- Support for multiple datasets
- Detailed evaluation metrics
- Customizable training pipeline

## Data
The data can be found in the data-folder. It consists of two files:
- irony_dataset.csv: Including the texts and labels from Claude (completely artifically generated)
- train.txt and train_labels.txt: The reduced training dataset by Schmidt and Harbusch (2023) consisting of 200 texts and labels.
- test.txt and test_labels.txt: The reduced testing dataset by Schmidt and Harbusch (2023) consisting of 100 texts and labels.

## Cross-Validation Results

The results using a 5-fold Cross Validation are the following:

| Metric     | Mean    | Standard Deviation |
|:----------:|:-------:|:------------------:|
| Accuracy   | 0.9405  | 0.0255             |
| Precision  | 0.9490  | 0.0383             |
| Recall     | 0.9360  | 0.0274             |
| F1-Score   | 0.9421  | 0.0274             |

For the hold-out dataset (test.txt and test_labels.txt) the results are as follows:

| Class        | Precision | Recall | F1-Score    |
|:------------:|:---------:|:------:|:-----------:|
| Not Ironic   | 0.6429    | 0.9000 | 0.7500      |
| Ironic       | 0.8333    | 0.5000 | 0.6250      |

## Installation
```bash
git clone https://github.com/yourusername/german-irony-detector.git
cd german-irony-detector
pip install -r requirements.txt
```

## Further steps
- Train on more samples
- Including positive/negative irony as prediction

## Comments/Questions
If you have comments or question, please send an email to jacob.schildknecht@zew.de