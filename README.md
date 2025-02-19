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

## Installation
```bash
git clone https://github.com/yourusername/german-irony-detector.git
cd german-irony-detector
pip install -r requirements.txt
```

## Comments/Questions
If you have comments or question, please send an email to jacob.schildknecht@zew.de