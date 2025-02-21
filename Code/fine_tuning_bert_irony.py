"""
German Irony Detection Module

This module provides functionality for training and evaluating irony detection models
specifically designed for German text. It uses BERT-based models and includes support
for k-fold cross-validation and model evaluation.

The module can handle multiple data sources and provides comprehensive evaluation metrics
and visualizations for model performance analysis.

Author: Jacob Schildknecht
Date: February 2025
"""


import os
import re
import pandas as pd
import numpy as np
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.trainer_callback import TrainerCallback
from sklearn.model_selection import KFold

import logging

# Add at the start of the script after imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/irony_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """
    Set seeds for reproducibility across all random number generators.
    
    Args:
        seed (int): Seed value for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

class IronyDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for irony detection data.
    
    This class handles data preprocessing, tokenization, and provides the necessary
    interface for PyTorch data loading.
    """
    def __init__(
            self,
            texts,
            labels,
            max_length: int = 128
    ):
        """
        Initialize the dataset with texts and labels.
        
        Args:
            texts (list): List of input texts
            labels (list): List of corresponding labels
            max_length (int): Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels

        # Compile regex patterns for text cleaning
        self.clean_chars = re.compile(r'[^A-Za-züöäÖÜÄß\s.,!?-]', re.MULTILINE)
        self.clean_http_urls = re.compile(r'https*\S+', re.MULTILINE)
        self.clean_at_mentions = re.compile(r'@\S+', re.MULTILINE)
        self.clean_question_marks = re.compile(r'\?{2,}', re.MULTILINE)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
        
        # Clean and tokenize all texts
        cleaned_texts = [self.clean_text(text) for text in self.texts]
        self.encodings = self.tokenizer(
            cleaned_texts, 
            return_tensors='pt', 
            truncation=True, 
            padding=True,
            max_length=max_length
        )
        
        self.input_ids = self.encodings['input_ids']
        self.attention_mask = self.encodings['attention_mask']
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize input text.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        text = str(text).replace("\n", " ")
        text = self.clean_http_urls.sub('', text)
        text = self.clean_at_mentions.sub('', text)
        text = self.clean_question_marks.sub('?', text)
        text = text.replace('??', '')
        text = ' '.join(text.split())
        text = text.strip()
        return text

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels
        """
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
        return item

    def __len__(self):
        """Get the total size of the dataset."""
        return len(self.labels)

class MetricsCallback(TrainerCallback):
    """
    Callback to compute and store metrics during training.
    
    This callback tracks various performance metrics throughout the training process
    and can be used for monitoring and analysis.
    """
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
        self.metrics_history = {
            'train_loss': [],
            'eval_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Compute and store metrics at the end of each epoch.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            **kwargs: Additional keyword arguments
        """
        # Get predictions on validation set
        eval_pred = self._trainer.predict(self._trainer.eval_dataset)
        predictions = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average='binary'
        )
        
        # Store metrics
        self.metrics_history['accuracy'].append(accuracy)
        self.metrics_history['precision'].append(precision)
        self.metrics_history['recall'].append(recall)
        self.metrics_history['f1'].append(f1)
        
        # Print current metrics
        logger.info(f"\nEpoch {state.epoch:.1f} Metrics:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

def save_model_with_tokenizer(model, tokenizer, save_path):
    """
    Save both the model and tokenizer to disk.
    
    Args:
        model: The trained model
        tokenizer: The associated tokenizer
        save_path (str): Directory path for saving
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Model and tokenizer saved to {save_path}")

def load_model_with_tokenizer(load_path):
    """
    Load a saved model and tokenizer.
    
    Args:
        load_path (str): Path to the saved model directory
        
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        model = AutoModelForSequenceClassification.from_pretrained(load_path)
        tokenizer = AutoTokenizer.from_pretrained(load_path)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model from {load_path}: {str(e)}")
        raise

def predict_batch(texts, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Make predictions for a batch of texts using the trained model.
    
    Args:
        texts (list): List of input texts to classify
        model: The trained model for prediction
        tokenizer: The associated tokenizer
        device (str): Device to run predictions on ('cuda' or 'cpu')
        
    Returns:
        list: List of dictionaries containing prediction results with format:
            {
                'text': original text,
                'is_ironic': boolean prediction,
                'confidence': confidence score
            }
    """
    model = model.to(device)
    model.eval()
    
    # Tokenize texts
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    ).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    # Convert to list of dictionaries
    results = []
    for text, pred in zip(texts, predictions):
        predicted_class = torch.argmax(pred).item()
        confidence = pred[predicted_class].item()
        results.append({
            'text': text,
            'is_ironic': bool(predicted_class),
            'confidence': confidence
        })
    
    return results

def test_model_interactive(model, tokenizer):
    """
    Provide an interactive command line interface for testing the model.
    
    Args:
        model: The trained model for prediction
        tokenizer: The associated tokenizer
        
    The function runs until the user types 'quit', allowing them to input
    text and see the model's predictions in real time.
    """
    logger.info("\nInteractive Testing Mode")
    logger.info("Enter text to classify (or 'quit' to exit)")
    
    while True:
        text = input("\nEnter text: ")
        if text.lower() == 'quit':
            break
            
        result = predict_batch([text], model, tokenizer)[0]
        logger.info(f"\nPrediction: {'Ironic' if result['is_ironic'] else 'Not Ironic'}")
        logger.info(f"Confidence: {result['confidence']:.4f}")

def evaluate_test_set(model, tokenizer, test_texts, test_labels):
    """
    Evaluate model performance on a test set and print detailed metrics.
    
    Args:
        model: The trained model for prediction
        tokenizer: The associated tokenizer
        test_texts (list): List of texts in the test set
        test_labels (list): List of true labels for the test set
        
    Prints:
        - Overall accuracy, precision, recall, and F1 scores
        - Examples of predictions with confidence scores
    """
    predictions = predict_batch(test_texts, model, tokenizer)
    pred_labels = [1 if p['is_ironic'] else 0 for p in predictions]
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, pred_labels, average='binary')
    
    # Print results
    logger.info("\nTest Set Evaluation:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Print some example predictions
    logger.info("\nExample Predictions:")
    for i, (text, pred, true_label) in enumerate(zip(test_texts, predictions, test_labels)):
        if i < 25:  # Show first 5 examples
            logger.info(f"\nText: {text}")
            logger.info(f"Predicted: {'Ironic' if pred['is_ironic'] else 'Not Ironic'} (Confidence: {pred['confidence']:.4f})")
            logger.info(f"Actual: {'Ironic' if true_label else 'Not Ironic'}")


def load_data(file_path='data/irony_dataset.csv'):
    """Load data with proper encoding for German characters"""
    try:
        # Try reading with different encodings
        encodings = ['latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                texts = []
                labels = []
                logger.info(f"Trying to read file with {encoding} encoding...")
                
                with open(file_path, 'r', encoding=encoding) as f:
                    # Skip header
                    next(f)
                    for line in f:
                        try:
                            # Split on semicolon, but only on the last occurrence
                            parts = line.strip().rsplit(';', 1)
                            if len(parts) == 2:
                                text, label = parts
                                texts.append(text.strip())
                                labels.append(int(label.strip()))
                        except Exception as e:
                            logger.info(f"Warning: Skipping malformed line: {line.strip()}")
                            continue
                
                logger.info(f"Successfully loaded {len(texts)} samples with {encoding} encoding")
                return pd.DataFrame({'text': texts, 'is_ironic': labels})
                
            except UnicodeDecodeError:
                logger.info(f"Failed to read with {encoding} encoding")
                continue
            
        raise ValueError("Could not read file with any of the attempted encodings")
        
    except Exception as e:
        logger.info(f"Error loading data: {str(e)}")
        raise

def load_combined_data(news_articles_file, news_labels_file, irony_dataset_file='data/irony_dataset.csv'):
    """Load and combine both datasets with source tracking."""
    # Load original irony dataset
    original_data = load_data()
    original_data['source'] = 'original'  # Add source column
    
    # Load news articles
    with open(news_articles_file, 'r', encoding='utf-8') as f:
        news_texts = [line.strip()[1:-1] for line in f if line.strip()]
    
    # Load news labels
    with open(news_labels_file, 'r', encoding='utf-8') as f:
        news_labels = [int(line.strip()) for line in f if line.strip()]
    
    # Create DataFrame for news data
    news_data = pd.DataFrame({
        'text': news_texts,
        'is_ironic': news_labels,
        'source': 'news'  # Add source column
    })
    
    # Combine datasets
    combined_data = pd.concat([
        original_data,
        news_data
    ], ignore_index=True)
    
    logger.info("Dataset sizes:")
    logger.info(f"Original irony dataset: {len(original_data)}")
    logger.info(f"Irony articles: {len(news_data)}")
    logger.info(f"Combined dataset: {len(combined_data)}")
    
    return combined_data

def evaluate_by_source(model, tokenizer, test_data):
    """
    Evaluate model performance separately for each data source.
    
    Args:
        model: The trained model for prediction
        tokenizer: The associated tokenizer
        test_data (pd.DataFrame): Test data containing 'source' column
            
    For each source:
        - logger.infos detailed metrics (accuracy, precision, recall, F1)
        - Generates confusion matrix visualization
        - Saves confusion matrix plot to disk
        
    Handles:
        - Original irony dataset evaluation
        - News article dataset evaluation
        - Missing class cases through error handling
    """
    # Split test data by source
    original_test = test_data[test_data['source'] == 'original']
    news_test = test_data[test_data['source'] == 'news']
    
    sources = {
        'Original Irony Dataset': original_test,
        'Irony Articles': news_test
    }
    
    for source_name, source_data in sources.items():
        try:
            logger.info(f"\n=== Evaluation on {source_name} ===")
            logger.info(f"Number of samples: {len(source_data)}")
            
            texts = source_data['text'].tolist()
            labels = source_data['is_ironic'].tolist()
            
            # Get predictions
            predictions = predict_batch(texts, model, tokenizer)
            predicted_labels = [int(pred['is_ironic']) for pred in predictions]
            
            # Calculate metrics
            cm = confusion_matrix(labels, predicted_labels)
            report = classification_report(labels, predicted_labels, 
                                        target_names=['Not Ironic', 'Ironic'],
                                        output_dict=True)
            
            # print results
            logger.info(f"\nOverall accuracy: {report['accuracy']:.4f}")
            
            logger.info("\nConfusion Matrix:")
            logger.info("                  Predicted Not Ironic  Predicted Ironic")
            logger.info(f"True Not Ironic       {cm[0][0]:^16d}    {cm[0][1]:^14d}")
            logger.info(f"True Ironic           {cm[1][0]:^16d}    {cm[1][1]:^14d}")
            
            logger.info("\nClassification Report:")
            logger.info(f"              Precision    Recall  F1-Score")
            logger.info(f"Not Ironic     {report['Not Ironic']['precision']:.4f}      {report['Not Ironic']['recall']:.4f}    {report['Not Ironic']['f1-score']:.4f}")
            logger.info(f"Ironic         {report['Ironic']['precision']:.4f}      {report['Ironic']['recall']:.4f}    {report['Ironic']['f1-score']:.4f}")
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Not Ironic', 'Ironic'],
                        yticklabels=['Not Ironic', 'Ironic'])
            plt.title(f'Confusion Matrix - {source_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'confusion_matrix_{source_name.lower().replace(" ", "_")}.png')
            plt.close()
        except IndexError:
            logger.info("IndexError: One of the classes is missing in the test set")

def load_additional_dataset(articles_file, labels_file):
    """
    Load an additional dataset with optional source tracking.
    
    Args:
        articles_file (str): Path to the text file containing articles
        labels_file (str): Path to the file containing corresponding labels
    
    Returns:
        pd.DataFrame: DataFrame with 'text', 'is_ironic', and 'source' columns
    """
    try:
        
        # Attempt to read files with multiple encodings
        encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        
        texts = []
        labels = []
        
        for encoding in encodings_to_try:
            try:
                # Read articles
                with open(articles_file, 'r', encoding=encoding) as f:
                    file_texts = [line.strip() for line in f if line.strip()]
                
                # Read labels
                with open(labels_file, 'r', encoding=encoding) as f:
                    file_labels = [int(line.strip()) for line in f if line.strip()]
                
                texts = file_texts
                labels = file_labels
                logger.info(f"Successfully read files with {encoding} encoding")
                break
            except UnicodeDecodeError:
                logger.info(f"Failed to read with {encoding} encoding")
                continue
        
        # Check if we successfully read any data
        if not texts or not labels:
            logger.info("Could not read any data from the files")
            return None
        
        # Ensure texts and labels match in length
        assert len(texts) == len(labels), f"Number of texts ({len(texts)}) and labels ({len(labels)}) must be the same"
        
        # Create DataFrame
        additional_data = pd.DataFrame({
            'text': texts,
            'is_ironic': labels,
            'source': 'news'  # Add source column
        })
        
        logger.info(f"\nAdditional dataset loaded:")
        logger.info(f"  Number of samples: {len(additional_data)}")
        logger.info(f"  Ironic samples: {sum(additional_data['is_ironic'])}")
        logger.info(f"  Non-ironic samples: {len(additional_data) - sum(additional_data['is_ironic'])}")
        
        return additional_data
    
    except Exception as e:
        logger.info(f"Unexpected error loading additional dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Add this function before the main() function
def perform_kfold_cross_validation(data, n_splits=5):
    """
    Perform K-Fold Cross-Validation
    
    Args:
        data (pd.DataFrame): Full dataset
        n_splits (int): Number of folds for cross-validation
    
    Returns:
        dict: Cross-validation results
    """
    # Initialize K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store cross-validation results
    cv_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(data), 1):
        logger.info(f"\n=== Fold {fold} ===")
        
        # Split data
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        
        # Create datasets
        train_dataset = IronyDataset(
            texts=train_data['text'].tolist(),
            labels=train_data['is_ironic'].tolist()
        )
        
        val_dataset = IronyDataset(
            texts=val_data['text'].tolist(),
            labels=val_data['is_ironic'].tolist()
        )
        
        # Initialize model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-german-cased', num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'./results/fold_{fold}',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss"
        )
        
        # Trainer setup
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Train model
        trainer.train()
        
        # Evaluate model on validation set
        predictions = predict_batch(
            val_data['text'].tolist(), 
            model, 
            tokenizer
        )
        
        # Convert predictions
        pred_labels = [1 if p['is_ironic'] else 0 for p in predictions]
        true_labels = val_data['is_ironic'].tolist()
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
        
        # Store results
        cv_results['accuracy'].append(accuracy)
        cv_results['precision'].append(precision)
        cv_results['recall'].append(recall)
        cv_results['f1_score'].append(f1)
        
        logger.info(f"Fold {fold} Metrics:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
    
    # Print cross-validation summary
    logger.info("\nCross-Validation Summary:")
    for metric, values in cv_results.items():
        logger.info(f"{metric.capitalize()}:")
        logger.info(f"  Mean: {np.mean(values):.4f}")
        logger.info(f"  Standard Deviation: {np.std(values):.4f}")
    
    return cv_results

def main(additional_dataset_files=None):
    """
    Main execution function.
    
    Args:
        additional_dataset_files (list, optional): Paths to additional dataset files
    """
    # Set random seed for reproducibility
    set_seed(42)

    # Load your news data
    data = load_combined_data(
        news_articles_file='data/train.txt',
        news_labels_file='data/train_labels.txt'
    )


    if additional_dataset_files:
        logger.info("\n===== Loading Additional Dataset =====")
        additional_data = load_additional_dataset(
            articles_file=additional_dataset_files[0],
            labels_file=additional_dataset_files[1]
        )

    # Perform K-Fold Cross-Validation
    logger.info("\n=== Performing 5-Fold Cross-Validation ===")
    cross_val_results = perform_kfold_cross_validation(data, n_splits=5)
    
    # Split into train/val/test
    train_ratio = 0.7
    val_ratio = 0.15
    
    # Shuffle data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    
    # Split data
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    logger.info("\nSplit sizes:")
    logger.info(f"Training set: {len(train_data)}")
    logger.info(f"Validation set: {len(val_data)}")
    logger.info(f"Test set: {len(test_data)}")
    
    # Initialize model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-german-cased', num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
    
    # Create datasets
    train_dataset = IronyDataset(
        texts=train_data['text'].tolist(),
        labels=train_data['is_ironic'].tolist()
    )
    
    val_dataset = IronyDataset(
        texts=val_data['text'].tolist(),
        labels=val_data['is_ironic'].tolist()
    )
    
    # Initialize trainer with metrics callback
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Add metrics callback
    metrics_callback = MetricsCallback(trainer)
    trainer.add_callback(metrics_callback)
    
    # Train model
    logger.info("\nStarting training...")
    trainer.train()
    
    # Save model and tokenizer
    save_model_with_tokenizer(model, tokenizer, './irony-detector-combined')
    # Evaluate separately
    evaluate_by_source(model, tokenizer, test_data)

    if additional_dataset_files and additional_data is not None:
        logger.info("\n===== Evaluation on Additional Dataset =====")
        logger.info(f"Number of samples: {len(additional_data)}")
        evaluate_by_source(model, tokenizer, additional_data)

if __name__ == "__main__":
    main(['data/test.txt', 
          'data/test_labels.txt'])