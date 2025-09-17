"""
4-Class Code-Switching Detection System for Tibetan Text
Classes:
  0: Non-switching Auto (continuing in Auto mode)
  1: Non-switching Allo (continuing in Allo mode)
  2: Switch TO Auto
  3: Switch TO Allo

Focus: Creating segments with transitions, sentence-based boundaries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import re
import docx
import os
import ssl
from typing import List, Tuple, Dict
from collections import defaultdict
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification,
    Trainer
)
from torch.utils.data import Dataset
from pathlib import Path

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
ssl._create_default_https_context = ssl._create_unverified_context

print(f"Number of GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# ============================================================================
# PART 1: DATA PROCESSING
# ============================================================================

def clean_and_normalize_text(text_content):
    """
    Clean text and normalize tags, ignore separator lines
    """
    lines = text_content.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty lines and separator lines (lines with only dashes, underscores, etc.)
        if not line or re.match(r'^[-_=\s]+$', line):
            continue

        # Normalize all tag variations to standard format
        line = re.sub(r'<\s*AUTO\s*>', '<auto>', line, flags=re.IGNORECASE)
        line = re.sub(r'<\s*ALLO\s*>', '<allo>', line, flags=re.IGNORECASE)

        cleaned_lines.append(line)

    # Join with spaces and clean up extra whitespace
    text_content = ' '.join(cleaned_lines)
    text_content = re.sub(r'\s+', ' ', text_content)

    # Ensure spaces around tags for proper splitting
    text_content = re.sub(r'<auto>', ' <auto> ', text_content)
    text_content = re.sub(r'<allo>', ' <allo> ', text_content)
    text_content = re.sub(r'\s+', ' ', text_content)

    return text_content.strip()


def split_into_sentences(text_content):
    """
    Split text into sentences based on . / // boundaries
    """
    # Split on sentence boundaries while preserving the tags
    sentences = []
    current_sentence = ""

    # First split by sentence endings, keeping the delimiters
    parts = re.split(r'(\.|//?)', text_content)

    for i, part in enumerate(parts):
        if part.strip() in ['.', '/', '//']:
            # End of sentence
            if current_sentence.strip():
                current_sentence += part
                sentences.append(current_sentence.strip())
                current_sentence = ""
        else:
            current_sentence += part

    # Add any remaining content
    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    return [s for s in sentences if s.strip()]


def extract_segments_with_transitions(sentences, max_sentences_per_segment=3):
    """
    Extract segments that contain transitions between allo and auto
    """
    segments = []

    # Process sentences in overlapping windows
    for i in range(len(sentences)):
        for window_size in range(1, min(max_sentences_per_segment + 1, len(sentences) - i + 1)):
            segment_sentences = sentences[i:i + window_size]
            segment_text = ' '.join(segment_sentences)

            # Check if this segment contains both <auto> and <allo> tags (transition)
            has_auto = '<auto>' in segment_text
            has_allo = '<allo>' in segment_text

            if has_auto and has_allo:
                # This segment contains a transition
                segments.append({
                    'text': segment_text,
                    'sentence_indices': list(range(i, i + window_size)),
                    'has_transition': True
                })

    return segments


def process_segment_to_tokens_labels(segment_text):
    """
    Convert a segment to tokens and 4-class labels
    """
    # Split by tags while keeping them
    parts = re.split(r'(<auto>|<allo>)', segment_text)

    tokens = []
    labels = []
    current_mode = None

    for i, part in enumerate(parts):
        if part == '<auto>':
            continue
        elif part == '<allo>':
            continue
        elif part.strip():
            words = part.strip().split()

            # Determine what mode this segment should be in
            segment_mode = None
            for j in range(i - 1, -1, -1):
                if parts[j] == '<auto>':
                    segment_mode = 'auto'
                    break
                elif parts[j] == '<allo>':
                    segment_mode = 'allo'
                    break

            if segment_mode is None:
                segment_mode = 'auto'  # Default

            for word_idx, word in enumerate(words):
                tokens.append(word)

                # Determine label based on position and mode change
                if word_idx == 0 and current_mode is not None and current_mode != segment_mode:
                    # This is a switch point
                    label = 2 if segment_mode == 'auto' else 3
                else:
                    # Continuation
                    label = 0 if segment_mode == 'auto' else 1

                labels.append(label)

            current_mode = segment_mode

    return tokens, labels


def process_file_to_segments(file_path, file_type):
    """
    Process a single file and extract segments with transitions
    """
    print(f"Processing {file_type} file: {file_path.name}")

    # Read file content
    if file_type == 'docx':
        doc = docx.Document(str(file_path))
        text_content = ' '.join([para.text for para in doc.paragraphs])
    else:  # txt file
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        text_content = None

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text_content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if text_content is None:
            raise ValueError(f"Could not read {file_path.name} with any encoding")

    # Clean and normalize
    text_content = clean_and_normalize_text(text_content)

    # Split into sentences
    sentences = split_into_sentences(text_content)
    print(f"  Found {len(sentences)} sentences")

    # Extract segments with transitions
    segments = extract_segments_with_transitions(sentences, max_sentences_per_segment=3)
    print(f"  Found {len(segments)} segments with transitions")

    # Convert segments to token-label pairs
    processed_segments = []
    for seg_idx, segment in enumerate(segments):
        tokens, labels = process_segment_to_tokens_labels(segment['text'])

        if len(tokens) > 0:
            # Count transitions in this segment
            num_transitions = sum(1 for l in labels if l in [2, 3])

            processed_segments.append({
                'segment_id': f"{file_path.stem}_{seg_idx}",
                'source_file': file_path.name,
                'file_type': file_type,
                'tokens': tokens,
                'labels': labels,
                'num_tokens': len(tokens),
                'num_transitions': num_transitions,
                'original_text': segment['text']
            })

    print(f"  Processed {len(processed_segments)} valid segments")
    return processed_segments


def process_all_files(data_dir):
    """
    Process all .txt and .docx files in the data directory
    """
    data_path = Path(data_dir)

    # Find all files
    txt_files = list(data_path.glob("*.txt"))
    docx_files = list(data_path.glob("*.docx"))

    all_files = [(f, 'txt') for f in txt_files] + [(f, 'docx') for f in docx_files]

    print(f"\nFound {len(txt_files)} .txt files and {len(docx_files)} .docx files")
    print(f"Total files to process: {len(all_files)}")

    all_segments = []

    for file_path, file_type in all_files:
        try:
            file_segments = process_file_to_segments(file_path, file_type)
            all_segments.extend(file_segments)
        except Exception as e:
            print(f"  ERROR processing {file_path.name}: {e}")
            continue

    print(f"\n=== Processing Complete ===")
    print(f"Total segments with transitions: {len(all_segments)}")

    # Create DataFrame
    segments_data = []
    for segment in all_segments:
        segments_data.append({
            'segment_id': segment['segment_id'],
            'source_file': segment['source_file'],
            'file_type': segment['file_type'],
            'tokens': ' '.join(segment['tokens']),
            'labels': ','.join(map(str, segment['labels'])),
            'num_tokens': segment['num_tokens'],
            'num_transitions': segment['num_transitions'],
            'original_text': segment['original_text']
        })

    df = pd.DataFrame(segments_data)

    # Print statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Total segments: {len(df)}")
    print(f"Total tokens: {df['num_tokens'].sum()}")
    print(f"Total transitions: {df['num_transitions'].sum()}")
    print(f"Files represented: {df['source_file'].nunique()}")

    # Label distribution
    all_labels = []
    for labels_str in df['labels']:
        all_labels.extend([int(l) for l in labels_str.split(',')])

    label_names = ['Non-switch Auto', 'Non-switch Allo', 'Switch to Auto', 'Switch to Allo']
    print(f"\nLabel distribution:")
    for i in range(4):
        count = sum(1 for l in all_labels if l == i)
        percentage = count / len(all_labels) * 100
        print(f"  Class {i} ({label_names[i]}): {count} ({percentage:.2f}%)")

    return df, all_segments


def create_train_val_test_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create stratified split ensuring file diversity and proper shuffling
    """
    print(f"\n=== Creating Train/Val/Test Split ===")

    # Group by source file
    file_groups = {}
    for file_name in df['source_file'].unique():
        file_data = df[df['source_file'] == file_name].copy()
        file_groups[file_name] = file_data

    train_dfs = []
    val_dfs = []
    test_dfs = []

    # Split each file's data to ensure all splits have data from all files
    for file_name, file_data in file_groups.items():
        n = len(file_data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # Shuffle within file
        file_data = file_data.sample(frac=1, random_state=42).reset_index(drop=True)

        train_data = file_data.iloc[:n_train]
        val_data = file_data.iloc[n_train:n_train + n_val]
        test_data = file_data.iloc[n_train + n_val:]

        if len(train_data) > 0:
            train_dfs.append(train_data)
        if len(val_data) > 0:
            val_dfs.append(val_data)
        if len(test_data) > 0:
            test_dfs.append(test_data)

        print(f"  {file_name}: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Combine and shuffle across files
    train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    # Save splits
    train_df.to_csv('train_segments.csv', index=False)
    val_df.to_csv('val_segments.csv', index=False)
    test_df.to_csv('test_segments.csv', index=False)

    print(f"\nFinal split:")
    print(f"  Training: {len(train_df)} segments from {train_df['source_file'].nunique()} files")
    print(f"  Validation: {len(val_df)} segments from {val_df['source_file'].nunique()} files")
    print(f"  Test: {len(test_df)} segments from {test_df['source_file'].nunique()} files")

    return train_df, val_df, test_df


# ============================================================================
# PART 2: DATASET CLASS
# ============================================================================

class CodeSwitchingDataset4Class(Dataset):
    """Dataset for 4-class token-level code-switching."""

    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = row['tokens'].split()
        labels = list(map(int, row['labels'].split(',')))

        encoding = self.tokenize_and_align_labels(tokens, labels)

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(encoding['labels'])
        }

    def tokenize_and_align_labels(self, tokens, labels):
        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

        word_ids = tokenized_inputs.word_ids()
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(labels[word_idx] if word_idx < len(labels) else -100)
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        tokenized_inputs['labels'] = aligned_labels
        return tokenized_inputs


# ============================================================================
# PART 3: LOSS FUNCTION
# ============================================================================

class ProximityAwareLoss4Class(nn.Module):
    """Loss function for 4-class code-switching with proximity awareness."""

    def __init__(self, switch_loss_weight=20.0, false_positive_penalty=5.0):
        super().__init__()

        # Higher weights for switch classes since they're rare but important
        self.class_weights = torch.tensor([
            1.0,  # Class 0: Non-switch auto
            1.0,  # Class 1: Non-switch allo
            switch_loss_weight,  # Class 2: Switch to auto
            switch_loss_weight   # Class 3: Switch to allo
        ])

        self.false_positive_penalty = false_positive_penalty

    def forward(self, logits, labels):
        batch_size, seq_len, num_classes = logits.shape
        device = logits.device

        self.class_weights = self.class_weights.to(device)

        # Cross-entropy loss with class weights
        ce_loss = F.cross_entropy(
            logits.view(-1, num_classes),
            labels.view(-1),
            weight=self.class_weights,
            reduction='none'
        ).view(batch_size, seq_len)

        valid_mask = (labels != -100).float()

        # Additional penalty for false positives on switch classes
        predictions = torch.argmax(logits, dim=-1)
        switch_mask = ((labels == 2) | (labels == 3)).float()
        pred_switch_mask = ((predictions == 2) | (predictions == 3)).float()

        false_positive_mask = (pred_switch_mask * (1 - switch_mask)) * valid_mask
        false_positive_penalty = false_positive_mask * self.false_positive_penalty

        total_loss = (ce_loss + false_positive_penalty) * valid_mask
        loss = total_loss.sum() / valid_mask.sum().clamp(min=1)

        return loss


# ============================================================================
# PART 4: METRICS AND EVALUATION
# ============================================================================

def compute_switch_metrics(predictions, labels):
    """Compute metrics focusing on switch detection"""
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]

    # Basic accuracy
    accuracy = (predictions == labels).mean()

    # Switch detection metrics
    true_switches = (labels == 2) | (labels == 3)
    pred_switches = (predictions == 2) | (predictions == 3)

    # True positives, false positives, false negatives
    tp = (true_switches & pred_switches).sum()
    fp = (~true_switches & pred_switches).sum()
    fn = (true_switches & ~pred_switches).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': float(accuracy),
        'switch_precision': float(precision),
        'switch_recall': float(recall),
        'switch_f1': float(f1),
        'true_switches': int(true_switches.sum()),
        'pred_switches': int(pred_switches.sum()),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def compute_metrics_for_trainer(eval_pred):
    """Compute metrics for the trainer"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Flatten all sequences
    all_predictions = predictions.flatten()
    all_labels = labels.flatten()

    return compute_switch_metrics(all_predictions, all_labels)


# ============================================================================
# PART 5: TRAINER
# ============================================================================

class ProximityAwareTrainer4Class(Trainer):
    """Custom trainer for 4-class system."""

    def __init__(self, switch_loss_weight=20.0, false_positive_penalty=5.0, *args, **kwargs):
        if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
            kwargs['processing_class'] = kwargs.get('tokenizer')

        super().__init__(*args, **kwargs)
        self.proximity_loss = ProximityAwareLoss4Class(
            switch_loss_weight=switch_loss_weight,
            false_positive_penalty=false_positive_penalty
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = self.proximity_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# PART 6: MAIN TRAINING PIPELINE
# ============================================================================

def train_tibetan_code_switching():
    """
    Main training pipeline for Tibetan code-switching detection
    """
    print("=" * 60)
    print("TIBETAN CODE-SWITCHING DETECTION TRAINING")
    print("=" * 60)

    # Step 1: Process all files
    print("\nSTEP 1: Processing files with transitions...")
    data_dir = 'classify_allo_auto/data'

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory {data_dir} not found!")
        return

    df, all_segments = process_all_files(data_dir)

    if len(df) == 0:
        print("ERROR: No segments with transitions found!")
        return

    # Save processed data
    df.to_csv('all_segments_with_transitions.csv', index=False)

    # Step 2: Create train/val/test split
    print("\nSTEP 2: Creating train/val/test split...")
    train_df, val_df, test_df = create_train_val_test_split(df)

    # Step 3: Initialize model
    print("\nSTEP 3: Initializing model...")
    model_name = 'OMRIDRORI/mbert-tibetan-continual-unicode-240k'
    output_dir = './tibetan_code_switching_model'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=4,
        label2id={'non_switch_auto': 0, 'non_switch_allo': 1, 'to_auto': 2, 'to_allo': 3},
        id2label={0: 'non_switch_auto', 1: 'non_switch_allo', 2: 'to_auto', 3: 'to_allo'}
    )

    # Initialize bias for switch classes
    with torch.no_grad():
        model.classifier.bias.data[2] = 2.0  # Switch to auto
        model.classifier.bias.data[3] = 2.0  # Switch to allo

    model = model.to(device)

    # Step 4: Create datasets
    print("\nSTEP 4: Creating datasets...")
    train_dataset = CodeSwitchingDataset4Class('train_segments.csv', tokenizer)
    val_dataset = CodeSwitchingDataset4Class('val_segments.csv', tokenizer)
    test_dataset = CodeSwitchingDataset4Class('test_segments.csv', tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Step 5: Training
    print("\nSTEP 5: Starting training...")

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=15,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model='switch_f1',
        greater_is_better=True,
        warmup_steps=100,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        report_to=[],  # Disable external reporting to avoid DVCLive issues
        push_to_hub=True,  # Enable pushing to Hugging Face Hub
        # hub_model_id="levshechter/tibetan-code-switching-detector",  # Change this to your HF username
        # hub_strategy="end",  # Push only at the end of training
    )

    trainer = ProximityAwareTrainer4Class(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_for_trainer,
        switch_loss_weight=20.0,
        false_positive_penalty=5.0
    )

    # Train the model
    trainer.train()

    # Save final model
    trainer.save_model(f'{output_dir}/final_model')
    tokenizer.save_pretrained(f'{output_dir}/final_model')

    # Step 6: Evaluation
    print("\nSTEP 6: Evaluating on test set...")

    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print(f"\n=== Final Test Results ===")
    print(f"Accuracy: {test_results['eval_accuracy']:.3f}")
    print(f"Switch F1: {test_results['eval_switch_f1']:.3f}")
    print(f"Switch Precision: {test_results['eval_switch_precision']:.3f}")
    print(f"Switch Recall: {test_results['eval_switch_recall']:.3f}")
    print(f"True Switches: {test_results['eval_true_switches']}")
    print(f"Predicted Switches: {test_results['eval_pred_switches']}")

    print(f"\n=== Training Complete ===")
    print(f"Model saved to: {output_dir}/final_model")

    return trainer, model, tokenizer, test_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    trainer, model, tokenizer, results = train_tibetan_code_switching()