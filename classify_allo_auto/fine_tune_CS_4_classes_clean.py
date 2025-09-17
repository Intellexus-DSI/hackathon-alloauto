"""
4-Class Code-Switching Detection System for Tibetan Text
Classes:
  0: Non-switching Auto (continuing in Auto mode)
  1: Non-switching Allo (continuing in Allo mode)
  2: Switch TO Auto
  3: Switch TO Allo

Focus: Proximity-aware loss and evaluation with 5-token tolerance
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

        # Skip empty lines and separator lines
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


def extract_segments_with_token_limit(sentences, min_tokens=300, max_tokens=400):
    """
    Extract segments of 300-400 tokens, trying to respect sentence boundaries
    """
    segments = []
    current_segment = []
    current_token_count = 0

    for sentence in sentences:
        # Count tokens in this sentence (simple whitespace tokenization for now)
        sentence_tokens = sentence.split()
        sentence_token_count = len(sentence_tokens)

        # If adding this sentence would exceed max_tokens
        if current_token_count + sentence_token_count > max_tokens:
            # Save current segment if it meets minimum
            if current_token_count >= min_tokens:
                segment_text = ' '.join(current_segment)
                # Check if segment contains switches
                if '<auto>' in segment_text and '<allo>' in segment_text:
                    segments.append({
                        'text': segment_text,
                        'token_count': current_token_count,
                        'has_transition': True
                    })
            # Start new segment with current sentence
            current_segment = [sentence]
            current_token_count = sentence_token_count
        else:
            # Add sentence to current segment
            current_segment.append(sentence)
            current_token_count += sentence_token_count

            # Check if we've reached the target range
            if current_token_count >= min_tokens:
                segment_text = ' '.join(current_segment)
                # Only keep if it has transitions
                if '<auto>' in segment_text and '<allo>' in segment_text:
                    segments.append({
                        'text': segment_text,
                        'token_count': current_token_count,
                        'has_transition': True
                    })
                # Start new segment
                current_segment = []
                current_token_count = 0

    # Handle remaining sentences
    if current_segment and current_token_count >= min_tokens // 2:  # Be lenient for last segment
        segment_text = ' '.join(current_segment)
        if '<auto>' in segment_text and '<allo>' in segment_text:
            segments.append({
                'text': segment_text,
                'token_count': current_token_count,
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

    # Extract segments with token limit (300-400 tokens)
    segments = extract_segments_with_token_limit(sentences, min_tokens=300, max_tokens=400)
    print(f"  Found {len(segments)} segments with transitions (300-400 tokens each)")

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
    print(f"Average tokens per segment: {df['num_tokens'].mean():.1f}")
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
    Create stratified split ensuring file diversity and no data leakage
    """
    print(f"\n=== Creating Train/Val/Test Split ===")

    # Group by source file to prevent leakage
    file_groups = {}
    for file_name in df['source_file'].unique():
        file_data = df[df['source_file'] == file_name].copy()
        file_groups[file_name] = file_data

    # Assign entire files to splits to ensure diversity
    files = list(file_groups.keys())
    np.random.seed(42)
    np.random.shuffle(files)

    n_files = len(files)
    n_train_files = int(n_files * 0.6)  # 60% of files for training
    n_val_files = int(n_files * 0.2)    # 20% of files for validation

    train_files = files[:n_train_files]
    val_files = files[n_train_files:n_train_files + n_val_files]
    test_files = files[n_train_files + n_val_files:]

    # Now split segments within each file group
    train_dfs = []
    val_dfs = []
    test_dfs = []

    for file_name, file_data in file_groups.items():
        # Shuffle segments within file
        file_data = file_data.sample(frac=1, random_state=42).reset_index(drop=True)

        if file_name in train_files:
            # Most segments go to train, some to val for diversity
            n = len(file_data)
            n_train = int(n * 0.85)
            train_dfs.append(file_data.iloc[:n_train])
            val_dfs.append(file_data.iloc[n_train:])

        elif file_name in val_files:
            # Most segments go to val, some to train for diversity
            n = len(file_data)
            n_val = int(n * 0.85)
            val_dfs.append(file_data.iloc[:n_val])
            train_dfs.append(file_data.iloc[n_val:])

        else:  # test_files
            # All segments go to test (no leakage)
            test_dfs.append(file_data)

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

    print(f"\nNo leakage check:")
    train_files_set = set(train_df['source_file'].unique())
    test_files_set = set(test_df['source_file'].unique())
    overlap = train_files_set.intersection(test_files_set)
    print(f"  Files appearing in both train and test: {len(overlap)}")
    if overlap:
        print(f"  Warning: Some files appear in both splits: {overlap}")

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
# PART 3: PROXIMITY-AWARE LOSS FUNCTION
# ============================================================================

class ProximityAwareLoss4Class(nn.Module):
    """
    Loss function that allows predicted switches to be within tolerance distance
    BUT requires matching switch types (auto->allo vs allo->auto)
    """

    def __init__(self, switch_loss_weight=20.0, proximity_tolerance=5, distance_decay=0.8):
        super().__init__()
        self.switch_loss_weight = switch_loss_weight
        self.proximity_tolerance = proximity_tolerance
        self.distance_decay = distance_decay

        # Base class weights
        self.class_weights = torch.tensor([
            1.0,  # Class 0: Non-switch auto
            1.0,  # Class 1: Non-switch allo
            switch_loss_weight,  # Class 2: Switch to auto
            switch_loss_weight   # Class 3: Switch to allo
        ])

    def forward(self, logits, labels):
        batch_size, seq_len, num_classes = logits.shape
        device = logits.device
        self.class_weights = self.class_weights.to(device)

        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, num_classes),
            labels.view(-1),
            weight=self.class_weights,
            reduction='none'
        ).view(batch_size, seq_len)

        valid_mask = (labels != -100).float()

        # Get predictions
        predictions = torch.argmax(logits, dim=-1)

        # Apply proximity-aware adjustments for switch predictions
        proximity_adjusted_loss = ce_loss.clone()

        for b in range(batch_size):
            # Find true switch positions BY TYPE
            true_switches_to_auto = torch.where(labels[b] == 2)[0]
            true_switches_to_allo = torch.where(labels[b] == 3)[0]
            pred_switches_to_auto = torch.where(predictions[b] == 2)[0]
            pred_switches_to_allo = torch.where(predictions[b] == 3)[0]

            # For each predicted "switch to auto", check proximity to true "switch to auto"
            for pred_pos in pred_switches_to_auto:
                pred_class = predictions[b, pred_pos].item()
                if len(true_switches_to_auto) > 0:
                    # Find minimum distance to SAME TYPE of switch
                    distances = torch.abs(true_switches_to_auto - pred_pos)
                    min_distance = torch.min(distances).item()

                    if 0 < min_distance <= self.proximity_tolerance:
                        # Reduce loss based on proximity (closer = less penalty)
                        decay_factor = self.distance_decay ** min_distance
                        proximity_adjusted_loss[b, pred_pos] *= decay_factor
                # If no matching true switches nearby, keep full penalty

            # For each predicted "switch to allo", check proximity to true "switch to allo"
            for pred_pos in pred_switches_to_allo:
                if len(true_switches_to_allo) > 0:
                    # Find minimum distance to SAME TYPE of switch
                    distances = torch.abs(true_switches_to_allo - pred_pos)
                    min_distance = torch.min(distances).item()

                    if 0 < min_distance <= self.proximity_tolerance:
                        # Reduce loss based on proximity (closer = less penalty)
                        decay_factor = self.distance_decay ** min_distance
                        proximity_adjusted_loss[b, pred_pos] *= decay_factor
                # If no matching true switches nearby, keep full penalty

            # For false negatives (missed switches), add penalty
            # Check "switch to auto" misses
            for true_pos in true_switches_to_auto:
                if len(pred_switches_to_auto) == 0:
                    # No predictions of this type at all - full penalty
                    proximity_adjusted_loss[b, true_pos] *= 2.0
                else:
                    # Check if any prediction of SAME TYPE is close
                    distances = torch.abs(pred_switches_to_auto - true_pos)
                    min_distance = torch.min(distances).item()

                    if min_distance > self.proximity_tolerance:
                        # No nearby prediction of same type - add penalty
                        proximity_adjusted_loss[b, true_pos] *= 1.5

            # Check "switch to allo" misses
            for true_pos in true_switches_to_allo:
                if len(pred_switches_to_allo) == 0:
                    # No predictions of this type at all - full penalty
                    proximity_adjusted_loss[b, true_pos] *= 2.0
                else:
                    # Check if any prediction of SAME TYPE is close
                    distances = torch.abs(pred_switches_to_allo - true_pos)
                    min_distance = torch.min(distances).item()

                    if min_distance > self.proximity_tolerance:
                        # No nearby prediction of same type - add penalty
                        proximity_adjusted_loss[b, true_pos] *= 1.5

        # Apply valid mask and compute mean
        total_loss = proximity_adjusted_loss * valid_mask
        loss = total_loss.sum() / valid_mask.sum().clamp(min=1)

        return loss


# ============================================================================
# PART 4: PROXIMITY-AWARE METRICS
# ============================================================================

def evaluate_switch_detection_with_proximity(true_labels, pred_labels, tolerance=5):
    """
    Evaluate switch detection with proximity tolerance and TYPE matching
    Switch types must match: class 2 with class 2, class 3 with class 3
    """
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # Find switch positions BY TYPE
    true_switches_to_auto = np.where(true_labels == 2)[0]  # Switch to Auto
    true_switches_to_allo = np.where(true_labels == 3)[0]  # Switch to Allo
    pred_switches_to_auto = np.where(pred_labels == 2)[0]  # Predicted Switch to Auto
    pred_switches_to_allo = np.where(pred_labels == 3)[0]  # Predicted Switch to Allo

    # Track matches separately by type
    matched_true_to_auto = set()
    matched_pred_to_auto = set()
    matched_true_to_allo = set()
    matched_pred_to_allo = set()

    exact_matches = 0
    proximity_matches = 0

    # Match "Switch to Auto" predictions with "Switch to Auto" ground truth
    for pred_pos in pred_switches_to_auto:
        if len(true_switches_to_auto) > 0:
            distances = np.abs(true_switches_to_auto - pred_pos)
            min_distance = np.min(distances)
            closest_true_idx = np.argmin(distances)
            closest_true_pos = true_switches_to_auto[closest_true_idx]

            # Only count if not already matched
            if closest_true_pos not in matched_true_to_auto:
                if min_distance == 0:
                    exact_matches += 1
                    matched_true_to_auto.add(closest_true_pos)
                    matched_pred_to_auto.add(pred_pos)
                elif min_distance <= tolerance:
                    proximity_matches += 1
                    matched_true_to_auto.add(closest_true_pos)
                    matched_pred_to_auto.add(pred_pos)

    # Match "Switch to Allo" predictions with "Switch to Allo" ground truth
    for pred_pos in pred_switches_to_allo:
        if len(true_switches_to_allo) > 0:
            distances = np.abs(true_switches_to_allo - pred_pos)
            min_distance = np.min(distances)
            closest_true_idx = np.argmin(distances)
            closest_true_pos = true_switches_to_allo[closest_true_idx]

            # Only count if not already matched
            if closest_true_pos not in matched_true_to_allo:
                if min_distance == 0:
                    exact_matches += 1
                    matched_true_to_allo.add(closest_true_pos)
                    matched_pred_to_allo.add(pred_pos)
                elif min_distance <= tolerance:
                    proximity_matches += 1
                    matched_true_to_allo.add(closest_true_pos)
                    matched_pred_to_allo.add(pred_pos)

    # Total counts
    total_true_switches = len(true_switches_to_auto) + len(true_switches_to_allo)
    total_pred_switches = len(pred_switches_to_auto) + len(pred_switches_to_allo)
    total_matched_true = len(matched_true_to_auto) + len(matched_true_to_allo)
    total_matched_pred = len(matched_pred_to_auto) + len(matched_pred_to_allo)

    total_matches = exact_matches + proximity_matches
    missed_switches = total_true_switches - total_matched_true
    false_switches = total_pred_switches - total_matched_pred

    # Calculate metrics
    precision = total_matches / total_pred_switches if total_pred_switches > 0 else 0
    recall = total_matches / total_true_switches if total_true_switches > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Also calculate per-type metrics
    to_auto_precision = len(matched_pred_to_auto) / len(pred_switches_to_auto) if len(pred_switches_to_auto) > 0 else 0
    to_auto_recall = len(matched_true_to_auto) / len(true_switches_to_auto) if len(true_switches_to_auto) > 0 else 0
    to_allo_precision = len(matched_pred_to_allo) / len(pred_switches_to_allo) if len(pred_switches_to_allo) > 0 else 0
    to_allo_recall = len(matched_true_to_allo) / len(true_switches_to_allo) if len(true_switches_to_allo) > 0 else 0

    return {
        'exact_matches': exact_matches,
        'proximity_matches': proximity_matches,
        'total_matches': total_matches,
        'true_switches': total_true_switches,
        'pred_switches': total_pred_switches,
        'missed_switches': missed_switches,
        'false_switches': false_switches,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        # Per-type metrics
        'true_to_auto': len(true_switches_to_auto),
        'true_to_allo': len(true_switches_to_allo),
        'pred_to_auto': len(pred_switches_to_auto),
        'pred_to_allo': len(pred_switches_to_allo),
        'matched_to_auto': len(matched_true_to_auto),
        'matched_to_allo': len(matched_true_to_allo),
        'to_auto_precision': to_auto_precision,
        'to_auto_recall': to_auto_recall,
        'to_allo_precision': to_allo_precision,
        'to_allo_recall': to_allo_recall
    }


def compute_metrics_for_trainer(eval_pred, tolerance=5):
    """
    Compute metrics for the trainer with proximity awareness and TYPE matching
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Flatten all sequences
    all_predictions = predictions.flatten()
    all_labels = labels.flatten()

    # Remove padding
    mask = all_labels != -100
    all_predictions = all_predictions[mask]
    all_labels = all_labels[mask]

    # Basic accuracy
    accuracy = (all_predictions == all_labels).mean()

    # Proximity-aware switch metrics with TYPE matching
    switch_metrics = evaluate_switch_detection_with_proximity(
        all_labels, all_predictions, tolerance=tolerance
    )

    return {
        'accuracy': float(accuracy),
        'switch_precision': float(switch_metrics['precision']),
        'switch_recall': float(switch_metrics['recall']),
        'switch_f1': float(switch_metrics['f1']),
        'true_switches': switch_metrics['true_switches'],
        'pred_switches': switch_metrics['pred_switches'],
        'exact_matches': switch_metrics['exact_matches'],
        'proximity_matches': switch_metrics['proximity_matches'],
        # Per-type metrics
        'to_auto_precision': float(switch_metrics['to_auto_precision']),
        'to_auto_recall': float(switch_metrics['to_auto_recall']),
        'to_allo_precision': float(switch_metrics['to_allo_precision']),
        'to_allo_recall': float(switch_metrics['to_allo_recall']),
        'true_to_auto': switch_metrics['true_to_auto'],
        'true_to_allo': switch_metrics['true_to_allo'],
        'matched_to_auto': switch_metrics['matched_to_auto'],
        'matched_to_allo': switch_metrics['matched_to_allo']
    }


# ============================================================================
# PART 5: CUSTOM TRAINER
# ============================================================================

class ProximityAwareTrainer4Class(Trainer):
    """Custom trainer with proximity-aware loss"""

    def __init__(self, switch_loss_weight=20.0, proximity_tolerance=5,
                 distance_decay=0.8, *args, **kwargs):
        # Handle tokenizer/processing_class
        if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
            kwargs['processing_class'] = kwargs.get('tokenizer')

        super().__init__(*args, **kwargs)

        # Initialize custom loss
        self.proximity_loss = ProximityAwareLoss4Class(
            switch_loss_weight=switch_loss_weight,
            proximity_tolerance=proximity_tolerance,
            distance_decay=distance_decay
        )
        self.proximity_tolerance = proximity_tolerance

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = self.proximity_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Override to use proximity-aware metrics
        self.compute_metrics = lambda eval_pred: compute_metrics_for_trainer(
            eval_pred, tolerance=self.proximity_tolerance
        )
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)


# ============================================================================
# PART 6: EVALUATION AND VISUALIZATION
# ============================================================================

def print_test_examples_proximity(model, tokenizer, test_csv='test_segments.csv',
                                 num_examples=5, tolerance=5):
    """
    Print detailed test examples with proximity-aware evaluation
    Shows TYPE-specific switch matching
    """
    print("\n" + "=" * 80)
    print(f"PROXIMITY-AWARE SWITCH DETECTION ANALYSIS (tolerance={tolerance} tokens)")
    print("Matching requires SAME switch type (auto→allo vs allo→auto)")
    print("=" * 80)

    test_df = pd.read_csv(test_csv)
    device = next(model.parameters()).device
    model.eval()

    label_names = {
        0: 'NonSwitch-Auto',
        1: 'NonSwitch-Allo',
        2: 'SWITCH→Auto',
        3: 'SWITCH→Allo'
    }

    # Sample examples
    sample_indices = np.random.choice(len(test_df), min(num_examples, len(test_df)), replace=False)

    for i, idx in enumerate(sample_indices):
        row = test_df.iloc[idx]
        tokens = row['tokens'].split()
        true_labels = list(map(int, row['labels'].split(',')))

        print(f"\n--- Example {i+1} (from {row['source_file']}) ---")
        print(f"Segment length: {row['num_tokens']} tokens")
        print(f"Expected transitions: {row['num_transitions']}")

        # Get predictions
        tokenizer_output = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(device) for k, v in tokenizer_output.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
            probs = torch.softmax(outputs.logits, dim=2)

        # Align predictions
        word_ids = tokenizer_output.word_ids()
        aligned_predictions = []
        aligned_probs = []
        previous_word_idx = None

        for j, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                pred = predictions[0][j].item()
                prob = probs[0][j].cpu().numpy()
                aligned_predictions.append(pred)
                aligned_probs.append(prob)
            previous_word_idx = word_idx

        # Trim to match
        final_len = min(len(aligned_predictions), len(true_labels), len(tokens))
        aligned_predictions = aligned_predictions[:final_len]
        true_labels = true_labels[:final_len]
        tokens = tokens[:final_len]

        # Evaluate with proximity and type matching
        switch_eval = evaluate_switch_detection_with_proximity(true_labels, aligned_predictions, tolerance)

        # Show switch analysis BY TYPE
        true_to_auto = [(j, 2) for j, l in enumerate(true_labels) if l == 2]
        true_to_allo = [(j, 3) for j, l in enumerate(true_labels) if l == 3]
        pred_to_auto = [(j, 2) for j, l in enumerate(aligned_predictions) if l == 2]
        pred_to_allo = [(j, 3) for j, l in enumerate(aligned_predictions) if l == 3]

        print(f"\nSwitch Detection Summary:")
        print(f"  True SWITCH→Auto: {len(true_to_auto)} at positions {[pos for pos, _ in true_to_auto]}")
        print(f"  True SWITCH→Allo: {len(true_to_allo)} at positions {[pos for pos, _ in true_to_allo]}")
        print(f"  Pred SWITCH→Auto: {len(pred_to_auto)} at positions {[pos for pos, _ in pred_to_auto]}")
        print(f"  Pred SWITCH→Allo: {len(pred_to_allo)} at positions {[pos for pos, _ in pred_to_allo]}")

        print(f"\nMatching Results (TYPE must match):")
        print(f"  Exact matches: {switch_eval['exact_matches']}")
        print(f"  Proximity matches: {switch_eval['proximity_matches']} (within {tolerance} tokens)")
        print(f"  SWITCH→Auto matched: {switch_eval['matched_to_auto']}/{switch_eval['true_to_auto']}")
        print(f"  SWITCH→Allo matched: {switch_eval['matched_to_allo']}/{switch_eval['true_to_allo']}")

        print(f"\nOverall Performance:")
        print(f"  Switch Precision: {switch_eval['precision']:.3f}")
        print(f"  Switch Recall: {switch_eval['recall']:.3f}")
        print(f"  Switch F1: {switch_eval['f1']:.3f}")

        # Detailed distance analysis
        if pred_to_auto or pred_to_allo:
            print(f"\nDistance Analysis (TYPE-SPECIFIC):")

            # Analyze Switch→Auto predictions
            for pred_pos, _ in pred_to_auto:
                if true_to_auto:
                    distances = [abs(pred_pos - true_pos) for true_pos, _ in true_to_auto]
                    min_distance = min(distances)

                    if min_distance == 0:
                        status = "✓ EXACT MATCH (Switch→Auto)"
                    elif min_distance <= tolerance:
                        status = f"✓ PROXIMITY MATCH (±{min_distance}, Switch→Auto)"
                    else:
                        status = f"✗ TOO FAR (±{min_distance})"
                else:
                    status = "✗ FALSE: predicted Switch→Auto but no true Switch→Auto exists"

                print(f"  Pred Switch→Auto at pos {pred_pos}: {status}")

            # Analyze Switch→Allo predictions
            for pred_pos, _ in pred_to_allo:
                if true_to_allo:
                    distances = [abs(pred_pos - true_pos) for true_pos, _ in true_to_allo]
                    min_distance = min(distances)

                    if min_distance == 0:
                        status = "✓ EXACT MATCH (Switch→Allo)"
                    elif min_distance <= tolerance:
                        status = f"✓ PROXIMITY MATCH (±{min_distance}, Switch→Allo)"
                    else:
                        status = f"✗ TOO FAR (±{min_distance})"
                else:
                    status = "✗ FALSE: predicted Switch→Allo but no true Switch→Allo exists"

                print(f"  Pred Switch→Allo at pos {pred_pos}: {status}")

        # Show switch regions only
        all_switch_positions = set([pos for pos, _ in true_to_auto + true_to_allo + pred_to_auto + pred_to_allo])

        if all_switch_positions:
            print(f"\nSwitch Region Details (showing ±3 context):")

            for switch_pos in sorted(all_switch_positions):
                start = max(0, switch_pos - 3)
                end = min(len(tokens), switch_pos + 4)

                print(f"\n  Around position {switch_pos}:")
                for pos in range(start, end):
                    if pos < len(tokens):
                        token = tokens[pos]
                        true_label = label_names[true_labels[pos]]
                        pred_label = label_names[aligned_predictions[pos]]
                        marker = "→" if pos == switch_pos else " "

                        # Check if types match when both are switches
                        if true_labels[pos] >= 2 and aligned_predictions[pos] >= 2:
                            if true_labels[pos] == aligned_predictions[pos]:
                                match = "✓ TYPE MATCH"
                            else:
                                match = "✗ WRONG TYPE"
                        elif true_labels[pos] == aligned_predictions[pos]:
                            match = "✓"
                        else:
                            match = "✗"

                        print(f"    {marker} [{pos:3d}] {token:15s} | True: {true_label:15s} | Pred: {pred_label:15s} | {match}")

    return sample_indices


# ============================================================================
# PART 7: MAIN TRAINING PIPELINE
# ============================================================================

def train_tibetan_code_switching():
    """
    Main training pipeline for Tibetan code-switching detection
    """
    print("=" * 60)
    print("TIBETAN CODE-SWITCHING DETECTION TRAINING")
    print("With Proximity-Aware Loss (5-token tolerance)")
    print("=" * 60)

    # Step 1: Process all files
    print("\nSTEP 1: Processing files...")
    data_dir = 'classify_allo_auto/data'

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory {data_dir} not found!")
        return

    df, all_segments = process_all_files(data_dir)

    if len(df) == 0:
        print("ERROR: No segments with transitions found!")
        return

    # Save processed data
    df.to_csv('all_segments_300_400_tokens.csv', index=False)

    # Step 2: Create train/val/test split
    print("\nSTEP 2: Creating train/val/test split...")
    train_df, val_df, test_df = create_train_val_test_split(df)

    # Step 3: Initialize model
    print("\nSTEP 3: Initializing model...")
    model_name = 'OMRIDRORI/mbert-tibetan-continual-unicode-240k'
    output_dir = './tibetan_code_switching_proximity_model'

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
    print("\nSTEP 5: Starting training with proximity-aware loss...")

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=4,  # Reduced due to larger segments
        per_device_eval_batch_size=4,
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
        report_to=[],
        gradient_accumulation_steps=2,  # Effective batch size of 8
    )

    # Custom trainer with proximity-aware loss
    trainer = ProximityAwareTrainer4Class(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics_for_trainer(eval_pred, tolerance=5),
        switch_loss_weight=20.0,
        proximity_tolerance=5,
        distance_decay=0.8
    )

    # Train the model
    trainer.train()

    # Save final model
    trainer.save_model(f'{output_dir}/final_model')
    tokenizer.save_pretrained(f'{output_dir}/final_model')

    # Step 6: Evaluation
    print("\nSTEP 6: Evaluating on test set with proximity tolerance...")

    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print(f"\n=== Final Test Results (5-token tolerance) ===")
    print(f"Accuracy: {test_results['eval_accuracy']:.3f}")
    print(f"Switch F1: {test_results['eval_switch_f1']:.3f}")
    print(f"Switch Precision: {test_results['eval_switch_precision']:.3f}")
    print(f"Switch Recall: {test_results['eval_switch_recall']:.3f}")
    print(f"Exact Matches: {test_results.get('eval_exact_matches', 0)}")
    print(f"Proximity Matches: {test_results.get('eval_proximity_matches', 0)}")
    print(f"True Switches: {test_results['eval_true_switches']}")
    print(f"Predicted Switches: {test_results['eval_pred_switches']}")

    # Show some examples
    print("\nShowing test examples...")
    print_test_examples_proximity(model, tokenizer, 'test_segments.csv', num_examples=3, tolerance=5)

    print(f"\n=== Training Complete ===")
    print(f"Model saved to: {output_dir}/final_model")
    print(f"Using 5-token proximity tolerance for switch detection")

    return trainer, model, tokenizer, test_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    trainer, model, tokenizer, results = train_tibetan_code_switching()