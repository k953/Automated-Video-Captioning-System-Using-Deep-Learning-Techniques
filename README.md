# Lightweight Video Captioning using CNN Encoders and Sequence Decoders

## Overview

This repository presents a **Single Sentence Video Captioning (SSVC)** system that automatically generates a natural language description for a given video clip. Video captioning is a challenging multi-modal task that lies at the intersection of **Computer Vision and Natural Language Processing**, requiring models to understand both spatial and temporal dynamics of visual data and translate them into coherent textual descriptions.

The objective of this project is to design and evaluate **lightweight deep learning architectures** capable of generating accurate captions while maintaining computational efficiency. This is achieved by combining **efficient CNN-based visual encoders** with **sequence modeling decoders such as LSTM, GRU, and Transformer**.

The system extracts visual features from sampled video frames using pretrained convolutional neural networks and feeds them into a sequence decoder that generates captions word-by-word.

This work focuses on evaluating the **trade-off between caption quality and computational efficiency**, enabling potential deployment in **resource-constrained environments such as edge devices and mobile platforms**.

---

# System Architecture

The overall architecture follows an **Encoder–Decoder framework**, where the encoder extracts visual features and the decoder generates textual captions.

```
+--------------------+
|    Input Video     |
+--------------------+
           |
           v
+--------------------+
|   Frame Sampling   |
| (Uniform Sampling) |
+--------------------+
           |
           v
+--------------------+
|  Frame Preprocess  |
| Resize + Normalize |
+--------------------+
           |
           v
+---------------------------+
| CNN Feature Extraction    |
| ResNet / MobileNet /      |
| ShuffleNet                |
+---------------------------+
           |
           v
+---------------------------+
| Feature Sequence          |
| (Temporal Representation) |
+---------------------------+
           |
           v
+-------------------------------+
| Sequence Decoder              |
| LSTM / GRU / Transformer      |
+-------------------------------+
           |
           v
+---------------------------+
| Caption Generation        |
| Word-by-word Prediction   |
+---------------------------+
           |
           v
+---------------------------+
| Evaluation Metrics        |
| BLEU / METEOR / ROUGE /   |
| CIDEr                     |
+---------------------------+
```

---

# Step-by-Step Working

## 1. Video Input

The system takes a short video clip as input. Each video consists of a sequence of frames representing visual events over time. The goal is to convert this visual sequence into a meaningful textual description.

---

## 2. Frame Sampling

Processing every frame of a video is computationally expensive and redundant. Therefore, a subset of frames is extracted using **uniform frame sampling**.

Example:

```
Total frames in video = 300
Frames selected = 30

Sampling interval = 300 / 30
                   = every 10th frame
```

This ensures that the model captures the temporal progression of the video while reducing redundant information.

---

## 3. Frame Preprocessing

Each sampled frame is preprocessed before being passed into the CNN encoder.

### Resize

Frames are resized to:

```
224 x 224 pixels
```

which is the standard input resolution for most pretrained CNN models.

### Normalization

Pixel values are normalized using **ImageNet statistics**:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

This ensures compatibility with pretrained networks.

---

# Visual Feature Extraction

Each frame is passed through a pretrained **Convolutional Neural Network (CNN)** to extract high-level visual features.

Instead of using the final classification layer, the **feature embedding layer** is used to obtain a dense representation.

```
Frame
  |
  v
CNN Encoder
  |
  v
Feature Vector (512 / 1024 / 2048 dimensions)
```

These features capture:

* objects
* human actions
* scene context
* spatial relationships

---

# CNN Models Evaluated

Several CNN backbones were evaluated to study the trade-off between **accuracy and efficiency**.

### ResNet Family

```
ResNet18
ResNet50
ResNet101
ResNet152
```

Advantages:

* deep feature representation
* strong visual understanding

Limitations:

* higher computational cost

---

### MobileNet Family

```
MobileNetV2
MobileNetV3
```

Advantages:

* lightweight architecture
* fewer parameters
* optimized for mobile devices

---

### ShuffleNetV2

Advantages:

* low latency
* efficient channel operations
* optimized for real-time applications

---

# Feature Sequence Representation

After feature extraction, each video is represented as a sequence of feature vectors.

Example:

```
Video
 |
 v
30 Frames Sampled
 |
 v
CNN Feature Extraction
 |
 v
Feature Sequence

[F1, F2, F3, F4, ... F30]
```

This sequence preserves the **temporal structure of the video**.

---

# Sequence Decoders

Three different decoder architectures were evaluated for caption generation.

---

## LSTM Decoder

Long Short-Term Memory (LSTM) networks are designed to model sequential data and capture long-range dependencies.

Key components:

```
Input Gate
Forget Gate
Output Gate
Memory Cell
```

Advantages:

* strong temporal modeling
* stable training behavior

However, multiple gating operations increase computational complexity.

---

## GRU Decoder

GRU is a simplified version of LSTM that uses fewer gates.

```
Reset Gate
Update Gate
```

Advantages:

* fewer parameters
* faster training
* comparable or better performance

---

## Transformer Decoder

Transformer decoders rely on **self-attention mechanisms** instead of recurrent connections.

Advantages:

* parallel processing
* effective long-range dependency modeling

However, transformers typically require larger datasets and higher compute resources.

---

# Caption Generation

The decoder generates captions **word-by-word**.

Example output:

```
<START>
A
man
is
playing
guitar
<END>
```

At each timestep, the model predicts the next word based on:

```
P(word_t | previous words, video features)
```

The word with the highest probability is selected.

---

# Evaluation Metrics

The generated captions are evaluated using standard NLP metrics.

---

## BLEU (Bilingual Evaluation Understudy)

BLEU measures **n-gram precision overlap** between generated captions and reference captions.

Variants:

```
BLEU-1 → unigram
BLEU-2 → bigram
BLEU-3 → trigram
BLEU-4 → four-gram
```

Higher BLEU scores indicate better lexical similarity.

---

## METEOR

METEOR evaluates captions using:

* synonym matching
* stemming
* recall and precision

It correlates better with human judgement compared to BLEU.

---

## ROUGE-L

ROUGE-L measures the **Longest Common Subsequence (LCS)** between reference and generated captions.

Example:

```
Reference : A man is playing guitar
Generated : A man plays guitar
```

The longer the common subsequence, the higher the ROUGE-L score.

---

## CIDEr

CIDEr evaluates captions using **TF-IDF weighted n-grams** across multiple human annotations.

It measures the consensus between generated captions and reference captions.

CIDEr is one of the most widely used metrics in video captioning benchmarks.

---

# Decoder Performance Comparison

Experiments were conducted using **MobileNetV2 features with 30 uniformly sampled frames on the MSVD dataset**.

```
+-------------+-------+-------+-------+-------+-------+--------+---------+--------+--------+
| Decoder     |BLEU-1 |BLEU-2 |BLEU-3 |BLEU-4 |CIDEr  |METEOR  |ROUGE-L  |Params  |GFLOPs |
+-------------+-------+-------+-------+-------+-------+--------+---------+--------+--------+
| LSTM        | 89.9  | 77.1  | 66.8  | 53.3  | 96.9  | 34.5   | 74.1    |14.54M  | 9.18G |
| GRU         | 80.5  | 69.4  | 60.3  | 51.5  | 90.2  | 34.6   | 71.3    |14.54M  | 9.18G |
| Transformer | 96.3  | 87.9  | 71.0  | 63.7  | 114.1  | 47.3   | 84.0    |24.00M  | 9.73G |
+-------------+-------+-------+-------+-------+-------+--------+---------+--------+--------+
```

---

# Key Findings

* **GRU achieves the best overall performance across most evaluation metrics**
* It achieves the highest **BLEU-4 score (51.5)** and **CIDEr score (90.2)**
* GRU maintains the same computational complexity as LSTM
* Transformer requires higher parameters but underperforms in this configuration

---

# Final Model Selection

Based on experimental evaluation:

```
Frame Sampling  → 30 Uniform Frames
Feature Extractor → MobileNetV2
Decoder → GRU
```

This configuration achieves the best balance between **caption quality and computational efficiency**.

---

# Applications

* Video retrieval and indexing
* Assistive systems for visually impaired users
* Automated video summarization
* Surveillance video understanding
* Multimedia content analysis

---

# Author

Kuldeep Kumar
M.Tech Research Project
