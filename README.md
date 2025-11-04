



# Automated-Video-Captioning-System-Using-Deep-Learning-Techniques

# 🎥 Automated Video Captioning System using Deep Learning Techniques

This project implements **three progressively advanced models** for generating textual captions from videos —  
starting from a basic **LSTM-based sequence-to-sequence (S2VT)** model to an **Attention-enhanced** version and finally a **Transformer-based Decoder model**.

---

## 🧠 Project Overview

The goal of this project is to make machines *watch a video and describe it in natural language* —  
for example:  
> “A man is playing football”  
> “A woman is cooking in the kitchen.”

This system bridges **Computer Vision** and **Natural Language Processing (NLP)**  
by combining **CNN (feature extraction)** and **sequence models (caption generation)**.

---

## ⚙️ Model Architectures

### 🟩 **1️⃣ S2VT – LSTM Encoder–Decoder**

- Based on the paper *"Sequence to Sequence - Video to Text (Venugopalan et al., CVPR 2015)"*
- Uses two stacked LSTMs:
  - **Encoder LSTM:** Encodes frame features sequentially.
  - **Decoder LSTM:** Generates caption words one by one.
- Does *not use attention*; relies on the last hidden state to represent the whole video.

**Architecture Flow:**



Video Frames → CNN (ResNet/I3D) → [40,2048] → Encoder LSTM → Decoder LSTM → Caption


**Pros:** Simple and stable.  
**Cons:** Loses context on long videos.

---

### 🟦 **2️⃣ LSTM + Bahdanau Attention**

- Builds on S2VT by adding **Bahdanau (Additive) Attention** between encoder and decoder.
- Decoder dynamically focuses on the most relevant frames while generating each word.
- Improves contextual understanding and interpretability.

**Architecture Flow:**


Video Frames → CNN → Encoder LSTM → Bahdanau Attention → Decoder LSTM → Caption


**Attention Mechanism:**
\[
\text{context} = \sum_i \alpha_i h_i, \quad \alpha_i = \text{softmax}(v^T \tanh(W_1 s_t + W_2 h_i))
\]

**Pros:** More accurate, interpretable captions.  
**Cons:** Still sequential (slower training).

---

### 🟧 **3️⃣ Transformer Decoder-based Model**

- The final and most powerful model.
- Uses **CNN (ResNet/I3D)** to extract frame features (acts as encoder).
- A **Transformer Decoder** learns temporal attention and global context.
- Adds **Positional Encoding**, **Multi-Head Self-Attention**, and **Label Smoothing Loss**.

**Architecture Flow:**


Video Frames → CNN → Linear Projection → Positional Encoding → Transformer Decoder → Caption


**Advantages:**
- Fully parallelized (faster training).
- Captures long-term dependencies.
- Best accuracy among all models.

**Loss Function:**
CrossEntropy with Label Smoothing  
\[
L = -\sum_i y_i \log(\hat{y}_i)
\]

---

## 📊 Performance Comparison

| Model | BLEU-4 | METEOR | ROUGE-L | CIDEr | Remarks |
|--------|---------|---------|----------|--------|----------|
| S2VT (LSTM) | 0.38 | 0.27 | 0.55 | 0.41 | Basic sequential model |
| LSTM + Attention | 0.51 | 0.33 | 0.61 | 0.67 | Frame-level focus improves accuracy |
| Transformer Decoder | **0.61** | **0.39** | **0.68** | **0.81** | Best context and fluency |

---

## 🧰 Tools & Frameworks

| Component | Library / Model |
|------------|----------------|
| Feature Extraction | ResNet-152 / I3D |
| Deep Learning | PyTorch |
| Dataset | MSVD (YouTubeClips) |
| Preprocessing | ffmpeg, numpy |
| Evaluation | BLEU, METEOR, ROUGE, CIDEr |
| Optimizer | Adam / AdamW |
| Loss | CrossEntropy / Label Smoothing |
| Platform | Google Colab / Drive |

---

## 🧩 Dataset Preparation

1. Download MSVD dataset (YouTubeClips).
2. Extract frames (e.g. 40 per video) using `ffmpeg`.
3. Extract CNN features and save as `.npy`.
4. Split dataset into train / val / test.
5. Build vocabulary from captions.

---

## 🚀 Training Commands

Each model has a separate training notebook:

| Model | File | Description |
|--------|------|-------------|
| S2VT LSTM | `S2VT_Encoder_Decoder_Model.ipynb` | Basic encoder-decoder |
| LSTM + Attention | `attention_cap13may.ipynb` | Adds Bahdanau attention |
| Transformer Decoder | `transferbase_capt.ipynb` | Transformer-based decoder with label smoothing |

---

## 📈 Example Result

**Input Video:** A man playing football  
**Generated Captions:**
- S2VT → “man playing ball”  
- Attention LSTM → “a man is playing football”  
- Transformer → “a man is playing football on the field”

---

## 🔮 Future Scope

- Real-time video captioning (using lightweight models like MobileNet + Transformer)
- Video Question Answering (VQA)
- Multimodal video understanding (audio + vision)
- Captioning for visually impaired assistance systems

---

## 👨‍💻 Contributors

**Developed by:** Kuldeep Kumar  
**Guided by:** [Your Guide/Professor Name]  
**Institute:** [Your College / University Name]

---

## 🏁 References

- Venugopalan et al., *Sequence to Sequence - Video to Text*, CVPR 2015  
- Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, 2015  
- Vaswani et al., *Attention is All You Need*, NIPS 2017  

---

### ⭐ If you like this project, don’t forget to star the repo!

<img width="1536" height="1024" alt="ChatGPT Image Nov 4, 2025, 07_01_42 PM" src="https://github.com/user-attachments/assets/09181994-c3eb-4134-8620-91e7de2038e1" />





