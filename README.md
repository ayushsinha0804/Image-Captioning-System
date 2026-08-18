# Image Captioning System

A working web app that generates natural-language captions for images: upload a photo and get a description back, powered by a pretrained vision-encoder/decoder model (ViT + GPT-2) served through a FastAPI backend, with a simple HTML/JS frontend.

The repo also keeps the original training notebook (VGG16 encoder + LSTM decoder trained from scratch on Flickr8k) under [`notebooks/`](notebooks/) as a reference pipeline — see [About](#about-the-original-notebook) below.

---

## Running the app

### Requirements

- Python 3.10+
- ~2GB free disk space (model weights are downloaded on first run and cached)

### Setup

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
cd backend
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser. Upload an image and click **Generate Caption**.

The first request downloads the pretrained model (`nlpconnect/vit-gpt2-image-captioning`, ~1GB) from Hugging Face and caches it locally — subsequent runs are fast.

### API

- `GET /` — serves the frontend.
- `POST /api/caption` — multipart form upload (`file`), returns `{"caption": "..."}`.
- `GET /api/health` — health check.

### Project structure

```
backend/     FastAPI app (main.py) + requirements.txt
frontend/    Static HTML/CSS/JS UI, served by the backend
notebooks/   Original Flickr8k training notebook (VGG16 + LSTM from scratch)
```

---

## About the original notebook

`notebooks/image_captioning.ipynb` is an end-to-end example of training an image captioning model **from scratch**: a convolutional neural network (CNN) extracts visual features and a recurrent neural network (RNN) generates captions word by word.

The core idea:

- Use a **pre-trained VGG16** model as an **encoder** to convert an input image into a fixed-length feature vector.
- Use an **Embedding + LSTM-based decoder** to generate a textual description conditioned on that visual feature vector.

### Key Features

- **CNN–RNN architecture**
  - Encoder: VGG16 pre-trained on ImageNet, using the second-last fully-connected layer (4096-D) as the image feature vector.
  - Decoder: Embedding layer → LSTM → Dense layers to predict the next word in the caption.

- **Sequence modeling**
  - Captions are tokenized and converted to integer sequences.
  - The model is trained to predict the next word given the image feature vector and the prefix of the caption seen so far.

- **Custom data pipeline**
  - Loads images from a directory and a captions file, builds `image_id → [captions...]`.
  - Cleans and normalizes text (lowercasing, filtering tokens, adding `startseq` / `endseq` tokens).
  - Uses a Keras `Tokenizer` to build the vocabulary and integer sequences.
  - Implements a Python generator to stream batches to the model (avoids loading all sequences into memory).

- **Evaluation**
  - Evaluates the trained model using **BLEU-1** and **BLEU-2** scores on a held-out test split.
  - Achieves reasonable BLEU scores for a tutorial-scale model (BLEU-1 ~0.52, BLEU-2 ~0.29; values vary depending on preprocessing and data).

- **Inference utilities**
  - Greedy decoding: generates captions one word at a time until `endseq` or `max_length` is reached.
  - Includes a helper to visualize an image alongside its ground-truth captions and the model's predicted caption.

### Technologies Used (notebook)

- **Language**: Python
- **Deep Learning**: TensorFlow / Keras
- **Computer Vision**: VGG16 (ImageNet pre-trained)
- **NLP / Evaluation**: Keras `Tokenizer`, `Embedding`, `LSTM`, `nltk.translate.bleu_score`
- **Utilities**: NumPy, pickle, Pillow, Matplotlib, tqdm

### How It Works (High Level)

1. **Feature Extraction** — each image is resized to 224×224 and passed through VGG16; the penultimate (4096-D) layer is the feature representation, cached to `features.pkl`.
2. **Caption Preprocessing** — captions are lowercased, cleaned, and wrapped with `startseq`/`endseq`.
3. **Tokenization & Sequence Building** — a Keras `Tokenizer` builds the vocabulary; `(image, partial_sequence) → next_word` pairs feed a custom generator.
4. **Model Training** — encoder and decoder branches merge and pass through Dense layers to predict a probability distribution over the vocabulary, trained with categorical cross-entropy.
5. **Caption Generation** — starting from `startseq`, the model iteratively predicts the next word until `endseq` or max length.

Requires the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) (images + captions file) to run, and was originally authored/run in Google Colab (paths reference `/content/drive/...`).

### Possible Extensions

- Replace VGG16 with a more modern encoder (e.g. ResNet, EfficientNet, or a vision-language model).
- Implement **beam search** decoding instead of greedy search for better captions.
- Add more robust text cleaning and handling of rare words.
- Fine-tune the deployed web app's model on a custom dataset.
