# Machine Translation NLP

A Neural Machine Translation project that translates English sentences into Urdu using a Seq2Seq LSTM model with an Attention mechanism implemented in PyTorch.

## Dataset

- OPUS-100 English–Urdu Dataset
- Approximately 34,000 training sentence pairs
- Approximately 3,800 validation sentence pairs
- Source: Hugging Face Datasets

## Model

```text
English Sentence
        │
        ▼
Embedding Layer
        │
        ▼
Encoder LSTM
        │
        ▼
Attention Mechanism
        │
        ▼
Decoder LSTM
        │
        ▼
Fully Connected Layer
        │
        ▼
Urdu Translation
```

### Model Configuration

- Embedding Size: 128
- Hidden Size: 256
- Number of LSTM Layers: 2
- Dropout: 0.3
- Batch Size: 16
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Teacher Forcing: 0.5

## Project Structure

```text
Machine-Translation-NLP
│
├── models
│   ├── attention.py
│   ├── encoder.py
│   ├── decoder.py
│   └── seq2seq.py
│
├── dataset.py
├── vocabulary.py
├── train.py
├── test.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
git clone https://github.com/your-username/Machine-Translation-NLP.git
cd Machine-Translation-NLP

pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Evaluation

```bash
python test.py
```

## Prediction

```bash
python predict.py
```

Example

```text
Enter English sentence: How are you?

Urdu:
کہاں ہو؟
```

## Features

- English to Urdu Translation
- Seq2Seq Architecture
- Attention Mechanism
- Custom Vocabulary
- Teacher Forcing
- Model Checkpoint Saving
- PyTorch Implementation

## Future Improvements

- Transformer Architecture
- SentencePiece Tokenization
- Beam Search Decoding
- BLEU Score Evaluation
- Larger Vocabulary
- Better Translation Quality

## Technologies Used

- Python
- PyTorch
- Hugging Face Datasets
- NLTK

## License

This project is open-source and available under the MIT License.