# Order Book Prediction Model

Deep learning model for predicting values from order book data using Transformer and LSTM architectures.

## Features

- Order book feature engineering
- Transformer and LSTM model implementations
- Time series k-fold cross-validation
- TensorBoard integration
- GPU support

## Project Structure

```
├── src/
│   ├── dataloader.py         # Data preprocessing
│   ├── LSTM_utils.py         # LSTM model
│   ├── transformer_utils.py  # Transformer model
│   └── utils.py             # Utility functions
├── inference_model.py        # Model inference
├── k-fold.py                # Cross validation
└── requirements.txt         # Dependencies
```

## Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Run inference:
```bash
python inference_model.py <path_to_csv>
```

## Configuration

Model parameters can be adjusted in the Config class:
- Window size and batch size
- Model architecture (Transformer/LSTM)
- Training parameters (learning rate, epochs)
- Feature reduction options