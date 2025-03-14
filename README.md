# Multilingual Text-to-Speech with CSM 1B [Work in progress]

This repository contains a multilingual adaptation of the CSM 1B text-to-speech model, enabling high-quality speech synthesis across multiple languages.

## Overview

This project extends the [CSM 1B model](https://github.com/SesameAILabs/csm) to support multilingual text-to-speech generation. The implementation includes:

- Language-specific text preprocessing
- Training pipeline for multiple languages
- Inference tools for generating speech in different languages
- Support for Mozilla Common Voice datasets

## Architecture

The system is based on the CSM 1B model architecture which combines:

- A Llama 3.2-1B backbone for text understanding
- A smaller Llama 3.2-100M decoder for audio token generation
- The Mimi codec for audio tokenization and reconstruction

Our multilingual extension adds:
- Language-specific text processors
- Adapted training pipeline for different languages
- Language-aware generation tools

## Features

- **Multilingual Support**: Generate speech in German and more to come
- **Language-Specific Processing**: Custom text normalization for each supported language
- **Complete Training Pipeline**: Scripts for data preparation, model training, and evaluation
- **Easy Inference**: Simple API for generating speech from text in any supported language
- **Configurable**: Language-specific configurations for fine-tuning

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/multilingual-tts.git
cd multilingual-tts

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generating Speech

Generate speech in German:

```bash
python generate_multilingual.py --text "Hallo, wie geht es dir?" --language de --checkpoint ./checkpoints/de/best_model.pt
```

### Training a New Language Model

The training process consists of two steps:

1. **Data Preparation**:
```bash
python prepare_commonvoice.py --input_dir /path/to/cv-corpus-20.0-de --output_dir ./data/de --language de --filter_quality
```

2. **Model Training**:
```bash
python train_multilingual.py --language de --train_csv ./data/de/train_de.tsv --data_dir ./data --checkpoint ckpt.pt
```

### Training Multiple Languages

Use the provided script to train models for multiple languages:

```bash
python train_multilingual_example.py --common_voice_root /path/to/common-voice --languages de --cv_version 20.0
```

## Dataset

This project uses the [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets) dataset for training. Common Voice is a public domain speech dataset available in multiple languages.

To train a model, you need to:

1. Download the appropriate language dataset from the Common Voice website
2. Use the `prepare_commonvoice.py` script to preprocess the data
3. Train the model using the prepared data

## Supported Languages

The following languages are supported with language-specific processing:

- 🇩🇪 German (de)
- 🇫🇷 French (fr) [coming soon]
- 🇪🇸 Spanish (es) [coming soon]

Additional languages can be used with the default text processor.

## Adding a New Language

To add support for a new language:

1. Extend the `LanguageProcessor` class with language-specific normalization
2. Add your new processor to the `get_processor` factory method
3. Create a language configuration file in `language_configs/`

Example for adding Italian support:

```python
class ItalianProcessor(LanguageProcessor):
    """Italian language text processor"""
    
    def __init__(self):
        super().__init__("it", "Italian")
        
        # Italian-specific replacements
        self.replacements = [
            (r'(\d)\.(\d)', r'\1,\2'),  # Convert decimal points to commas
            (r'Sig\.', 'Signore'),      # Expand common abbreviations
            (r'Sig.ra', 'Signora'),
            (r'(\d)(\s*)€', r'\1 euro'),
        ]
    
    def normalize_text(self, text: str) -> str:
        """Italian-specific text normalization"""
        text = super().normalize_text(text)
        
        # Apply Italian-specific replacements
        for pattern, replacement in self.replacements:
            text = re.sub(pattern, replacement, text)
            
        return text
```

## Performance Considerations

- GPU with at least 8GB VRAM recommended for inference
- 16GB+ VRAM recommended for training
- Mixed precision training is supported to reduce memory requirements
- Model checkpoints are approximately 2GB in size per language

## Acknowledgments

This project builds upon:

- [CSM 1B](https://github.com/SesameAILabs/csm) - The base text-to-speech model
- [Mozilla Common Voice](https://commonvoice.mozilla.org) - Multilingual speech datasets
- [Llama 3.2](https://ai.meta.com/blog/meta-llama-3/) - The backbone language model

## License

This project is licensed under the [MIT License](LICENSE).
```
