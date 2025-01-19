
# Genre Detection System for Tagore Song Lyrics

This document provides an overview of the genre detection system built to classify song lyrics into one of 23 predefined genres. It uses a pretrained **BanglaBERT** model and the Transformers library for fine-tuning and evaluation.

---

## System Overview

The genre detection system is designed to:

1. **Load Data**: Import song lyrics and their associated genres from CSV files.
2. **Preprocess Data**: Tokenize the text and map genre labels to numeric IDs dynamically.
3. **Train a Model**: Fine-tune the pretrained **BanglaBERT** model for the classification task.
4. **Evaluate Performance**: Assess the model's performance using metrics like accuracy, precision, recall, and F1-score.
5. **Save and Reuse**: Save the trained model, tokenizer, and label mappings for deployment.

---

## Project Structure

- `data/train.csv`: Training data with lyrics and genres.
- `data/test.csv`: Testing data with lyrics and genres.
- `label_to_id.json`: JSON file to dynamically store and load label-to-ID mappings.
- `id_to_label.json`: JSON file to dynamically store and load ID_to label mappings.
- `results/`: Directory where the fine-tuned model and tokenizer are saved. This could not be pushed in the repository because of the model size.
- `logs/`: Directory for training logs.
- `genreDetection.py` : Script for training the model and saving results.
- `trainingSetDownload/` : Directory for web scraping scripts and related data.


---

### Inside `trainingSetDownload/`
```
scrapLinkAndCategories.py        # Script to scrape song links and categories from the web.
scraped_song_details.csv         # Intermediate CSV file containing scraped song details.
song_urls_and_categories.csv     # CSV file mapping song URLs to their categories.
scrapLyrics.py                   # Script to scrape song lyrics from URLs.
song_lyrics_and_details.csv      # Final CSV file containing lyrics and song details.
```

---

## Workflow

### 1. Data Collection
- Use the scripts in the `trainingSetDownload/` directory to scrape data:
  1. Run `scrapLinkAndCategories.py` to scrape song URLs and their categories.
  2. Run `scrapLyrics.py` to scrape the lyrics for each song.
  3. Combine the data into `song_lyrics_and_details.csv`.

### 2. Preprocessing
- Use `saveDatasetLabel.py` to:
  - Map genre names to label IDs and save them in `label_to_id.json`.
  - Map label IDs back to genre names and save them in `id_to_label.json`.

### 3. Training
- Run `genreDetection.py` to:
  - Load the dataset from `data/train.csv` and `data/test.csv`.
  - Preprocess the data and train the BanglaBERT model.
  - Save the trained model and evaluation results in the `results/` directory.
  - Logs are saved in the `logs/` directory.

### 4. Testing
- Use `test.py` to:
  - Load the trained model from `results/`.
  - Evaluate the model on test data or new inputs.
  - Output performance metrics and predictions.

## Current Limitation

- The model couldn't be trained enough on CPU. No GPU is used. Hardware power is low on which the model is trained.
- To make the performance better, the model must be trained in GPU with parameters tested.
- Size of the training dataset comparatively small.

## Future Improvements
- Enhance web scraping for better data quality.
- Fine-tune BanglaBERT for more accurate genre classification.
- Expand the dataset to include more genres and songs.
- Optimize training parameters for better model performance.

---

## Logs and Error Tracking
- Training and evaluation logs are stored in the `logs/` directory.
- Errors encountered during execution are logged in `error_log.txt`.

---


For any issues or queries, please refer to the `README.md` or contact the contributors.

