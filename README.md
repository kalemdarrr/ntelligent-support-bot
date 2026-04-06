# Customer Support Intent Detector Chatbot

This is a minimum viable Customer Support chatbot application built for Track 1 of the Intelligent Application Final Project.

## Project Structure
- `data/dataset.csv`: English dataset mapped to predefined intents.
- `src/preprocess.py`: Text cleaning and normalization logic.
- `src/train.py`: Model training pipeline (TF-IDF + Logistic Regression).
- `src/evaluate.py`: Generates evaluation metrics on a 20% test split.
- `src/inference.py`: Handles raw input directly into structured bot responses.
- `app.py`: Streamlit-based graphical user interface.

## Quick Start
1. Ensure Python 3.10+ is installed.
2. Initialize virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the Streamlit User Interface:
   ```bash
   streamlit run app.py
   ```

*Note: The model files are already trained and located under `models/` folder. If you wish to retrain, just run `python src/train.py`.*
