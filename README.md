# Latent Lab

Latent Lab is an interactive Streamlit app for exploring, visualizing, and searching AI-generated image and text embeddings. It uses a CLIP-based encoder to embed images and text, stores them in a vector database (FAISS), and provides 2D/3D visualizations (UMAP, t-SNE, PCA) and similarity search. Moreover, you can understand (Encoder/Decoders) and how they work to encode and decode the image to learn patterns.

## Features
- Upload images or enter text to encode into a shared latent space
- Vector database with similarity search (FAISS)
- Interactive 2D and 3D visualizations of embeddings (UMAP, t-SNE, PCA)
- See where your uploads/queries lie in the embedding space
- Supports both image and text queries


## Setup

### 1. Clone the repository
```sh
git clone <your-repo-url>
cd LatentLab
```

### 2. Install dependencies
Create a virtual environment (recommended) and install requirements:
```sh
python -m venv .venv
.venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 3. Download CLIP model weights
- Download the OpenAI CLIP model (ViT-B/32 or similar) from Hugging Face or OpenAI.
- Place the model files in `E:/models/clip/` (or update the path in `encoder.py`).
- The folder should contain `pytorch_model.bin`, `config.json`, `tokenizer.json`, etc.

### 4. Prepare sample images
- Place sample images (e.g., `car.jpg`, `cat.jpg`, etc.) in the `assets/` folder.

### 5. Run the app
```sh
streamlit run app.py
```

## Usage
- **Upload & Encode tab:** Upload an image or enter text to encode. New items are added to the vector DB and visualized in 3D.
- **Search tab:** Find similar images/text in the DB to your current query.
- **Interpolate tab:** Interpolate between two images in latent space and visualize the path.
- **3D Visualization:** See all items in the DB projected into 3D (UMAP). Uploaded images are saved only once and labeled for easy retrieval.

## Troubleshooting
- If you see errors about 'meta' device or model loading, check your model files in `E:/models/clip/`.
- If the model can't be loaded, the app will use a dummy encoder so you can still test the UI.
- Uploaded images are deduplicated by content hash to avoid multiple saves.

## File Structure
- `app.py` — Main Streamlit app
- `encoder.py` — CLIP encoder and dummy fallback
- `vector_db.py` — Vector database using FAISS
- `interpolate.py` — Latent space interpolation
- `assets/` — Uploaded and sample images
- `requirements.txt` — Python dependencies

## Requirements
- Python 3.8+
- streamlit
- numpy
- faiss-cpu
- pillow
- plotly
- umap-learn
- scikit-learn
- transformers

## License
MIT License
