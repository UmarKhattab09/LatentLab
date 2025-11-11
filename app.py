# app.py
import streamlit as st
from encoder import LatentEncoder
from vector_db import VectorDB
from interpolate import interpolate
from PIL import Image
import os
import time
from pathlib import Path
import plotly.express as px
import umap
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
st.title("Latent Lab: Explore AI Embeddings")

# Persist encoder and database instances in Streamlit session state so they survive
# script reruns (Streamlit reruns the script on every interaction). This prevents
# the DB from being recreated empty when the user switches tabs or encodes text.
if 'encoder' not in st.session_state:
    st.session_state['encoder'] = LatentEncoder()
encoder = st.session_state['encoder']

if 'db' not in st.session_state:
    st.session_state['db'] = VectorDB()
db = st.session_state['db']

if 'db_built' not in st.session_state:
    # Add your new filenames here
    samples = [
        "car.jpg",
        "cat.jpg",
        "beach.jpg",
        "dog.jpg",
        "forest.jpg",
        "mountain.jpg",
        "city.jpg",
        "bird.jpg",
    ]

    labels = []
    vectors = []
    for s in samples:
        try:
            img_path = f"E:/LatentLab/assets/{s}"
            
            img = Image.open(img_path)
            
        
            vec = encoder.encode_image(img)
            
            vectors.append(vec)
            labels.append(s)
            print(f"Loaded {s} successfully.")
        except FileNotFoundError:
            print(f"File not found: {img_path}. Check path and extension.")
        except Exception as e:
            print(f"Error loading {s}: {str(e)}")
    if vectors:
        db.add(np.array(vectors), labels)
        print(f"Added {len(vectors)} items to DB.")
    else:
        print("No items added to DB. Searches won't work until fixed.")
    st.session_state.db_built = True

# Track labels we've explicitly added from uploads/text so we don't duplicate on reruns
if 'added_labels' not in st.session_state:
    st.session_state['added_labels'] = []
if 'text_map' not in st.session_state:
    # Map text-label -> full text for display in search results
    st.session_state['text_map'] = {}

tab1, tab2, tab3,tab4 = st.tabs(["Upload & Encode", "Search", "Interpolate",'Decode and Generate'])

with tab1:
    st.write(f"Database loaded with {db.index.ntotal} items and {len(db.metadata)} labels.")

    uploaded = st.file_uploader("Upload Image", type=["png", "jpg"])
    text_input = st.text_input("Or type a description")

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=200)
        # Save upload to assets only if not already saved in this session
        assets_dir = Path("assets")
        assets_dir.mkdir(parents=True, exist_ok=True)
        orig_name = Path(uploaded.name).name if hasattr(uploaded, 'name') else 'unnamed.png'
        safe_name = orig_name.replace(' ', '_')
        # Use a hash of the image bytes to avoid duplicate saves
        import hashlib
        img_bytes = img.tobytes()
        img_hash = hashlib.sha256(img_bytes).hexdigest()[:16]
        filename = f"uploaded_{img_hash}_{safe_name}"
        save_path = assets_dir / filename
        # Only save if file does not already exist
        if not save_path.exists():
            try:
                img.save(save_path)
            except Exception:
                try:
                    uploaded.seek(0)
                    with open(save_path, 'wb') as f:
                        f.write(uploaded.read())
                except Exception:
                    pass

        z = encoder.encode_image(img)
        st.write("Latent Vector (512D):", z[:5], "...")
        st.session_state.z = z
        # Use the saved filename as the DB label so we can display the image later
        label = filename
        if label not in st.session_state['added_labels']:
            db.add(np.array([z]), [label])
            st.session_state['added_labels'].append(label)
            st.success(f"Added {label} to DB")
            # Ensure 3D view is enabled so the user sees their item
            st.session_state['show_3d'] = True

    if text_input:
        z = encoder.encode_text(text_input)
        st.write("Text → Latent Vector")
        st.write(z[:5])
        st.session_state.z = z
        # Add text embedding to DB once (use truncated text as label)
        short = text_input[:40]
        label = f"text:{short}"
        if label not in st.session_state['added_labels']:
            db.add(np.array([z]), [label])
            st.session_state['added_labels'].append(label)
            # Save full text so we can display it on search results
            st.session_state['text_map'][label] = text_input
            st.success("Added text query to DB")
            st.session_state['show_3d'] = True
    
    # 3D visualization of database embeddings
    # if st.checkbox("Show 3D DB embedding (UMAP)", key="show_3d"):
    if db.index.ntotal == 0:
        st.warning("No items in DB to visualize.")
    else:
        vectors_all, labels_all = db.get_all()
        if vectors_all is None or len(labels_all) == 0:
            st.warning("Database has no stored vectors or labels to visualize.")
        else:
            viz_method = st.selectbox("Reduction Method", ["UMAP", "t-SNE", "PCA"])
            try:
                if viz_method == "t-SNE":
                    reducer3 = TSNE(n_components=3, random_state=42, perplexity=min(30, len(vectors_all)-1))

                elif viz_method == "PCA":
                    reducer3 = PCA(n_components=3)
                else:
                    reducer3 = umap.UMAP(n_components=3, random_state=42)
                coords3 = reducer3.fit_transform(vectors_all)
            except Exception as e:
                st.error(f"Failed to run UMAP (3D): {e}")
                coords3 = None

            if coords3 is not None:
                fig3 = px.scatter_3d(
                    x=coords3[:, 0],
                    y=coords3[:, 1],
                    z=coords3[:, 2],
                    color=labels_all,
                    hover_name=labels_all,
                    title="3D UMAP of DB",
                )
                st.plotly_chart(fig3, use_container_width=True)
        

with tab2:
    if st.button("Search Similar"):
        if 'z' in st.session_state and st.session_state['z'] is not None:
            results = None
            try:
                results = db.search(st.session_state['z'], k=3)
            except Exception as e:
                # Log the exception and surface a user-friendly message in Streamlit
                print(e)
                st.error(f"Search failed: {e}")
                results = None
            # Only proceed if results is a non-empty iterable
            if results:
                cols = st.columns(3)
                for (path, score), col in zip(results, cols):
                    try:
                        if isinstance(path, str) and path.startswith("text:"):
                            # Display text query (lookup full text if available)
                            full = st.session_state.get('text_map', {}).get(path, path[len('text:'):])
                            col.write(f"Text: {full}")
                        else:
                            img_path = Path("assets") / path
                            if img_path.exists():
                                col.image(str(img_path), width=150)
                            else:
                                # Not an image file we recognize - just show label
                                col.write(path)
                        col.write(f"{score:.3f}")
                    except Exception as e:
                        col.write(path)
                        col.write(f"{score:.3f}")
            else:
                st.warning("No results found. Try adding more samples or check your query.")
        else:
            st.error("Please upload an image or enter text in the 'Upload & Encode' tab first.")
with tab3:
    img1 = st.file_uploader("Image 1", key="i1")
    img2 = st.file_uploader("Image 2", key="i2")
    interp_method = st.selectbox("Interpolation Method", ["Linear (Original)", "PCA-Reduced"])
    if img1 and img2:
        z1 = encoder.encode_image(Image.open(img1))
        z2 = encoder.encode_image(Image.open(img2))
        
        if interp_method == "PCA-Reduced":
            # Reduce to e.g., 50D with PCA, then interpolate in reduced space
            pca = PCA(n_components=50)
            all_z = np.vstack([z1, z2])
            pca.fit(all_z)  # Fit on the two points (or use full DB for better)
            z1_pca = pca.transform(z1.reshape(1, -1))[0]
            z2_pca = pca.transform(z2.reshape(1, -1))[0]
            interp = interpolate(z1_pca, z2_pca, 5)  # Interp in PCA space
            # Optional: Inverse transform back to original space if needed
            interp = [pca.inverse_transform(np.array([zi]))[0] for zi in interp]
        else:
            interp = interpolate(z1, z2, 5)  # Original linear
        
        st.write("Interpolation in latent space (visualize with UMAP)")
        all_z = np.vstack([z1, z2] + interp)
        reducer = umap.UMAP()
        proj = reducer.fit_transform(all_z)
        fig = px.scatter(x=proj[:,0], y=proj[:,1], title="Latent Path")
        st.plotly_chart(fig)


with tab4:
    if 'z' in st.session_state and st.session_state['z'] is not None:
        st.write("Using current encoded latent (from Tab 1) for decoding/generation.")
        if st.button("Decode/Reconstruct Image (VAE)"):
            try:
                # Load pre-trained VAE (from Stable Diffusion)
                from diffusers import AutoencoderKL  # For VAE decoder
                vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
                vae.eval()

                # Re-encode the original image to VAE latent (CLIP isn't decodable, so use original img if available)
                if uploaded:  # Assumes you have the uploaded image from Tab 1
                    img = Image.open(uploaded).convert("RGB")
                    # Preprocess for VAE (resize to 256x256, normalize)
                    img_tensor = torch.tensor(np.array(img.resize((256, 256))) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
                    with torch.no_grad():
                        vae_latent = vae.encode(img_tensor).latent_dist.sample() * 0.18215  # Standard SD scaling
                        decoded = vae.decode(vae_latent / 0.18215).sample[0]
                    # Postprocess decoded tensor to image
                    decoded_img = Image.fromarray(((decoded.permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8))
                    st.image(decoded_img, caption="Decoded/Reconstructed Image", width=300)
            except Exception as e:
                st.error(f"Decoding failed: {e}. Note: Text latents can't be decoded; use image input.")
    else:
        st.info("Encode an image in Tab 1 first to decode.")