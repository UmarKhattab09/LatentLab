# encoder.py
from transformers import CLIPProcessor, CLIPModel
import torch
import hashlib
import io
import numpy as np


class LatentEncoder:
    def __init__(self, dim=512):
        """Load CLIP model safely. If the model weights are left on the 'meta'
        device (common with some sharded or incomplete checkpoints), attempt to
        reload forcing CPU placement. If that fails, fall back to a deterministic
        dummy encoder so the app remains usable.

        The dummy encoder returns consistent 512-D vectors (normalized) derived
        from a hash of the input so results are reproducible across calls.
        """
        self.dim = dim
        model_path = r"E:/models/clip/"
        self._dummy = False
        self.model = None
        self.processor = None

        try:
            # Try to load model and move to CPU
            self.model = CLIPModel.from_pretrained(model_path)
            self.model.to(torch.device('cpu'))
            self.processor = CLIPProcessor.from_pretrained(model_path)
        except Exception as e:
            # Try a more conservative load
            try:
                self.model = CLIPModel.from_pretrained(model_path, low_cpu_mem_usage=False)
                self.model.to(torch.device('cpu'))
                self.processor = CLIPProcessor.from_pretrained(model_path)
            except Exception as e2:
                print(f"Warning: CLIP model load failed: {e2}")
                self._dummy = True

        # If loaded, check for any 'meta' parameters and attempt a reload with device_map
        if not self._dummy and self.model is not None:
            try:
                any_meta = any(getattr(p, 'device', torch.device('cpu')).type == 'meta' for p in self.model.parameters())
            except Exception:
                any_meta = False

            if any_meta:
                try:
                    # device_map requires accelerate; try to use it if available
                    self.model = CLIPModel.from_pretrained(model_path, device_map='cpu')
                    self.processor = CLIPProcessor.from_pretrained(model_path)
                except Exception:
                    try:
                        self.model.to(torch.device('cpu'))
                    except Exception:
                        # Give up and use dummy encoder
                        print("Warning: model parameters on 'meta' and reload failed; using dummy encoder")
                        self._dummy = True

        if not self._dummy and self.model is not None:
            self.model.eval()
        else:
            # Prepare deterministic fallback
            self._dummy = True

    def _deterministic_vector(self, data_bytes: bytes):
        """Create a deterministic float vector from arbitrary bytes.

        Uses SHA-256 to create a seed and a NumPy RandomState to generate a
        reproducible vector of size `self.dim`.
        """
        h = hashlib.sha256(data_bytes).hexdigest()
        # Use part of hash as integer seed
        seed = int(h[:16], 16) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float32)
        # normalize
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    @torch.no_grad()
    def encode_image(self, pil_image):
        # If model failed to load properly, return deterministic dummy vector
        if self._dummy:
            # convert PIL image to bytes
            try:
                bio = io.BytesIO()
                pil_image.save(bio, format='PNG')
                b = bio.getvalue()
            except Exception:
                b = b'img'
            return self._deterministic_vector(b)

        # Otherwise run through the real model, with robust device handling
        try:
            inputs = self.processor(images=pil_image, return_tensors="pt")
            try:
                model_device = next(self.model.parameters()).device
                if model_device.type == 'meta':
                    model_device = torch.device('cpu')
            except StopIteration:
                model_device = torch.device('cpu')

            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            embedding = self.model.get_image_features(**inputs)
            return embedding[0].cpu().numpy()
        except Exception as e:
            print(f"Encoder runtime error (image): {e}; falling back to dummy vector")
            try:
                bio = io.BytesIO()
                pil_image.save(bio, format='PNG')
                b = bio.getvalue()
            except Exception:
                b = b'img'
            return self._deterministic_vector(b)

    @torch.no_grad()
    def encode_text(self, text):
        if self._dummy:
            return self._deterministic_vector(text.encode('utf-8'))

        try:
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            try:
                model_device = next(self.model.parameters()).device
                if model_device.type == 'meta':
                    model_device = torch.device('cpu')
            except StopIteration:
                model_device = torch.device('cpu')

            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            embedding = self.model.get_text_features(**inputs)
            return embedding[0].cpu().numpy()
        except Exception as e:
            print(f"Encoder runtime error (text): {e}; falling back to dummy vector")
            return self._deterministic_vector(text.encode('utf-8'))