# app/core/local_vision.py (memory-optimized + lazy-loading)
"""
Memory-optimized version of LocalVisionModels.
- Adds channels_last memory format for BLIP/CLIP on CUDA
- Sets use_cache=False for DETR/TrOCR where available
- Supports lazy-loading and unloading of DETR and TrOCR (to avoid OOM)
- Optional CPU-offload for DETR/TrOCR (keep them off GPU unless explicitly loaded)
- Safe fallbacks for half() / channels_last

This file is intended to be a drop-in replacement for the original with the
same public methods (`caption_image`, `get_clip_*`, `detect_objects`, `run_ocr`, `analyze_image`, ...)
but it will keep large models unloaded until they are needed.
"""
from __future__ import annotations
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import numpy as np
from PIL import Image, UnidentifiedImageError

import torch

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
    AutoImageProcessor,
    DetrForObjectDetection,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HAS_HEIF = True
except ImportError:
    HAS_HEIF = False


logger = logging.getLogger(__name__)


@dataclass
class VisionImageAnalysis:
    caption: Optional[str] = None
    tags: Optional[List[str]] = None
    objects: Optional[List[str]] = None
    mood: Optional[str] = None
    contains_text: Optional[bool] = None
    is_sensitive: Optional[bool] = None
    ocr_text: Optional[str] = None
    raw_objects: Optional[List[Dict[str, Any]]] = None
    sharpness_score: Optional[float] = None
    color_palette: Optional[List[List[int]]] = None  # [[r,g,b], ...]
    


class LocalVisionModels:
    """
    Memory-optimized vision stack. Key features:
      - BLIP-large for captioning (kept loaded)
      - CLIP for embeddings + aesthetics (kept loaded)
      - DETR for object detection (lazy-loaded, optional CPU-offload)
      - TrOCR for OCR (lazy-loaded, optional CPU-offload)

    Initialization keeps only BLIP and CLIP resident on the configured device
    (GPU if available). DETR and TrOCR are loaded only when first requested
    via detect_objects/run_ocr or analyze_image.
    """
    def __init__(
        self,
        device: Optional[str] = None,
        lazy_load_detection: bool = True,
        offload_detection_to_cpu: bool = True,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not torch.cuda.is_available():
            logger.warning("[Vision] CUDA not available. Running ALL models on CPU.")
            self.device = "cpu"
            self.offload_detection_to_cpu = True
            self.lazy_load_detection = False   # Load models normally


        self.lazy_load_detection = lazy_load_detection
        # If True, keep DETR/TrOCR on CPU until explicitly loaded to GPU
        self.offload_detection_to_cpu = offload_detection_to_cpu

        # idle-unload timers
        self._last_detr_use: float = 0.0
        self._last_ocr_use: float = 0.0
        self._idle_unload_seconds: float = 60.0  # tweak as needed

        logger.info(
            f"[Vision] Initializing LocalVisionModels device={self.device} "
            f"lazy_load_detection={self.lazy_load_detection} "
            f"offload_detection_to_cpu={self.offload_detection_to_cpu}"
        )

        # ---- BLIP (captioning) ----
        logger.info("[Vision] Loading BLIP-large captioning model (resident)...")
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        self._move_model_to_device(self.blip_model, self.device, try_half=True, name="BLIP")
        self.blip_model.eval()

        # ---- CLIP (embeddings + aesthetics) ----
        logger.info("[Vision] Loading CLIP model (resident)...")
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self._move_model_to_device(self.clip_model, self.device, try_half=True, name="CLIP")
        self.clip_model.eval()

        # DETR / TrOCR placeholders (lazy)
        self.detr_processor = None
        self.detr_model = None
        self.ocr_processor = None
        self.ocr_model = None

        self._detr_on_gpu = False
        self._ocr_on_gpu = False

        if self.device == "cuda":
            try:
                self._set_channels_last(self.blip_model, name="BLIP")
                self._set_channels_last(self.clip_model, name="CLIP")
            except Exception:
                pass
        
        self.debug_aggressive_unload = False


        logger.info("[Vision] LocalVisionModels (memory-optimized) initialized.")

    # -------------------------
    # Model utility helpers
    # -------------------------
    def _move_model_to_device(self, model, device: str, try_half: bool = False, name: str = ""):
        """Move model to device safely. Optionally convert to half precision.
        """
        try:
            model.to(device)
            if try_half and device == "cuda":
                try:
                    model.half()
                except Exception as e:
                    logger.debug(f"[Vision] {name} half() failed: {e}")
        except Exception as e:
            logger.warning(f"[Vision] Failed to move {name} to {device}: {e}")

    def _set_channels_last(self, model, name: str = ""):
        """Attempt to set model parameters/tensors to channels_last memory format.
        This is a best-effort optimization for CUDA.
        """
        try:
            # Not all models respond to this; surround in try/except.
            model.to(memory_format=torch.channels_last)
            logger.debug(f"[Vision] Set {name} to channels_last (best-effort)")
        except Exception as e:
            logger.debug(f"[Vision] channels_last for {name} failed: {e}")

    def _set_use_cache_false(self, model, name: str = ""):
        """Try to disable caching to lower peak memory (best-effort)."""
        try:
            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = False
                logger.debug(f"[Vision] set use_cache=False for {name}")
        except Exception:
            pass

    # -------------------------
    # Lazy load / unload for DETR and TrOCR
    # -------------------------
    def load_detr(self, to_gpu: Optional[bool] = None):
        """Load DETR model and processor. If offload_detection_to_cpu is True, keep on CPU by default.
        Pass to_gpu=True to force move to GPU (if available).
        """
        if self.detr_model is not None:
            # already loaded
            if to_gpu and not self._detr_on_gpu:
                self._move_model_to_device(self.detr_model, self.device, try_half=True, name="DETR")
                self._detr_on_gpu = self.device == "cuda"
            return

        logger.info("[Vision] Lazy-loading DETR model...")
        self.detr_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        # disable caching / attention-slicing best-effort
        self._set_use_cache_false(self.detr_model, name="DETR")

        # decide where to place it
        if to_gpu is None:
            to_gpu = not self.offload_detection_to_cpu and self.device == "cuda"

        if to_gpu and self.device == "cuda":
            self._move_model_to_device(self.detr_model, self.device, try_half=True, name="DETR")
            self._detr_on_gpu = True
        else:
            # keep on CPU to save GPU VRAM
            self._move_model_to_device(self.detr_model, "cpu", try_half=False, name="DETR")
            self._detr_on_gpu = False

        self.detr_model.eval()
        logger.info(f"[Vision] DETR loaded (on_gpu={self._detr_on_gpu})")

    def unload_detr(self):
        """Unload DETR model from memory (frees GPU & CPU memory)."""
        if self.detr_model is None:
            return
        try:
            # Move to CPU and delete
            try:
                self.detr_model.to("cpu")
            except Exception:
                pass
            del self.detr_model
            self.detr_model = None
            self.detr_processor = None
            self._detr_on_gpu = False
            torch.cuda.empty_cache()
            logger.info("[Vision] DETR unloaded and cache cleared.")
        except Exception as e:
            logger.warning(f"[Vision] Error unloading DETR: {e}")

    def load_ocr(self, to_gpu: Optional[bool] = None):
        """Load TrOCR processor and model. Similar offload semantics to DETR."""
        if self.ocr_model is not None:
            if to_gpu and not self._ocr_on_gpu:
                self._move_model_to_device(self.ocr_model, self.device, try_half=True, name="TrOCR")
                self._ocr_on_gpu = self.device == "cuda"
            return

        logger.info("[Vision] Lazy-loading TrOCR model...")
        self.ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

        # disable cache
        self._set_use_cache_false(self.ocr_model, name="TrOCR")

        if to_gpu is None:
            to_gpu = not self.offload_detection_to_cpu and self.device == "cuda"

        if to_gpu and self.device == "cuda":
            self._move_model_to_device(self.ocr_model, self.device, try_half=True, name="TrOCR")
            self._ocr_on_gpu = True
        else:
            self._move_model_to_device(self.ocr_model, "cpu", try_half=False, name="TrOCR")
            self._ocr_on_gpu = False

        self.ocr_model.eval()
        logger.info(f"[Vision] TrOCR loaded (on_gpu={self._ocr_on_gpu})")

    def unload_ocr(self):
        if self.ocr_model is None:
            return
        try:
            try:
                self.ocr_model.to("cpu")
            except Exception:
                pass
            del self.ocr_model
            self.ocr_model = None
            self.ocr_processor = None
            self._ocr_on_gpu = False
            torch.cuda.empty_cache()
            logger.info("[Vision] TrOCR unloaded and cache cleared.")
        except Exception as e:
            logger.warning(f"[Vision] Error unloading TrOCR: {e}")

    # -------------------------
    # Core helpers
    # -------------------------
    def _load_image(self, path: str | Path) -> Image.Image:
        """
        Robust image loader with HEIC support.

        Strategy:
        1. Try PIL.Image.open directly.
        2. If that fails and file is HEIC/HEIF → use pillow-heif to decode.
        3. If still failing → convert to JPEG in-memory and load.
        Always returns an RGB PIL.Image or raises.
        """
        path = Path(path)
        ext = path.suffix.lower()

        # 1) Try direct PIL open
        try:
            img = Image.open(path)
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except (UnidentifiedImageError, OSError) as e:
            logger.warning("PIL failed to open %s directly: %s", path, e)

        # 2) If HEIC and pillow-heif is available, try direct HEIF decode
        if ext in {".heic", ".heif"} and HAS_HEIF:
            try:
                heif_file = pillow_heif.read_heif(str(path))
                img = Image.frombytes(
                    heif_file.mode,
                    heif_file.size,
                    heif_file.data,
                    "raw",
                    heif_file.mode,
                    0,
                    1,
                )
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            except Exception as e:
                logger.warning("pillow-heif failed to decode %s directly: %s", path, e)

        # 3) Last fallback: convert to JPEG in-memory (Option C final fallback)
        if ext in {".heic", ".heif"} and HAS_HEIF:
            try:
                heif_file = pillow_heif.read_heif(str(path))
                img = Image.frombytes(
                    heif_file.mode,
                    heif_file.size,
                    heif_file.data,
                    "raw",
                    heif_file.mode,
                    0,
                    1,
                )
                with io.BytesIO() as buf:
                    img.save(buf, format="JPEG", quality=95)
                    buf.seek(0)
                    jpeg_img = Image.open(buf)
                    jpeg_img.load()
                    if jpeg_img.mode != "RGB":
                        jpeg_img = jpeg_img.convert("RGB")
                    logger.info("Fallback: converted HEIC to in-memory JPEG for %s", path)
                    return jpeg_img
            except Exception as e:
                logger.error("Final HEIC→JPEG fallback failed for %s: %s", path, e)

        # Non-HEIC or everything failed
        raise RuntimeError(f"Cannot load image: {path}")

    # -------------------------
    # Captioning (BLIP)
    # -------------------------
    def caption_image(self, image_path: Path, max_length: int = 64) -> str:
        try:
            img = self._load_image(image_path)
            inputs = self.blip_processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_ids = self.blip_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True,
                )
            caption = self.blip_processor.decode(output_ids[0], skip_special_tokens=True)
            caption = caption.strip()
            logger.debug(f"[Vision] Caption for {image_path}: {caption}")
            return caption
        except Exception as e:
            logger.error(f"[Vision] Error captioning {image_path}: {e}")
            return ""

    # -------------------------
    # CLIP embeddings & aesthetics
    # -------------------------
    def get_clip_text_embedding(self, text: str) -> np.ndarray:
        try:
            tokens = self.clip_processor(text=[text], return_tensors="pt").to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**tokens)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features[0].detach().cpu().float().numpy()
        except Exception as e:
            logger.error(f"[Vision] Error computing CLIP text embedding: {e}")
            return np.zeros((768,), dtype=np.float32)

    def get_clip_image_embedding(self, image_path: Path) -> np.ndarray:
        try:
            img = self._load_image(image_path)
            inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features[0].detach().cpu().float().numpy()
        except Exception as e:
            logger.error(f"[Vision] Error computing CLIP embedding: {e}")
            return np.zeros((768,), dtype=np.float32)

    def estimate_aesthetic_score(self, image_path: Path) -> float:
        try:
            img = self._load_image(image_path)
            prompts = ["a beautiful professional photo", "a low-quality bad photo"]
            inputs = self.clip_processor(
                text=prompts, images=img, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                image_emb = outputs.image_embeds
                text_emb = outputs.text_embeds

            image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)

            logits = (image_emb @ text_emb.t())[0]
            probs = torch.softmax(logits, dim=-1)
            beautiful_prob = float(probs[0].item())
            return float(max(0.0, min(1.0, beautiful_prob)))
        except Exception as e:
            logger.error(f"[Vision] Error estimating aesthetic score: {e}")
            return 0.5

    # -------------------------
    # Object detection (DETR) - lazy loaded
    # -------------------------
    def detect_objects(
        self,
        image_path: Path,
        score_threshold: float = 0.7,
        max_objects: int = 10,
        force_gpu: bool = False,
    ) -> List[Dict[str, Any]]:
        # ensure DETR is loaded
        if self.lazy_load_detection and self.detr_model is None:
            # load to GPU if user asks or offload disabled
            self.load_detr(to_gpu=(force_gpu or (not self.offload_detection_to_cpu and self.device == "cuda")))

        if self.detr_model is None:
            # nothing to do
            logger.debug("[Vision] DETR not available (lazy-loaded and not loaded). Returning empty list.")
            return []

        try:
            img = self._load_image(image_path)
            # processor expects PIL
            inputs = self.detr_processor(images=img, return_tensors="pt").to(
                "cuda" if self._detr_on_gpu else "cpu"
            )
            with torch.no_grad():
                outputs = self.detr_model(**inputs)
            
            self._last_detr_use = time.monotonic()

            target_sizes = torch.tensor([img.size[::-1]])
            results = self.detr_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes
            )[0]

            detections: List[Dict[str, Any]] = []
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                score_val = float(score.item())
                if score_val < score_threshold:
                    continue
                if len(detections) >= max_objects:
                    break

                box = box.detach().cpu().tolist()
                label_str = self.detr_model.config.id2label[int(label.item())]

                detections.append(
                    {"label": label_str, "score": score_val, "bbox": [float(b) for b in box]}
                )

            return detections
        except Exception as e:
            logger.error(f"[Vision] Error detecting objects in {image_path}: {e}")
            return []

    # -------------------------
    # OCR (TrOCR) - lazy loaded
    # -------------------------
    def run_ocr(self, image_path: Path, max_length: int = 128, force_gpu: bool = False) -> str:
        if self.lazy_load_detection and self.ocr_model is None:
            self.load_ocr(to_gpu=(force_gpu or (not self.offload_detection_to_cpu and self.device == "cuda")))

        if self.ocr_model is None:
            logger.debug("[Vision] TrOCR not available (lazy-loaded and not loaded). Returning empty string.")
            return ""

        try:
            img = self._load_image(image_path)
            device = "cuda" if self._ocr_on_gpu else "cpu"
            pixel_values = self.ocr_processor(images=img, return_tensors="pt").pixel_values.to(device)

            with torch.no_grad():
                generated_ids = self.ocr_model.generate(
                    pixel_values,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                )
            self._last_ocr_use = time.monotonic() 
            text = self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()
        except Exception as e:
            logger.error(f"[Vision] Error running OCR on {image_path}: {e}")
            return ""

    def has_text(self, image_path: Path, min_chars: int = 3) -> bool:
        text = self.run_ocr(image_path)
        return len(text.strip()) >= min_chars

    # -------------------------
    # High-level combined analysis
    # -------------------------
    def analyze_image(self, image_path: Path, max_tags: int = 15) -> VisionImageAnalysis:
        image_path = Path(image_path)

        caption = self.caption_image(image_path)
        tags = self._tags_from_caption(caption, max_tags=max_tags)

        # Object detection (lazy-load). We don't force GPU here; let user override with force_gpu flag in detect_objects
        raw_objects = self.detect_objects(image_path, score_threshold=0.7)
        object_labels = sorted({det["label"] for det in raw_objects}) if raw_objects else []

        # OCR / text presence
        ocr_text = self.run_ocr(image_path)
        contains_text = len(ocr_text.strip()) > 3

        mood = self._infer_mood_from_caption(caption)
        is_sensitive = self._is_potentially_sensitive(caption, object_labels)

        # sharpness + palette
        sharpness_score = self.estimate_sharpness(image_path)
        color_palette = self.estimate_color_palette(image_path)

        return VisionImageAnalysis(
            caption=caption or None,
            tags=tags or None,
            objects=object_labels or None,
            mood=mood,
            contains_text=contains_text,
            is_sensitive=is_sensitive,
            ocr_text=ocr_text or None,
            raw_objects=raw_objects or None,
            sharpness_score=sharpness_score,
            color_palette=color_palette or None,
        )

    # -------------------------
    # Small NLP helpers (same as original)
    # -------------------------
    def _tags_from_caption(self, caption: str, max_tags: int = 15) -> List[str]:
        if not caption:
            return []
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "with",
            "of",
            "in",
            "on",
            "at",
            "for",
            "to",
            "is",
            "are",
            "this",
            "that",
            "there",
            "it",
            "from",
            "by",
            "near",
            "view",
            "photo",
            "picture",
            "image",
        }
        words = [w.strip(".,!?;:()[]{}\"' ").lower() for w in caption.split()]
        keywords = [w for w in words if len(w) > 2 and w not in stopwords and not w.isdigit()]
        seen = set()
        unique = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return unique[:max_tags]

    def _infer_mood_from_caption(self, caption: str) -> Optional[str]:
        if not caption:
            return None
        text = caption.lower()
        positive = ["happy", "smiling", "friends", "celebration", "sunny", "beautiful"]
        negative = ["sad", "alone", "rainy", "dark", "angry", "upset"]
        calm = ["calm", "peaceful", "serene", "quiet", "relaxed"]
        score_pos = sum(word in text for word in positive)
        score_neg = sum(word in text for word in negative)
        score_calm = sum(word in text for word in calm)
        if max(score_pos, score_neg, score_calm) == 0:
            return None
        if score_pos >= score_neg and score_pos >= score_calm:
            return "positive"
        if score_neg >= score_pos and score_neg >= score_calm:
            return "negative"
        return "calm"

    def _is_potentially_sensitive(self, caption: str, object_labels: List[str]) -> bool:
        sensitive_words = {
            "blood",
            "weapon",
            "gun",
            "knife",
            "nude",
            "naked",
            "violence",
            "fight",
            "injury",
            "wound",
        }
        text = (caption or "").lower()
        if any(w in text for w in sensitive_words):
            return True
        if any(lbl.lower() in {"weapon", "knife"} for lbl in object_labels or []):
            return True
        return False
    
    def estimate_sharpness(self, image_path: Path) -> float:
        """
        Estimate sharpness using Laplacian variance.
        Runs on GPU if available, falls back to CPU.
        Returns a float where higher ~ sharper.
        """
        try:
            img = self._load_image(image_path)
            arr = np.array(img, dtype=np.float32) / 255.0  # H, W, 3
            # convert to grayscale
            gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            gray_t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)  # 1,1,H,W

            device = self.device if torch.cuda.is_available() else "cpu"
            gray_t = gray_t.to(device)

            # 3x3 Laplacian kernel
            kernel = torch.tensor(
                [[0.0,  1.0, 0.0],
                 [1.0, -4.0, 1.0],
                 [0.0,  1.0, 0.0]],
                dtype=torch.float32,
                device=device,
            ).view(1, 1, 3, 3)

            with torch.no_grad():
                edges = torch.nn.functional.conv2d(gray_t, kernel, padding=1)
                # variance of response as blur proxy
                var = edges.pow(2).mean()

            score = float(var.item())

            # squash to [0, 1] for nicer scale (tuned heuristic)
            # assuming var usually in [0, ~0.02]
            norm_score = max(0.0, min(1.0, score / 0.02))
            return norm_score
        except Exception as e:
            logger.error(f"[Vision] Error estimating sharpness for {image_path}: {e}")
            return 0.5

    def estimate_color_palette(
        self,
        image_path: Path,
        k: int = 5,
        max_pixels: int = 50_000,
        iterations: int = 10,
    ) -> List[List[int]]:
        """
        Estimate dominant colors using simple k-means on GPU if available.
        Returns list of RGB triplets as ints [0-255].
        """
        try:
            img = self._load_image(image_path)
            arr = np.array(img, dtype=np.float32) / 255.0  # H, W, 3
            h, w, c = arr.shape
            flat = arr.reshape(-1, 3)  # N, 3

            # subsample if very large
            if flat.shape[0] > max_pixels:
                idx = np.random.choice(flat.shape[0], max_pixels, replace=False)
                flat = flat[idx]

            device = self.device if torch.cuda.is_available() else "cpu"
            x = torch.from_numpy(flat).to(device)  # N, 3

            # init centers by random sample
            if x.shape[0] < k:
                k = max(1, x.shape[0])
            idx0 = torch.randperm(x.shape[0], device=device)[:k]
            centers = x[idx0].clone()  # k,3

            for _ in range(iterations):
                # assign
                # N,k distances
                dists = (x.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(dim=2)
                labels = dists.argmin(dim=1)  # N

                # recompute centers
                new_centers = []
                for ki in range(k):
                    mask = labels == ki
                    if mask.any():
                        new_centers.append(x[mask].mean(dim=0))
                    else:
                        # re-sample if empty cluster
                        new_centers.append(
                            x[torch.randint(0, x.shape[0], (1,), device=device)][0]
                        )
                centers = torch.stack(new_centers, dim=0)

            centers_cpu = centers.detach().cpu().clamp(0.0, 1.0).numpy()
            # convert to 0-255 ints
            palette = (centers_cpu * 255.0).round().astype(np.int32).tolist()
            return palette
        except Exception as e:
            logger.error(f"[Vision] Error estimating color palette for {image_path}: {e}")
            return []
        
    def maybe_unload_idle_models(self):
        """
        Best-effort cleanup: unload DETR / TrOCR if idle too long.
        Call this occasionally from the worker.
        """
        now = time.monotonic()

        if self.debug_aggressive_unload:
            logger.info("[Vision] DEBUG: Unloading DETR & TrOCR aggressively")
            self.unload_detr()
            self.unload_ocr()
            return

        if self.detr_model is not None and self._last_detr_use > 0:
            if now - self._last_detr_use > self._idle_unload_seconds:
                logger.info("[Vision] DETR idle too long, unloading...")
                self.unload_detr()

        if self.ocr_model is not None and self._last_ocr_use > 0:
            if now - self._last_ocr_use > self._idle_unload_seconds:
                logger.info("[Vision] TrOCR idle too long, unloading...")
                self.unload_ocr()

    def unload_temp_models(self):
        """
        Aggressively free memory for heavy models that don't need to stay resident.
        Keep BLIP resident; unload DETR, TrOCR, maybe CLIP vision heads.
        """
        import gc
        import torch

        # Example: if you stored them as self.detr, self.trocr, self.clip_model, etc.
        if hasattr(self, "detr") and self.detr is not None:
            logger.info("[Vision] Unloading DETR")
            try:
                del self.detr
            except Exception:
                pass
            self.detr = None

        if hasattr(self, "trocr") and self.trocr is not None:
            logger.info("[Vision] Unloading TrOCR")
            try:
                del self.trocr
            except Exception:
                pass
            self.trocr = None

        # Optionally partially unload CLIP vision if you keep a big one:
        if hasattr(self, "clip_model") and getattr(self, "offload_clip_vision", False):
            logger.info("[Vision] Offloading CLIP vision backbone")
            try:
                self.clip_model.vision_model = self.clip_model.vision_model.cpu()
            except Exception:
                pass

        torch.cuda.empty_cache()
        gc.collect()

