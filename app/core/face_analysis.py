#core/face_analysis.py

"""
Face analysis using InsightFace with DBSCAN clustering
Windows-compatible version
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from insightface.app import FaceAnalysis
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HAS_HEIF = True
except ImportError:
    HAS_HEIF = False


logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Face detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    embedding: np.ndarray  # 512-dim face embedding
    confidence: float
    landmarks: Optional[np.ndarray] = None  # 5 facial landmarks
    gender: Optional[str] = None
    age: Optional[int] = None
    eye_open: Optional[float] = None  # Eye openness probability


class FaceAnalyzer:
    """Face detection and clustering using InsightFace"""

    def __init__(self, model_name: str = 'buffalo_l'):
        self.model_name = model_name
        self.app = None
        self.genderage_model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize InsightFace model in a Windows-safe way"""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            from insightface.model_zoo import get_model

            # Try to detect available ONNXRuntime providers
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
            except Exception as e:
                logger.warning(
                    "onnxruntime not available or failed to query providers: %s. "
                    "Falling back to CPUExecutionProvider only.",
                    e,
                )
                available_providers = ["CPUExecutionProvider"]

            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                ctx_id = 0  # GPU
            else:
                providers = ["CPUExecutionProvider"]
                ctx_id = -1  # CPU only

            logger.info(f"Using InsightFace providers: {providers}")

            # Initialize face analysis app
            self.app = FaceAnalysis(
                name=self.model_name,
                providers=providers
            )
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

            # Load gender/age model (optional)
            try:
                self.genderage_model = get_model('genderage')
                self.genderage_model.prepare(ctx_id=ctx_id)
            except Exception as e:
                logger.warning(f"Gender/Age model not available: {e}")
                self.genderage_model = None

            logger.info(f"InsightFace model '{self.model_name}' initialized")

        except ImportError as e:
            logger.error(f"InsightFace not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            raise

    def detect_faces(self, image_path: Path) -> List[FaceDetection]:
        """Detect faces in image and extract embeddings (Windows-safe image loading)"""

        # First attempt: normal OpenCV read
        img = cv2.imread(str(image_path))

        # On Windows, non-ASCII or weird paths sometimes break imread
        if img is None:
            try:
                rgb = load_image_any(image_path)
                img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.error(f"Failed to read image (fallback) from {image_path}: {e}")
                return []


        # Run face detection
        faces = self.app.get(img)

        results: List[FaceDetection] = []
        for face in faces:
            # Extract embedding
            embedding = face.normed_embedding

            # Calculate eye openness (simplified - using landmark distances)
            eye_open = self._calculate_eye_openness(
                face.landmark if hasattr(face, "landmark") else None
            )

            # Get gender/age if model available
            gender = None
            age = None
            if self.genderage_model is not None and hasattr(face, 'bbox'):
                x1, y1, x2, y2 = [int(coord) for coord in face.bbox]
                face_crop = img[y1:y2, x1:x2]
                if face_crop.size > 0:
                    try:
                        gender_age = self.genderage_model.get(face_crop)
                        if gender_age:
                            # gender_age[0]: 0=female, 1=male
                            # gender_age[1]: age
                            gender = 'F' if gender_age[0] == 0 else 'M'
                            age = int(gender_age[1])
                    except Exception:
                        # Fail silently, keep gender/age as None
                        pass

            detection = FaceDetection(
                bbox=tuple(int(coord) for coord in face.bbox),
                embedding=embedding,
                confidence=face.det_score,
                landmarks=face.landmark if hasattr(face, 'landmark') else None,
                gender=gender,
                age=age,
                eye_open=eye_open
            )
            results.append(detection)

        return results

    def _calculate_eye_openness(self, landmarks: Optional[np.ndarray]) -> float:
        """Calculate eye openness from landmarks (very simplified)"""
        if landmarks is None or len(landmarks) < 5:
            return 0.5

        try:
            # For InsightFace, landmarks are: left_eye, right_eye, nose, left_mouth, right_mouth
            left_eye_center = landmarks[0]
            right_eye_center = landmarks[1]

            # Simple openness measure based on distance between the eyes
            eye_width = np.linalg.norm(right_eye_center - left_eye_center)
            openness = min(1.0, eye_width / 100.0)  # normalize roughly
            return float(openness)
        except Exception:
            return 0.5

    def cluster_faces(
        self,
        embeddings: List[np.ndarray],
        eps: float = 0.5,
        min_samples: int = 2
    ) -> np.ndarray:
        """
        Cluster face embeddings using DBSCAN
        Returns cluster labels (-1 for noise, but remapped to unique negatives)
        """
        if len(embeddings) < 2:
            return np.array([0] * len(embeddings)) if embeddings else np.array([])

        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        # Apply DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clustering.fit_predict(embeddings_scaled)

        # Convert -1 (noise) to unique negative IDs to avoid confusion
        unique_neg = -1
        for i in range(len(labels)):
            if labels[i] == -1:
                labels[i] = unique_neg
                unique_neg -= 1

        return labels

    def find_similar_faces(
        self,
        query_embedding: np.ndarray,
        embeddings: List[np.ndarray],
        threshold: float = 0.6
    ) -> List[Tuple[int, float]]:
        """Find faces similar to query embedding using cosine similarity"""
        if not embeddings:
            return []

        similarities: List[Tuple[int, float]] = []
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        for i, emb in enumerate(embeddings):
            emb_norm = emb / np.linalg.norm(emb)
            similarity = float(np.dot(query_norm, emb_norm))
            similarities.append((i, similarity))

        # Filter by threshold and sort
        similar = [(i, sim) for i, sim in similarities if sim >= threshold]
        similar.sort(key=lambda x: x[1], reverse=True)

        return similar

    def save_embeddings(self, embeddings: List[np.ndarray], filepath: Path):
        """Save embeddings to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)

    def load_embeddings(self, filepath: Path) -> List[np.ndarray]:
        """Load embeddings from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
        
def load_image_any(path: str | Path) -> np.ndarray:
    """
    Robust loader for faces; returns RGB numpy array.
    HEIC is decoded via pillow-heif if available.
    """
    path = Path(path)
    ext = path.suffix.lower()

    # 1) Try plain PIL
    try:
        img = Image.open(path)
        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)
    except (UnidentifiedImageError, OSError) as e:
        logger.warning("Face loader: PIL failed for %s: %s", path, e)

    # 2) HEIC via pillow-heif
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
            return np.array(img)
        except Exception as e:
            logger.error("Face loader: HEIC decode failed for %s: %s", path, e)

    raise RuntimeError(f"Face loader: cannot read image {path}")


class FaceDatabase:
    """Manage face clusters and identities"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        # cluster_id -> dict
        self.clusters: Dict[int, Dict] = {}
        # person_id -> dict
        self.identities: Dict[int, Dict] = {}
        self.next_person_id: int = 1
        self.next_cluster_id: int = 0

        # Load existing data
        self._load()

    def _load(self):
        """Load face database from disk"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.clusters = data.get('clusters', {})
                    self.identities = data.get('identities', {})
                    self.next_person_id = data.get('next_person_id', 1)
                    self.next_cluster_id = data.get('next_cluster_id', 0)
            except Exception as e:
                logger.error(f"Failed to load face database: {e}")

    def save(self):
        """Save face database to disk"""
        try:
            with open(self.db_path, 'wb') as f:
                data = {
                    'clusters': self.clusters,
                    'identities': self.identities,
                    'next_person_id': self.next_person_id,
                    'next_cluster_id': self.next_cluster_id
                }
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save face database: {e}")

    def add_unknown_cluster(self, embeddings: List[np.ndarray]) -> int:
        """Add new cluster of unknown faces"""
        cluster_id = self.next_cluster_id
        self.clusters[cluster_id] = {
            'embeddings': embeddings,
            'person_id': None,
            'count': len(embeddings),
            'created': np.datetime64('now')
        }
        self.next_cluster_id += 1
        self.save()
        return cluster_id

    def assign_cluster_to_person(self, cluster_id: int, person_name: str) -> int:
        """Assign a cluster to a person (create new or merge with existing)"""
        if cluster_id not in self.clusters:
            raise ValueError(f"Cluster {cluster_id} not found")

        cluster_embeddings = self.clusters[cluster_id]['embeddings']
        person_id = self._find_similar_person(cluster_embeddings)

        if person_id is None:
            # Create new person
            person_id = self.next_person_id
            self.identities[person_id] = {
                'name': person_name,
                'embeddings': list(cluster_embeddings),
                'clusters': [cluster_id],
                'first_seen': np.datetime64('now'),
                'last_seen': np.datetime64('now'),
                'photo_count': len(cluster_embeddings)
            }
            self.next_person_id += 1
        else:
            # Merge with existing person
            existing = self.identities[person_id]
            existing['embeddings'].extend(cluster_embeddings)
            existing['clusters'].append(cluster_id)
            existing['last_seen'] = np.datetime64('now')
            existing['photo_count'] += len(cluster_embeddings)

        # Update cluster
        self.clusters[cluster_id]['person_id'] = person_id

        self.save()
        return person_id

    def _find_similar_person(
        self,
        embeddings: List[np.ndarray],
        threshold: float = 0.7
    ) -> Optional[int]:
        """Find existing person with similar face embeddings"""
        if not self.identities:
            return None

        analyzer = FaceAnalyzer()

        for person_id, data in self.identities.items():
            person_embeddings = data['embeddings']

            # Check similarity between cluster and person embeddings
            for cluster_emb in embeddings:
                similar = analyzer.find_similar_faces(
                    cluster_emb,
                    person_embeddings,
                    threshold
                )
                if similar:
                    return person_id

        return None

    def search_person(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.6
    ) -> List[Tuple[int, float]]:
        """Search for person by face embedding"""
        results: List[Tuple[int, float]] = []
        analyzer = FaceAnalyzer()

        for person_id, data in self.identities.items():
            person_embeddings = data['embeddings']
            similar = analyzer.find_similar_faces(
                query_embedding,
                person_embeddings,
                threshold
            )
            if similar:
                best_sim = max(sim for _, sim in similar)
                results.append((person_id, best_sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
