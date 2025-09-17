"""
Multimodal Retriever for Hybrid RAG System

This module provides comprehensive multimodal retrieval capabilities including:
- Text-based document retrieval
- Image content analysis and retrieval
- Audio transcription and search
- Video content analysis
- Cross-modal similarity search
- Fusion of multiple modality results
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import asyncio
from pathlib import Path
import base64
import io
import hashlib

import numpy as np
import torch
from PIL import Image
import cv2
from sentence_transformers import SentenceTransformer
import clip
import whisper
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModel, AutoImageProcessor
)

from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultimodalDocument:
    """Represents a multimodal document with multiple content types"""
    id: str
    text_content: str = ""
    image_content: Optional[np.ndarray] = None
    audio_content: Optional[np.ndarray] = None
    video_content: Optional[List[np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)

@dataclass
class RetrievalResult:
    """Represents a retrieval result with score and modality information"""
    document: MultimodalDocument
    score: float
    modality: str  # text, image, audio, video, fusion
    explanation: str = ""

class MultimodalEmbeddings:
    """Handles embeddings for different modalities"""

    def __init__(
        self,
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        image_model: str = "openai/clip-vit-base-patch32",
        audio_model: str = "openai/whisper-base"
    ):
        # Text embeddings
        self.text_encoder = SentenceTransformer(text_model)

        # Image embeddings
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        self.clip_model.eval()

        # Image captioning for better text-image alignment
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # Audio embeddings (using Whisper for transcription + text embeddings)
        self.whisper_model = whisper.load_model("base")

        logger.info("Initialized multimodal embeddings")

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text into embeddings"""
        if not texts:
            return np.array([])
        return self.text_encoder.encode(texts, convert_to_tensor=False)

    def encode_image(self, images: List[np.ndarray]) -> np.ndarray:
        """Encode images into embeddings using CLIP"""
        if not images:
            return np.array([])

        embeddings = []
        with torch.no_grad():
            for img_array in images:
                # Convert numpy array to PIL Image
                if img_array.dtype != np.uint8:
                    img_array = (img_array * 255).astype(np.uint8)

                pil_image = Image.fromarray(img_array)
                image_input = self.clip_preprocess(pil_image).unsqueeze(0)

                # Get CLIP embedding
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.cpu().numpy().flatten())

        return np.array(embeddings)

    def generate_image_caption(self, image: np.ndarray) -> str:
        """Generate caption for an image using BLIP"""
        try:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            pil_image = Image.fromarray(image)
            inputs = self.blip_processor(pil_image, return_tensors="pt")

            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)

            return caption
        except Exception as e:
            logger.error(f"Error generating image caption: {e}")
            return ""

    def encode_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Tuple[np.ndarray, str]:
        """Encode audio by transcribing it and then encoding the text"""
        try:
            # Transcribe audio using Whisper
            result = self.whisper_model.transcribe(audio_data)
            transcript = result["text"]

            # Encode transcript as text
            if transcript.strip():
                embedding = self.encode_text([transcript])[0]
                return embedding, transcript
            else:
                return np.array([]), ""

        except Exception as e:
            logger.error(f"Error encoding audio: {e}")
            return np.array([]), ""

    def encode_video(self, video_frames: List[np.ndarray], audio_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Encode video by processing frames and audio separately"""
        try:
            # Sample frames (take every nth frame to avoid redundancy)
            n_frames = len(video_frames)
            sample_rate = max(1, n_frames // 10)  # Sample ~10 frames
            sampled_frames = video_frames[::sample_rate]

            # Encode sampled frames
            frame_embeddings = self.encode_image(sampled_frames)

            # Generate captions for keyframes
            captions = []
            for frame in sampled_frames[:5]:  # Caption first 5 frames
                caption = self.generate_image_caption(frame)
                if caption:
                    captions.append(caption)

            # Encode captions as text
            caption_text = " ".join(captions)
            text_embedding = self.encode_text([caption_text])[0] if caption_text else np.array([])

            # Encode audio if available
            audio_embedding = np.array([])
            audio_transcript = ""
            if audio_data is not None:
                audio_embedding, audio_transcript = self.encode_audio(audio_data)

            return {
                'frame_embeddings': frame_embeddings,
                'text_embedding': text_embedding,
                'audio_embedding': audio_embedding,
                'captions': captions,
                'audio_transcript': audio_transcript
            }

        except Exception as e:
            logger.error(f"Error encoding video: {e}")
            return {}


class MultimodalRetriever:
    """
    Main multimodal retriever that handles text, image, audio, and video content
    """

    def __init__(
        self,
        store_path: str = "hybrid_rag/stores/multimodal",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        fusion_method: str = "weighted_average",  # weighted_average, max_pooling, learned_fusion
        modality_weights: Dict[str, float] = None
    ):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        # Initialize embeddings
        self.embeddings = MultimodalEmbeddings(text_model=embedding_model)

        # Vector stores for different modalities
        self.text_vectorstore = None
        self.image_vectorstore = None
        self.audio_vectorstore = None
        self.video_vectorstore = None

        # Document storage
        self.documents: Dict[str, MultimodalDocument] = {}

        # Fusion configuration
        self.fusion_method = fusion_method
        self.modality_weights = modality_weights or {
            'text': 1.0,
            'image': 0.8,
            'audio': 0.6,
            'video': 0.7
        }

        # Caching
        self.embedding_cache = {}

        logger.info(f"Initialized MultimodalRetriever with store path: {self.store_path}")

    def _generate_doc_id(self, content: str) -> str:
        """Generate unique document ID"""
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from file path"""
        try:
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def _load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """Load audio from file path"""
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            return audio
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return None

    def _load_video(self, video_path: str) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
        """Load video frames and audio from file path"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

            # Extract audio using moviepy
            try:
                from moviepy.editor import VideoFileClip
                clip = VideoFileClip(video_path)
                if clip.audio:
                    audio = clip.audio.to_soundarray(fps=16000)
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)  # Convert to mono
                    clip.close()
                    return frames, audio
                else:
                    clip.close()
                    return frames, None
            except:
                return frames, None

        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return None, None

    def add_text_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """Add a text document to the retriever"""
        doc_id = self._generate_doc_id(text)
        metadata = metadata or {}

        # Create multimodal document
        doc = MultimodalDocument(
            id=doc_id,
            text_content=text,
            metadata=metadata
        )

        # Generate text embedding
        text_embedding = self.embeddings.encode_text([text])[0]
        doc.embeddings['text'] = text_embedding

        self.documents[doc_id] = doc

        # Add to text vectorstore
        if self.text_vectorstore is None:
            from langchain.embeddings import HuggingFaceEmbeddings
            hf_embeddings = HuggingFaceEmbeddings(model_name=self.embeddings.text_encoder.get_sentence_embedding_dimension)
            self.text_vectorstore = FAISS.from_texts([text], hf_embeddings, metadatas=[metadata])
        else:
            self.text_vectorstore.add_texts([text], metadatas=[metadata])

        logger.debug(f"Added text document: {doc_id}")
        return doc_id

    def add_image_document(self, image_path: str, metadata: Dict[str, Any] = None) -> Optional[str]:
        """Add an image document to the retriever"""
        image = self._load_image(image_path)
        if image is None:
            return None

        metadata = metadata or {}
        metadata['image_path'] = image_path

        # Generate caption for the image
        caption = self.embeddings.generate_image_caption(image)
        doc_id = self._generate_doc_id(f"image_{image_path}_{caption}")

        # Create multimodal document
        doc = MultimodalDocument(
            id=doc_id,
            text_content=caption,
            image_content=image,
            metadata=metadata
        )

        # Generate embeddings
        image_embedding = self.embeddings.encode_image([image])[0]
        text_embedding = self.embeddings.encode_text([caption])[0] if caption else np.array([])

        doc.embeddings['image'] = image_embedding
        if caption:
            doc.embeddings['text'] = text_embedding

        self.documents[doc_id] = doc

        logger.debug(f"Added image document: {doc_id}")
        return doc_id

    def add_audio_document(self, audio_path: str, metadata: Dict[str, Any] = None) -> Optional[str]:
        """Add an audio document to the retriever"""
        audio = self._load_audio(audio_path)
        if audio is None:
            return None

        metadata = metadata or {}
        metadata['audio_path'] = audio_path

        # Transcribe and encode audio
        audio_embedding, transcript = self.embeddings.encode_audio(audio)
        doc_id = self._generate_doc_id(f"audio_{audio_path}_{transcript}")

        # Create multimodal document
        doc = MultimodalDocument(
            id=doc_id,
            text_content=transcript,
            audio_content=audio,
            metadata=metadata
        )

        # Store embeddings
        if audio_embedding.size > 0:
            doc.embeddings['audio'] = audio_embedding
            doc.embeddings['text'] = audio_embedding  # Audio is encoded as text

        self.documents[doc_id] = doc

        logger.debug(f"Added audio document: {doc_id}")
        return doc_id

    def add_video_document(self, video_path: str, metadata: Dict[str, Any] = None) -> Optional[str]:
        """Add a video document to the retriever"""
        frames, audio = self._load_video(video_path)
        if frames is None:
            return None

        metadata = metadata or {}
        metadata['video_path'] = video_path

        # Encode video
        video_data = self.embeddings.encode_video(frames, audio)
        doc_id = self._generate_doc_id(f"video_{video_path}")

        # Combine captions and transcript
        combined_text = " ".join(video_data.get('captions', []))
        if video_data.get('audio_transcript'):
            combined_text += " " + video_data['audio_transcript']

        # Create multimodal document
        doc = MultimodalDocument(
            id=doc_id,
            text_content=combined_text,
            video_content=frames,
            audio_content=audio,
            metadata=metadata
        )

        # Store embeddings
        doc.embeddings.update({
            'video_frames': video_data.get('frame_embeddings', np.array([])),
            'text': video_data.get('text_embedding', np.array([])),
            'audio': video_data.get('audio_embedding', np.array([]))
        })

        self.documents[doc_id] = doc

        logger.debug(f"Added video document: {doc_id}")
        return doc_id

    def add_multimodal_document(
        self,
        text: str = "",
        image_path: str = "",
        audio_path: str = "",
        video_path: str = "",
        metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """Add a document with multiple modalities"""
        metadata = metadata or {}
        content_parts = []

        # Collect content for ID generation
        if text:
            content_parts.append(f"text_{text[:50]}")
        if image_path:
            content_parts.append(f"image_{image_path}")
        if audio_path:
            content_parts.append(f"audio_{audio_path}")
        if video_path:
            content_parts.append(f"video_{video_path}")

        if not content_parts:
            logger.error("No content provided for multimodal document")
            return None

        doc_id = self._generate_doc_id("_".join(content_parts))

        # Load content
        image = self._load_image(image_path) if image_path else None
        audio = self._load_audio(audio_path) if audio_path else None
        video_frames, video_audio = self._load_video(video_path) if video_path else (None, None)

        # Generate captions and transcripts
        combined_text = text
        if image is not None:
            caption = self.embeddings.generate_image_caption(image)
            combined_text += " " + caption

        if audio is not None:
            _, transcript = self.embeddings.encode_audio(audio)
            combined_text += " " + transcript

        if video_frames is not None:
            video_data = self.embeddings.encode_video(video_frames, video_audio)
            combined_text += " " + " ".join(video_data.get('captions', []))
            if video_data.get('audio_transcript'):
                combined_text += " " + video_data['audio_transcript']

        # Create multimodal document
        doc = MultimodalDocument(
            id=doc_id,
            text_content=combined_text.strip(),
            image_content=image,
            audio_content=audio or video_audio,
            video_content=video_frames,
            metadata=metadata
        )

        # Generate embeddings for each modality
        if combined_text.strip():
            doc.embeddings['text'] = self.embeddings.encode_text([combined_text])[0]

        if image is not None:
            doc.embeddings['image'] = self.embeddings.encode_image([image])[0]

        if audio is not None:
            audio_emb, _ = self.embeddings.encode_audio(audio)
            if audio_emb.size > 0:
                doc.embeddings['audio'] = audio_emb

        if video_frames is not None:
            video_data = self.embeddings.encode_video(video_frames, video_audio)
            doc.embeddings.update({
                'video_frames': video_data.get('frame_embeddings', np.array([])),
                'video_text': video_data.get('text_embedding', np.array([])),
                'video_audio': video_data.get('audio_embedding', np.array([]))
            })

        self.documents[doc_id] = doc

        logger.debug(f"Added multimodal document: {doc_id}")
        return doc_id

    def _compute_similarity(self, query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        if query_embedding.size == 0 or doc_embedding.size == 0:
            return 0.0

        # Ensure both embeddings are 1D
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding.flatten()
        if len(doc_embedding.shape) > 1:
            doc_embedding = doc_embedding.flatten()

        # Compute cosine similarity
        dot_product = np.dot(query_embedding, doc_embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_doc = np.linalg.norm(doc_embedding)

        if norm_query == 0 or norm_doc == 0:
            return 0.0

        return dot_product / (norm_query * norm_doc)

    def retrieve_text(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve documents based on text query"""
        query_embedding = self.embeddings.encode_text([query])[0]
        results = []

        for doc_id, doc in self.documents.items():
            if 'text' in doc.embeddings and doc.embeddings['text'].size > 0:
                similarity = self._compute_similarity(query_embedding, doc.embeddings['text'])
                results.append(RetrievalResult(
                    document=doc,
                    score=similarity,
                    modality='text',
                    explanation=f"Text similarity: {similarity:.3f}"
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def retrieve_image(self, image_path: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve documents based on image query"""
        query_image = self._load_image(image_path)
        if query_image is None:
            return []

        query_embedding = self.embeddings.encode_image([query_image])[0]
        results = []

        for doc_id, doc in self.documents.items():
            if 'image' in doc.embeddings and doc.embeddings['image'].size > 0:
                similarity = self._compute_similarity(query_embedding, doc.embeddings['image'])
                results.append(RetrievalResult(
                    document=doc,
                    score=similarity,
                    modality='image',
                    explanation=f"Image similarity: {similarity:.3f}"
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def retrieve_multimodal(
        self,
        query_text: str = "",
        query_image_path: str = "",
        query_audio_path: str = "",
        k: int = 5,
        modality_weights: Dict[str, float] = None
    ) -> List[RetrievalResult]:
        """Retrieve documents using multimodal query"""
        weights = modality_weights or self.modality_weights
        query_embeddings = {}

        # Generate query embeddings
        if query_text:
            query_embeddings['text'] = self.embeddings.encode_text([query_text])[0]

        if query_image_path:
            query_image = self._load_image(query_image_path)
            if query_image is not None:
                query_embeddings['image'] = self.embeddings.encode_image([query_image])[0]

        if query_audio_path:
            query_audio = self._load_audio(query_audio_path)
            if query_audio is not None:
                query_emb, _ = self.embeddings.encode_audio(query_audio)
                if query_emb.size > 0:
                    query_embeddings['audio'] = query_emb

        if not query_embeddings:
            return []

        # Compute similarities for each document
        results = []
        for doc_id, doc in self.documents.items():
            modality_scores = {}

            # Compute similarity for each modality
            for modality, query_emb in query_embeddings.items():
                if modality in doc.embeddings and doc.embeddings[modality].size > 0:
                    similarity = self._compute_similarity(query_emb, doc.embeddings[modality])
                    modality_scores[modality] = similarity

            if not modality_scores:
                continue

            # Fuse scores based on fusion method
            if self.fusion_method == "weighted_average":
                total_weight = sum(weights.get(mod, 1.0) for mod in modality_scores.keys())
                fused_score = sum(
                    score * weights.get(mod, 1.0) for mod, score in modality_scores.items()
                ) / total_weight if total_weight > 0 else 0.0

            elif self.fusion_method == "max_pooling":
                fused_score = max(modality_scores.values())

            else:  # Simple average
                fused_score = sum(modality_scores.values()) / len(modality_scores)

            explanation = ", ".join([f"{mod}: {score:.3f}" for mod, score in modality_scores.items()])

            results.append(RetrievalResult(
                document=doc,
                score=fused_score,
                modality='fusion',
                explanation=f"Multimodal fusion ({explanation})"
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def save_index(self):
        """Save the multimodal index to disk"""
        logger.info(f"Saving multimodal index to {self.store_path}")

        # Save documents (excluding large binary data)
        docs_data = {}
        for doc_id, doc in self.documents.items():
            doc_data = {
                'id': doc.id,
                'text_content': doc.text_content,
                'metadata': doc.metadata,
                'embeddings': {}
            }

            # Convert embeddings to lists for JSON serialization
            for modality, embedding in doc.embeddings.items():
                if embedding.size > 0:
                    doc_data['embeddings'][modality] = embedding.tolist()

            docs_data[doc_id] = doc_data

        with open(self.store_path / "documents.json", 'w') as f:
            json.dump(docs_data, f, indent=2)

        # Save configuration
        config = {
            'fusion_method': self.fusion_method,
            'modality_weights': self.modality_weights
        }

        with open(self.store_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info("Multimodal index saved successfully")

    def load_index(self):
        """Load the multimodal index from disk"""
        logger.info(f"Loading multimodal index from {self.store_path}")

        try:
            # Load documents
            docs_path = self.store_path / "documents.json"
            if docs_path.exists():
                with open(docs_path, 'r') as f:
                    docs_data = json.load(f)

                self.documents = {}
                for doc_id, doc_data in docs_data.items():
                    # Convert embeddings back to numpy arrays
                    embeddings = {}
                    for modality, embedding_list in doc_data['embeddings'].items():
                        embeddings[modality] = np.array(embedding_list)

                    doc = MultimodalDocument(
                        id=doc_data['id'],
                        text_content=doc_data['text_content'],
                        metadata=doc_data['metadata'],
                        embeddings=embeddings
                    )

                    self.documents[doc_id] = doc

            # Load configuration
            config_path = self.store_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.fusion_method = config.get('fusion_method', 'weighted_average')
                    self.modality_weights = config.get('modality_weights', self.modality_weights)

            logger.info(f"Multimodal index loaded successfully. Documents: {len(self.documents)}")
            return True

        except Exception as e:
            logger.error(f"Error loading multimodal index: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the multimodal index"""
        modality_counts = {'text': 0, 'image': 0, 'audio': 0, 'video': 0}
        total_docs = len(self.documents)

        for doc in self.documents.values():
            if doc.text_content:
                modality_counts['text'] += 1
            if doc.image_content is not None:
                modality_counts['image'] += 1
            if doc.audio_content is not None:
                modality_counts['audio'] += 1
            if doc.video_content is not None:
                modality_counts['video'] += 1

        return {
            'total_documents': total_docs,
            'modality_counts': modality_counts,
            'fusion_method': self.fusion_method,
            'modality_weights': self.modality_weights
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize the multimodal retriever
    retriever = MultimodalRetriever()

    # Add some sample documents
    doc1_id = retriever.add_text_document(
        "Apple Inc. is a technology company that designs and manufactures consumer electronics.",
        metadata={"source": "tech_companies.txt"}
    )

    # Example multimodal search
    results = retriever.retrieve_text("technology company", k=3)
    print(f"Text retrieval results: {len(results)}")

    for result in results:
        print(f"Score: {result.score:.3f}, Modality: {result.modality}")
        print(f"Content: {result.document.text_content[:100]}...")
        print(f"Explanation: {result.explanation}\n")

    # Save the index
    retriever.save_index()

    # Get statistics
    stats = retriever.get_stats()
    print(f"Index stats: {stats}")