import asyncio
import typing as t
from abc import ABC
from dataclasses import field
import numpy as np
from langchain_core.embeddings import Embeddings
from pydantic.dataclasses import dataclass
from ragas.cache import CacheInterface, cacher
from ragas.run_config import RunConfig, add_async_retry, add_retry


class BaseRagasEmbeddings(Embeddings, ABC):
    """
    Abstract base class for Ragas embeddings.

    This class extends the Embeddings class and provides methods for embedding
    text and managing run configurations.

    Attributes:
        run_config (RunConfig): Configuration for running the embedding operations.

    """

    run_config: RunConfig
    cache: t.Optional[CacheInterface] = None

    def __init__(self, cache: t.Optional[CacheInterface] = None):
        super().__init__()
        self.cache = cache
        if self.cache is not None:
            self.embed_query = cacher(cache_backend=self.cache)(self.embed_query)
            self.embed_documents = cacher(cache_backend=self.cache)(
                self.embed_documents
            )
            self.aembed_query = cacher(cache_backend=self.cache)(self.aembed_query)
            self.aembed_documents = cacher(cache_backend=self.cache)(
                self.aembed_documents
            )

    async def embed_text(self, text: str, is_async=True) -> t.List[float]:
        """
        Embed a single text string.
        """
        embs = await self.embed_texts([text], is_async=is_async)
        return embs[0]

    async def embed_texts(
        self, texts: t.List[str], is_async: bool = True
    ) -> t.List[t.List[float]]:
        """
        Embed multiple texts.
        """
        if is_async:
            aembed_documents_with_retry = add_async_retry(
                self.aembed_documents, self.run_config
            )
            return await aembed_documents_with_retry(texts)
        else:
            loop = asyncio.get_event_loop()
            embed_documents_with_retry = add_retry(
                self.embed_documents, self.run_config
            )
            return await loop.run_in_executor(None, embed_documents_with_retry, texts)

from dataclasses import field
from pydantic.dataclasses import dataclass

@dataclass
class HuggingfaceEmbeddings:
    """
    HuggingfaceEmbeddings provides a wrapper around SentenceTransformer to be used with RAGAS.

    Attributes:
        model_name (str): The pretrained model to use (e.g., "all-MiniLM-L6-v2").

    Methods:
        embed_documents(texts: List[str]) -> List[List[float]]:
            Returns embeddings for a list of documents.
        embed_query(text: str) -> List[float]:
            Returns an embedding for a single query string.

    Why it's useful:
    - RAGAS needs access to embedding space for several of its metrics.
    - Provides standardized access to high-quality sentence embeddings.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_folder: t.Optional[str] = None
    model_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    encode_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    cache: t.Optional[CacheInterface] = None

    def __post_init__(self):
        super().__init__(cache=self.cache)
        try:
            import sentence_transformers
            from transformers import AutoConfig
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc

        config = AutoConfig.from_pretrained(self.model_name)

        self.is_cross_encoder = bool(
            np.intersect1d(
                list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values()),
                config.architectures,
            )
        )

        if self.is_cross_encoder:
            self.model = sentence_transformers.CrossEncoder(
                self.model_name, **self.model_kwargs
            )
        else:
            self.model = sentence_transformers.SentenceTransformer(
                self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
            )

        if "convert_to_tensor" not in self.encode_kwargs:
            self.encode_kwargs["convert_to_tensor"] = True

        if self.cache is not None:
            self.predict = cacher(cache_backend=self.cache)(self.predict)


    def embed_query(self, text: str) -> t.List[float]:
        """
        Embed a single query text.
        """
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        """
        Embed multiple documents.
        """
        from sentence_transformers.SentenceTransformer import SentenceTransformer
        from torch import Tensor

        assert isinstance(
            self.model, SentenceTransformer
        ), "Model is not of the type Bi-encoder"
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, **self.encode_kwargs
        )

        assert isinstance(embeddings, Tensor)
        return embeddings.tolist()

    def predict(self, texts: t.List[t.List[str]]) -> t.List[t.List[float]]:
        """
        Make predictions using a cross-encoder model.
        """
        if not self.is_cross_encoder:
            raise NotImplementedError("Predict is only available for cross-encoder models")
        return self.model.predict(texts).tolist() 

