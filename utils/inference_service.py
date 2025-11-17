"""
Inference Service with Fallback Mechanism

Supports multiple inference backends with automatic fallback:
1. OpenAI-compatible server (llama.cpp, LM Studio)
2. Local SentenceTransformers (fallback)

Usage:
    # From config file
    embedder = InferenceServer.from_config("./configs/config.yml", service_type="embeddings")
    llm = InferenceServer.from_config("./configs/config.yml", service_type="llm")

    # Manual configuration
    embedder = InferenceServer(
        url="http://localhost:8000/v1",
        fallback_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    embeddings = await embedder.encode(["text1", "text2"])

    # Chat/LLM
    llm = InferenceServer(
        url="http://localhost:1234/v1",
        model_name="llama-3.2-3b-instruct"
    )
    response = await llm.chat([{"role": "user", "content": "Hello"}])
"""

from openai import AsyncOpenAI
import asyncio
from typing import List, Union, Optional
import httpx
import yaml
from pathlib import Path


class InferenceServer:
    """
    Unified inference server with fallback mechanism.

    Tries OpenAI-compatible server first, falls back to local models if unavailable.
    """

    def __init__(
        self,
        url: str = "http://localhost:8000/v1",
        model_name: str = "default",
        fallback_model: Optional[str] = None,
        timeout: float = 5.0
    ):
        """
        Initialize inference server with fallback.

        Args:
            url: OpenAI-compatible server URL (llama.cpp, LM Studio)
            model_name: Model name for the server
            fallback_model: SentenceTransformers model name for fallback
            timeout: Timeout for server health check
        """
        self.url = url
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.timeout = timeout

        self.async_client = None
        self.local_model = None
        self.use_fallback = False

        # Try to connect to server
        self._check_server()

    @classmethod
    def from_config(cls, config_path: str, service_type: str = "llm"):
        """
        Create InferenceServer instance from config file.

        Args:
            config_path: Path to YAML config file
            service_type: Type of service - "llm" or "embeddings"

        Returns:
            InferenceServer instance configured from file

        Example:
            llm = InferenceServer.from_config("./configs/config.yml", "llm")
            embedder = InferenceServer.from_config("./configs/config.yml", "embeddings")
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if 'inference' not in config:
            raise ValueError("Config file missing 'inference' section")

        if service_type not in config['inference']:
            raise ValueError(f"Config file missing 'inference.{service_type}' section")

        service_config = config['inference'][service_type]

        return cls(
            url=service_config.get('url', 'http://localhost:8000/v1'),
            model_name=service_config.get('model_name', 'default'),
            fallback_model=service_config.get('fallback_model'),
            timeout=service_config.get('timeout', 5.0)
        )

    def _check_server(self):
        """Check if OpenAI-compatible server is available"""
        try:
            # Try to create client
            self.async_client = AsyncOpenAI(
                api_key="none",
                base_url=self.url,
                timeout=self.timeout
            )

            # Test connection synchronously
            import requests
            health_url = self.url.replace("/v1", "/health")
            try:
                response = requests.get(health_url, timeout=self.timeout)
                print(f"[InferenceServer] ✓ Connected to server at {self.url}")
                self.use_fallback = False
            except:
                # Try models endpoint
                try:
                    models_url = f"{self.url}/models"
                    response = requests.get(models_url, timeout=self.timeout)
                    print(f"[InferenceServer] ✓ Connected to server at {self.url}")
                    self.use_fallback = False
                except:
                    raise ConnectionError("Server not reachable")

        except Exception as e:
            print(f"[InferenceServer] !  Server unavailable at {self.url}: {e}")
            self._init_fallback()

    def _init_fallback(self):
        """Initialize local fallback model"""
        if self.fallback_model:
            print(f"[InferenceServer]  Falling back to local model: {self.fallback_model}")
            try:
                from sentence_transformers import SentenceTransformer
                self.local_model = SentenceTransformer(self.fallback_model)
                self.use_fallback = True
                print(f"[InferenceServer] ✓ Loaded local model: {self.fallback_model}")
            except Exception as e:
                print(f"[InferenceServer] ✗ Failed to load fallback model: {e}")
                raise
        else:
            print(f"[InferenceServer] ✗ No fallback model specified")
            raise ConnectionError(f"Server unavailable and no fallback configured")

    async def encode(
        self,
        messages: Union[str, List[str]],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False
    ) -> List[List[float]]:
        """
        Encode text(s) to embeddings.

        Args:
            messages: Single text or list of texts
            batch_size: Batch size for encoding (fallback only)
            normalize_embeddings: Normalize embeddings (fallback only)
            show_progress_bar: Show progress bar (fallback only)

        Returns:
            List of embedding vectors
        """
        # Ensure messages is a list
        if isinstance(messages, str):
            messages = [messages]

        if self.use_fallback:
            # Use local SentenceTransformers
            embeddings = self.local_model.encode(
                messages,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True  # Convert to numpy array
            )
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        else:
            # Use OpenAI-compatible server
            try:
                response = await self.async_client.embeddings.create(
                    model=self.model_name,
                    input=messages
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                print(f"[InferenceServer] Server error, attempting fallback: {e}")
                if self.fallback_model:
                    self._init_fallback()
                    return await self.encode(messages, batch_size, normalize_embeddings, show_progress_bar)
                else:
                    raise

    async def chat(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        if self.use_fallback:
            raise NotImplementedError("Chat fallback not implemented. Use a server for LLM inference.")

        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            response = await self.async_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[InferenceServer] Chat error: {e}")
            raise

    def encode_sync(
        self,
        messages: Union[str, List[str]],
        **kwargs
    ) -> List[List[float]]:
        """Synchronous version of encode()"""
        return asyncio.run(self.encode(messages, **kwargs))

    def is_using_fallback(self) -> bool:
        """Check if currently using fallback"""
        return self.use_fallback


async def test_inference_service(config_path: str = "./configs/config.yml"):
    """
    Test the inference service with config-based initialization.

    Args:
        config_path: Path to YAML config file (default: "./configs/config.yml")
    """

    print("="*70)
    print("Testing InferenceServer from Config")
    print("="*70)

    # Test 1: Embeddings from config
    print("\n1. Testing embeddings from config...")
    try:
        embedder = InferenceServer.from_config(
            config_path=config_path,
            service_type="embeddings"
        )

        print(f"   Config: {embedder.url} (model: {embedder.model_name})")

        texts = ["Hello world", "Bonjour le monde"]
        embeddings = await embedder.encode(texts)
        print(f"   Encoded {len(texts)} texts")
        print(f"   Embedding dimension: {len(embeddings[0])}")
        print(f"   Using fallback: {embedder.is_using_fallback()}")
    except Exception as e:
        print(f"   Embeddings test failed: {e}")

    # Test 2: Chat from config
    print("\n2. Testing chat from config...")
    try:
        llm = InferenceServer.from_config(
            config_path=config_path,
            service_type="llm"
        )

        print(f"   Config: {llm.url} (model: {llm.model_name})")

        messages = [{"role": "user", "content": "Translate 'bread' to Italian"}]
        response = await llm.chat(messages)
        print(f"   Response: {response}")

    except Exception as e:
        print(f"   Chat not available: {e}")

    print("\n" + "="*70)
    print("✓ Tests completed")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(test_inference_service())
