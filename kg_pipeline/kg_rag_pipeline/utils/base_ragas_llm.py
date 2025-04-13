from typing import List, Optional, Dict, Any
import google.generativeai as genai
from ragas.llms.base import BaseRagasLLM, LLMResult, RunConfig
from ragas.prompt.base import BasePrompt
import time

class GeminiClient:
    def __init__(self, api_key: str , model_name: str = "gemini-2.0-flash", temperature: float = 0.3):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def generate_content(self, prompt: str, temperature: float = None) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature if temperature is not None else self.temperature,
                    "max_output_tokens": 1024,
                }
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini Error: {e}")
            return ""

class GeminiLLMWrapper(BaseRagasLLM):
    """
    A wrapper for the Gemini API that implements the BaseRagasLLM interface.
    
    This wrapper is designed to work with RAGAS evaluation framework.
    """
    
    def __init__(self, client: GeminiClient, retry_attempts: int = 3, retry_delay: int = 2):
        """
        Initialize the GeminiLLMWrapper with a client.
        
        Args:
            client: The GeminiClient to use for generation
            retry_attempts: Number of retry attempts for failed API calls
            retry_delay: Base delay between retries (will use exponential backoff)
        """
        super().__init__()
        self.client = client
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.run_config = RunConfig()
        
    def _process_response(self, text: str) -> LLMResult:
        """
        Convert raw text response to LLMResult format expected by RAGAS.
        
        Args:
            text: The text response from the model
            
        Returns:
            An LLMResult with the text as completion
        """
    # Create a generation object in the format RAGAS expects
        generation = {
            "text": text,
            "generation_info": {
                # Include any other metadata you want to track
                "model": self.client.model_name,
            }
        }
        
        # Return the list of generations 
        return LLMResult(
            generations=[[generation]],  # Note the double brackets - this is important!
            llm_output={"model_name": self.client.model_name}
        )
        
    def generate_text(
        self,

        prompt,
        n: int = 1,
        temperature: float = 1e-8,
        stop: Optional[List[str]] = None,
        callbacks = None,
    ) -> LLMResult:
        """
        Generate text using the Gemini API.
        
        Args:
            prompt: The prompt to generate text from
            n: Number of completions to generate (not supported by Gemini API)
            temperature: The temperature to use for generation
            stop: Stop sequences (not supported by Gemini API)
            callbacks: Callbacks for generation events
            
        Returns:
            An LLMResult with the generated completions
        """
        prompt_text = prompt.to_string()
        for attempt in range(self.retry_attempts):
            try:
                response_text = self.client.generate_content(prompt_text, temperature)
                if response_text:
                    return self._process_response(response_text)
                
            except Exception as e:
                print(f"Error on attempt {attempt+1}/{self.retry_attempts}: {e}")
            
            # Wait before retrying (exponential backoff)
            wait_time = self.retry_delay * (2 ** attempt)
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        
        print("All generation attempts failed")
        return self._process_response("Unable to generate a response after multiple attempts.")
    
    async def agenerate_text(
        self,
        prompt,
        n: int = 1,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        callbacks = None,
    ) -> LLMResult:
        """
        Asynchronously generate text (fallback to synchronous for now).
        
        Args:
            prompt: The prompt to generate text from
            n: Number of completions to generate
            temperature: The temperature to use for generation
            stop: Stop sequences
            callbacks: Callbacks for generation events
            
        Returns:
            An LLMResult with the generated completions
        """
        # For simplicity, we'll just call the synchronous version
        # In a production environment, you should implement a proper async version
        temp = temperature if temperature is not None else self.get_temperature(n)
        return self.generate_text(prompt, n, temp, stop, callbacks)
    
    def is_finished(self, response: LLMResult) -> bool:
        """
        Check if the generation is finished.
        
        Args:
            response: The response to check
            
        Returns:
            True if the generation is finished, False otherwise
        """
        # For Gemini, we always consider the generation finished once we get a response
        return True
    
    def set_run_config(self, run_config: RunConfig):
        """
        Set the run configuration.
        
        Args:
            run_config: The run configuration to use
        """
        self.run_config = run_config
    
    def __repr__(self) -> str:
        """
        Get a string representation of the wrapper.
        
        Returns:
            A string representation of the wrapper
        """
        return f"GeminiLLMWrapper(model={self.client.model_name})"