"""
MLLM Evaluator for VULCA Framework

This module provides the core functionality for evaluating Multimodal Large Language Models
using Chinese painting critiques with persona-guided prompting.
"""

import os
import json
import base64
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path


class MLLMEvaluator:
    """
    Main evaluator class for MLLM critique generation.
    
    Supports both local vLLM server and Hugging Face API endpoints.
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        api_endpoint: str = "http://localhost:8000",
        api_type: str = "vllm",  # "vllm" or "huggingface"
        api_key: Optional[str] = None
    ):
        """
        Initialize the MLLM evaluator.
        
        Args:
            model_name: Name/ID of the model to use
            api_endpoint: API endpoint URL
            api_type: Type of API ("vllm" or "huggingface")
            api_key: API key (for HuggingFace, can be set via HF_TOKEN env var)
        """
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.api_type = api_type
        self.api_key = api_key or os.getenv("HF_TOKEN")
        
        # Set up API endpoints
        if api_type == "vllm":
            self.api_url = f"{api_endpoint}/v1/chat/completions"
        else:
            self.api_url = api_endpoint
            
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def generate_critique(
        self,
        image_path: str,
        prompt: str,
        persona: Optional[str] = None,
        knowledge_context: Optional[List[str]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate a critique for the given image.
        
        Args:
            image_path: Path to the image file
            prompt: Base prompt text
            persona: Optional persona description
            knowledge_context: Optional list of knowledge snippets
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Dictionary containing the critique and metadata
        """
        # Encode image
        encoded_image = self.encode_image(image_path)
        
        # Construct enhanced prompt
        full_prompt = self._construct_prompt(prompt, persona, knowledge_context)
        
        # Prepare API request
        if self.api_type == "vllm":
            payload = self._prepare_vllm_payload(
                encoded_image, full_prompt, max_tokens, temperature
            )
        else:
            payload = self._prepare_hf_payload(
                encoded_image, full_prompt, max_tokens, temperature
            )
        
        # Make API call
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_type == "huggingface":
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        response = requests.post(self.api_url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        if self.api_type == "vllm":
            critique_text = result["choices"][0]["message"]["content"]
        else:
            critique_text = result.get("generated_text", "")
            
        return {
            "critique": critique_text,
            "model": self.model_name,
            "persona": persona,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    
    def _construct_prompt(
        self, 
        base_prompt: str, 
        persona: Optional[str], 
        knowledge: Optional[List[str]]
    ) -> str:
        """Construct the full prompt with persona and knowledge."""
        parts = [base_prompt]
        
        if persona:
            parts.append(f"\n\nPersona: {persona}")
            
        if knowledge:
            parts.append("\n\nRelevant Knowledge:")
            for snippet in knowledge[:3]:  # Limit to top 3 snippets
                parts.append(f"- {snippet}")
                
        return "\n".join(parts)
    
    def _prepare_vllm_payload(
        self, 
        image_b64: str, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> Dict:
        """Prepare payload for vLLM OpenAI-compatible API."""
        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    
    def _prepare_hf_payload(
        self, 
        image_b64: str, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> Dict:
        """Prepare payload for HuggingFace API."""
        return {
            "inputs": {
                "image": image_b64,
                "text": prompt
            },
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True
            }
        }
    
    def evaluate_batch(
        self,
        image_paths: List[str],
        prompt: str,
        personas: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple images with optional multiple personas.
        
        Args:
            image_paths: List of image file paths
            prompt: Base prompt text
            personas: Optional list of personas (one per image or shared)
            **kwargs: Additional arguments for generate_critique
            
        Returns:
            List of critique results
        """
        results = []
        
        # Handle persona assignment
        if personas is None:
            personas = [None] * len(image_paths)
        elif len(personas) == 1:
            personas = personas * len(image_paths)
            
        for image_path, persona in zip(image_paths, personas):
            try:
                result = self.generate_critique(
                    image_path, prompt, persona, **kwargs
                )
                result["image_path"] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "image_path": image_path,
                    "persona": persona
                })
                
        return results