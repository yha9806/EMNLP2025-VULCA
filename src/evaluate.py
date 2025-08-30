#!/usr/bin/env python
"""
VULCA Framework - MLLM Evaluation Module
Generates art critiques using Multimodal Large Language Models (MLLMs)
"""

import argparse
import os
import json
import base64
from datetime import datetime
from typing import Dict, Optional, Any
import requests
import yaml


def load_config(config_path: str = "configs/model_config.yaml") -> Dict:
    """Load model configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """
    Encode image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (base64_string, mime_type)
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            
        # Determine MIME type
        mime_type = "image/jpeg"  # Default
        if image_path.lower().endswith(".png"):
            mime_type = "image/png"
        elif image_path.lower().endswith(".webp"):
            mime_type = "image/webp"
            
        return encoded_string, mime_type
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise Exception(f"Failed to encode image: {e}")


def call_mllm_api(
    image_path: str,
    prompt_text: str,
    model_name: str,
    api_endpoint: str = "http://localhost:8000/v1/chat/completions",
    model_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Call MLLM API with image and prompt.
    
    Args:
        image_path: Path to input image
        prompt_text: Text prompt for the model
        model_name: Model identifier
        api_endpoint: API endpoint URL
        model_params: Additional model parameters
        
    Returns:
        Generated critique text
    """
    # Encode image
    encoded_image, mime_type = encode_image_to_base64(image_path)
    
    # Construct payload
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_image}"
                    }
                }
            ]
        }]
    }
    
    # Add model parameters
    if model_params:
        if 'max_new_tokens' in model_params:
            payload['max_tokens'] = model_params['max_new_tokens']
        if 'temperature' in model_params:
            payload['temperature'] = model_params['temperature']
    
    # Make API call
    try:
        response = requests.post(
            api_endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=180
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Extract generated text
        if 'choices' in result and result['choices']:
            return result['choices'][0]['message']['content'].strip()
        else:
            raise ValueError(f"Unexpected API response format: {result}")
            
    except requests.exceptions.Timeout:
        raise TimeoutError(f"API request timed out after 180 seconds")
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(f"Failed to connect to API server: {e}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP error during API call: {e}")


def load_prompt(prompt_file: Optional[str] = None) -> str:
    """Load prompt from file or return default prompt."""
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Default prompt
    return """你是一位专业的艺术评论家。请仔细观察提供的图像，然后撰写一段约300-500字的艺术评论。

你的评论应包括：
1. 整体印象与主题内容
2. 构图布局
3. 笔墨技巧与色彩运用
4. 细节描绘
5. 文化内涵

请确保评论语言流畅，观点明确，分析具有深度。"""


def load_persona(persona_file: Optional[str] = None) -> str:
    """Load persona description from file."""
    if persona_file and os.path.exists(persona_file):
        with open(persona_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


def generate_critique(
    image_path: str,
    model_name: str,
    prompt_file: Optional[str] = None,
    persona_file: Optional[str] = None,
    knowledge_file: Optional[str] = None,
    api_endpoint: str = "http://localhost:8000/v1/chat/completions",
    model_params: Optional[Dict] = None,
    output_dir: str = "outputs/critiques"
) -> str:
    """
    Generate a critique for an image using MLLM.
    
    Args:
        image_path: Path to input image
        model_name: Model identifier
        prompt_file: Path to prompt template file
        persona_file: Path to persona description file
        knowledge_file: Path to knowledge base file
        api_endpoint: API endpoint URL
        model_params: Model generation parameters
        output_dir: Directory to save generated critiques
        
    Returns:
        Path to saved critique file
    """
    # Load components
    base_prompt = load_prompt(prompt_file)
    persona = load_persona(persona_file)
    
    # Construct full prompt
    full_prompt = base_prompt
    if persona:
        full_prompt = f"{persona}\n\n{base_prompt}"
    
    # Generate critique
    critique_text = call_mllm_api(
        image_path=image_path,
        prompt_text=full_prompt,
        model_name=model_name,
        api_endpoint=api_endpoint,
        model_params=model_params
    )
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    model_safe = model_name.replace('/', '_')
    
    output_file = os.path.join(
        output_dir,
        f"{base_name}_{model_safe}_{timestamp}.txt"
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(critique_text)
    
    print(f"Critique saved to: {output_file}")
    return output_file


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate art critiques using MLLMs"
    )
    parser.add_argument(
        "--image_path", 
        type=str, 
        required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model identifier"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        help="Path to prompt template file"
    )
    parser.add_argument(
        "--persona_file",
        type=str,
        help="Path to persona description file"
    )
    parser.add_argument(
        "--api_endpoint",
        type=str,
        default="http://localhost:8000/v1/chat/completions",
        help="API endpoint URL"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/critiques",
        help="Output directory"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode"
    )
    
    args = parser.parse_args()
    
    if args.test:
        print("Evaluation module loaded successfully.")
        return
    
    # Load config if exists
    config = load_config(args.config)
    model_params = config.get('models', {}).get(args.model_name, {})
    
    # Generate critique
    output_file = generate_critique(
        image_path=args.image_path,
        model_name=args.model_name,
        prompt_file=args.prompt_file,
        persona_file=args.persona_file,
        api_endpoint=args.api_endpoint,
        model_params=model_params,
        output_dir=args.output_dir
    )
    
    print(f"Generation complete: {output_file}")


if __name__ == "__main__":
    main()