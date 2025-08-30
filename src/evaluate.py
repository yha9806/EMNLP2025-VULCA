#!/usr/bin/env python
"""
VULCA Framework - MLLM Evaluation Module
Simplified version for EMNLP 2025
"""

import os
import json
import base64
from datetime import datetime
from typing import Dict, Optional, Any
import requests




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


def construct_prompt(persona_text: str = "", knowledge_context: str = "") -> str:
    """Construct evaluation prompt with optional persona and knowledge enhancement."""
    base_prompt = """你是一位专业的艺术评论家。请仔细观察提供的图像，然后撰写一段约300-500字的艺术评论。

你的评论应包括：
1. 整体印象与主题内容
2. 构图布局
3. 笔墨技巧与色彩运用
4. 细节描绘
5. 文化内涵

请确保评论语言流畅，观点明确，分析具有深度。"""
    
    # Add persona if provided
    if persona_text:
        base_prompt = f"{persona_text}\n\n{base_prompt}"
    
    # Add knowledge context if provided
    if knowledge_context:
        base_prompt = f"{base_prompt}\n\n相关背景知识：\n{knowledge_context}"
    
    return base_prompt




def generate_critique(
    image_path: str,
    model_name: str,
    persona_text: str = "",
    knowledge_base: Optional[Dict] = None,
    api_endpoint: str = "http://localhost:8000/v1/chat/completions",
    model_params: Optional[Dict] = None,
    output_dir: str = "outputs/critiques"
) -> str:
    """
    Generate a critique for an image using MLLM.
    
    Args:
        image_path: Path to input image
        model_name: Model identifier
        persona_text: Persona description text
        knowledge_base: Knowledge base dictionary
        api_endpoint: API endpoint URL
        model_params: Model generation parameters
        output_dir: Directory to save generated critiques
        
    Returns:
        Generated critique text
    """
    # Extract relevant knowledge if provided
    knowledge_context = ""
    if knowledge_base:
        # Simple knowledge extraction (can be enhanced)
        contexts = []
        for category in knowledge_base.get('categories', []):
            if 'content' in category:
                contexts.append(category['content'][:200])  # Limit context length
        knowledge_context = "\n".join(contexts)
    
    # Construct prompt
    full_prompt = construct_prompt(persona_text, knowledge_context)
    
    # Generate critique
    critique_text = call_mllm_api(
        image_path=image_path,
        prompt_text=full_prompt,
        model_name=model_name,
        api_endpoint=api_endpoint,
        model_params=model_params
    )
    
    # Save output if directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_file = os.path.join(
            output_dir,
            f"{base_name}_{timestamp}.txt"
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(critique_text)
        print(f"✓ Critique saved to: {output_file}")
    
    return critique_text


def main():
    """Command-line interface."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate art critiques using MLLMs")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name")
    parser.add_argument("--persona", help="Path to persona file")
    parser.add_argument("--api", default="http://localhost:8000/v1/chat/completions", help="API endpoint")
    parser.add_argument("--output", default="outputs/critiques", help="Output directory")
    
    args = parser.parse_args()
    
    # Load persona if specified
    persona_text = ""
    if args.persona and os.path.exists(args.persona):
        with open(args.persona, 'r', encoding='utf-8') as f:
            persona_text = f.read()
    
    # Generate critique
    critique = generate_critique(
        image_path=args.image,
        model_name=args.model,
        persona_text=persona_text,
        api_endpoint=args.api,
        output_dir=args.output
    )
    
    print(f"\nGenerated critique:\n{critique[:500]}...")


if __name__ == "__main__":
    main()