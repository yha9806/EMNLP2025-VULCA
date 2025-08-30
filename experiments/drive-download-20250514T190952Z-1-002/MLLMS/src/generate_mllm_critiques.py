import argparse
import os
import json
from datetime import datetime
import requests # Added for API calls
import base64 # Added for image encoding

# HUGGING_FACE_API_TOKEN is no longer needed for vLLM local server
# Use environment variables if needed: os.getenv("HF_TOKEN")

def call_mllm_api(image_path, prompt_text, model_name, model_params=None):
    """
    Calls the specified MLLM, now assumed to be served locally by vLLM
    using an OpenAI-compatible API.

    Args:
        image_path (str): Path to the input image.
        prompt_text (str): The text prompt for the MLLM.
        model_name (str): Identifier for the MLLM (used in vLLM payload).
        model_params (dict, optional): Additional parameters for the model generation.

    Returns:
        str: The generated text critique or an error message.
    """
    # vLLM OpenAI-compatible endpoint
    api_url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
        # No Authorization header needed for local vLLM typically
    }

    # Encode image to base64
    encoded_image_string = ""
    try:
        with open(image_path, "rb") as image_file:
            encoded_image_string = base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return f"[ERROR: Image file not found: {image_path}]"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return f"[ERROR: Failed to encode image: {e}]"

    # Determine image MIME type
    image_mime_type = "image/jpeg" # Default
    if image_path.lower().endswith(".png"):
        image_mime_type = "image/png"
    elif image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg"):
        image_mime_type = "image/jpeg"
    elif image_path.lower().endswith(".webp"):
        image_mime_type = "image/webp"
    # Add more types or a more robust way to determine mime type if needed

    image_content_part = {
        "type": "image_url",
        "image_url": {
            "url": f"data:{image_mime_type};base64,{encoded_image_string}"
        }
    }

    # Construct OpenAI-compatible payload
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    image_content_part
                ]
            }
        ]
    }

    # Map model_params to OpenAI-compatible parameters
    if model_params:
        if 'max_new_tokens' in model_params:
            payload['max_tokens'] = model_params['max_new_tokens']
        if 'temperature' in model_params:
            payload['temperature'] = model_params['temperature']
        # Add other parameter mappings here if needed (e.g., top_p, stop)
        # Note: 'do_sample' is implicitly handled by temperature in OpenAI API.
        # If temperature is low (e.g., 0), it's deterministic. If higher, it samples.

    print(f"--- CALLING vLLM LOCAL SERVER ({model_name}) ---")
    print(f"API URL: {api_url}")
    prompt_preview = prompt_text[:100].replace('\n', ' ')
    print(f"Prompt (first 100 chars): {prompt_preview}...")
    # Print relevant parts of payload for debugging, excluding full base64 image
    print(f"Payload model: {payload['model']}")
    print(f"Payload messages (text part): {prompt_text[:100]}...")
    print(f"Payload image type: {image_mime_type}")
    if 'max_tokens' in payload: print(f"Payload max_tokens: {payload['max_tokens']}")
    if 'temperature' in payload: print(f"Payload temperature: {payload['temperature']}")

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=180) 
        response.raise_for_status()  
        
        result = response.json()
        generated_critique = ""

        if 'choices' in result and result['choices'] and \
           isinstance(result['choices'][0], dict) and \
           'message' in result['choices'][0] and \
           isinstance(result['choices'][0]['message'], dict) and \
           'content' in result['choices'][0]['message']:
            generated_critique = result['choices'][0]['message']['content']
        elif 'error' in result: # Handle explicit API error messages from vLLM
            error_message = result.get('error', 'Unknown API error')
            if isinstance(error_message, dict) and 'message' in error_message:
                error_message = error_message['message']
            print(f"vLLM API Error: {error_message}")
            return f"[API ERROR from {model_name} via vLLM: {error_message}]"
        else:
            print(f"Error: Unexpected API response format from vLLM. Full response: {str(result)[:500]}")
            return f"[ERROR: Unexpected API response format from vLLM. Response: {str(result)[:200]}]"
        
        response_preview = generated_critique[:100].replace('\n', ' ')
        print(f"--- vLLM API RESPONSE RECEIVED (first 100 chars): {response_preview}... ---")
        return generated_critique.strip()

    except requests.exceptions.Timeout:
        print(f"Error: Request to {api_url} timed out after 180 seconds.")
        return f"[ERROR: API request timed out for {model_name} via vLLM]"
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection Error: Failed to connect to vLLM server at {api_url}. Ensure vLLM is running. Details: {conn_err}")
        return f"[ERROR: Connection failed for vLLM server at {api_url}. Details: {conn_err}]"
    except requests.exceptions.HTTPError as http_err:
        error_message_detail = str(http_err)
        try:
            error_content = http_err.response.json()
            error_message = error_content.get('error', str(http_err))
            if isinstance(error_message, dict) and 'message' in error_message: 
                error_message_detail = error_message['message']
            elif isinstance(error_message, str):
                 error_message_detail = error_message
        except json.JSONDecodeError:
            pass
        print(f"HTTP Error during vLLM API call: {error_message_detail}")
        return f"[ERROR: HTTP Error for {model_name} via vLLM. Details: {error_message_detail}]"
    except Exception as e:
        import traceback
        print(f"Error during vLLM API call: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return f"[ERROR: API call failed for {model_name} via vLLM. Details: {e}]"

def main():
    parser = argparse.ArgumentParser(description="Generate art critiques using MLLMs.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model_name", type=str, required=True, help="Name or identifier of the MLLM to use (e.g., 'Qwen/Qwen3-235B-A22B').")
    parser.add_argument("--prompt_file", type=str, help="Path to a .txt file containing the base prompt. If not provided, a default prompt is used.")
    parser.add_argument("--num_critiques", type=int, default=1, help="Number of critiques to generate for the image.")
    parser.add_argument("--output_dir", type=str, default="experiment/MLLMS/result", help="Directory to save the generated critiques.")
    parser.add_argument("--model_params_json", type=str, help="JSON string or path to a JSON file containing model-specific parameters.")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompt
    prompt_text = ""
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
        except FileNotFoundError:
            print(f"Error: Prompt file {args.prompt_file} not found. Using default prompt if available or exiting.")
            # Fallback to a very basic default or handle error
            prompt_text = f"请对提供的图像 '{os.path.basename(args.image_path)}' 生成一段描述性评论。"

    else: # Default prompt if no file is provided (can be expanded)
        prompt_text = f"""你是一位专业的艺术评论家。请仔细观察提供的图像《{os.path.basename(args.image_path)}》。然后，请撰写一段约300-500字的艺术评论。你的评论应包括但不限于以下几个方面：
1.  **整体印象与主题内容**：画作描绘的场景、传达的氛围。
2.  **构图布局**：画面元素的组织方式，视觉引导。
3.  **笔墨技巧与色彩运用**：线条、墨色、设色的特点及其效果。
4.  **细节描绘**：画作中值得注意的细节及其意义。
5.  **文化内涵**：画作如何反映了当时的社会习俗、节庆文化或审美趣味。

请确保评论语言流畅，观点明确，分析具有一定深度。
"""
        print("No prompt file provided. Using a default detailed prompt.")


    # Load model parameters if provided
    model_params = None
    if args.model_params_json:
        try:
            if os.path.exists(args.model_params_json):
                with open(args.model_params_json, 'r', encoding='utf-8') as f:
                    model_params = json.load(f)
            else:
                model_params = json.loads(args.model_params_json)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON string or file for model_params_json: {args.model_params_json}")
            return
        except FileNotFoundError:
            print(f"Error: Model params JSON file not found: {args.model_params_json}")
            return


    print(f"Starting critique generation for image: {args.image_path}")
    print(f"Model: {args.model_name}")
    print(f"Number of critiques: {args.num_critiques}")
    print(f"Output directory: {args.output_dir}")

    base_image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    sanitized_model_name = args.model_name.replace('/', '_') # Sanitize model name for filename

    for i in range(args.num_critiques):
        print(f"Generating critique {i+1}/{args.num_critiques}...")
        
        # In a real scenario, the image data itself might be passed or loaded within call_mllm_api
        critique_text = call_mllm_api(args.image_path, prompt_text, args.model_name, model_params)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_image_name}_{sanitized_model_name}_critique_{i+1}_{timestamp}.txt"
        output_filepath = os.path.join(args.output_dir, output_filename)
        
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(critique_text)
            print(f"Saved critique to: {output_filepath}")
        except IOError as e:
            print(f"Error saving critique to {output_filepath}: {e}")

    print("Critique generation process complete.")

if __name__ == "__main__":
    main() 