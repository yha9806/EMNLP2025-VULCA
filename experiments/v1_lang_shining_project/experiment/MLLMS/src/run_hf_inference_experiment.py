import os
import json
import time
import base64
import logging
from PIL import Image
from tqdm import tqdm
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

# Phase 1: Project Setup & Configuration
# =======================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler("run_hf_inference_experiment.log"),
        logging.StreamHandler()
    ]
)

# Define Constants - Paths
PROJECT_ROOT = "I:/deeplearning/text collection/v1_lang_shining_project/" # Ensure correct slashes for your OS
IMAGES_DIR = os.path.join(PROJECT_ROOT, "experiment/MLLMS/images")
PERSONA_DIR = os.path.join(PROJECT_ROOT, "experiment/MLLMS/persona")
PROMPT_FILE_PATH = os.path.join(PROJECT_ROOT, "experiment/MLLMS/prompt/prompt.md") # Renamed for clarity
KNOWLEDGE_BASE_FILE_PATH = os.path.join(PROJECT_ROOT, "experiment/MLLMS/knowledge_dataset/knowledge_base.json") # Renamed for clarity

# It's good practice to derive the model name for paths from MODEL_ID
MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
MODEL_NAME_FOR_PATH = MODEL_ID.replace("/", "_") # Creates a safe directory name
OUTPUT_DIR_ROOT = os.path.join(PROJECT_ROOT, "experiment/MLLMS/feedbacks", MODEL_NAME_FOR_PATH)

# Define Constants - API
HF_API_TOKEN = os.getenv("HF_TOKEN") # Get token from environment variable
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY_SECONDS = 20 # Increased delay for potentially large model
INFERENCE_TIMEOUT_SECONDS = 300 # Timeout for the API call

# Initialize API Client (do this after token check in main)
client = None

# Phase 2: Helper Functions - Data Loading
# ==========================================

def load_text_file(file_path):
    """Loads text content from a given file path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading text file {file_path}: {e}")
        return None

def load_json_file(file_path):
    """Loads JSON content from a given file path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading JSON file {file_path}: {e}")
        return None

def get_monthly_image_folders():
    """Gets all subdirectories in IMAGES_DIR, assuming each is a 'Monthly Image'."""
    if not os.path.isdir(IMAGES_DIR):
        logging.error(f"Images directory not found: {IMAGES_DIR}")
        return []
    try:
        return [d for d in os.listdir(IMAGES_DIR) if os.path.isdir(os.path.join(IMAGES_DIR, d))]
    except Exception as e:
        logging.error(f"Error listing monthly image folders in {IMAGES_DIR}: {e}")
        return []

def get_image_slices(monthly_image_folder_name):
    """Gets all image files (slices) within a specific 'Monthly Image' folder's 'large' subdirectory."""
    # According to project structure: data/images/proceed/清_清院_十二月令图_一月_正月/large/
    # And experiment structure: experiment/MLLMS/images/清_清院_十二月令图_一月_正月/large/
    # The README says "Upload all large-format slices corresponding to the current 'Monthly Image'"
    # And "images: .../MLLMS/images ... Each image may consist of multiple large-format slices."
    # It implies the slices are directly under .../MLLMS/images/<Image_Name>/ not a further /large/ subdir for the experiment images.
    # Let's adjust to expect slices in .../MLLMS/images/<Image_Name>/large/ as per typical structure.
    # If "large-format slices" means files directly under <Image_Name>, this needs adjustment.
    # Based on `experiment/MLLMS/images/清_清院_十二月令图_一月_正月/large/` this seems correct.

    monthly_image_path = os.path.join(IMAGES_DIR, monthly_image_folder_name, "large") # Adjusted to include "large"
    if not os.path.isdir(monthly_image_path):
        logging.warning(f"Image slices 'large' subdirectory not found or is not a dir: {monthly_image_path}")
        # Fallback: check directly under monthly_image_folder_name if 'large' is not present
        monthly_image_path_fallback = os.path.join(IMAGES_DIR, monthly_image_folder_name)
        if not os.path.isdir(monthly_image_path_fallback):
            logging.error(f"Monthly image folder not found: {monthly_image_path_fallback}")
            return []
        logging.info(f"Trying fallback: checking for images directly under {monthly_image_path_fallback}")
        monthly_image_path = monthly_image_path_fallback


    image_files = []
    try:
        for item in os.listdir(monthly_image_path):
            if os.path.isfile(os.path.join(monthly_image_path, item)):
                # Add basic image file extension check
                if item.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    image_files.append(os.path.join(monthly_image_path, item))
                else:
                    logging.warning(f"Skipping non-image file in slices directory: {item} in {monthly_image_path}")
        if not image_files:
            logging.warning(f"No image files found in {monthly_image_path}")
        return image_files
    except Exception as e:
        logging.error(f"Error listing image slices in {monthly_image_path}: {e}")
        return []


def get_persona_files():
    """Gets all .txt files in PERSONA_DIR, assuming they are persona files."""
    if not os.path.isdir(PERSONA_DIR):
        logging.error(f"Persona directory not found: {PERSONA_DIR}")
        return []
    try:
        return [os.path.join(PERSONA_DIR, f) for f in os.listdir(PERSONA_DIR) if f.endswith(".txt") and os.path.isfile(os.path.join(PERSONA_DIR, f))]
    except Exception as e:
        logging.error(f"Error listing persona files in {PERSONA_DIR}: {e}")
        return []

def prepare_output_path_and_dir(image_name, persona_name_or_basic):
    """
    Constructs the full output file path and ensures its parent directory exists.
    <Image_Name> will be the folder name of the monthly image.
    <Persona_Card_Name> will be the filename of the persona card without .txt.
    Output Structure: .../MLLMS/feedbacks/<Model_Name>/<Image_Name>/<Image_Name>+<Persona_Card_Name>.txt
                      .../MLLMS/feedbacks/<Model_Name>/<Image_Name>/<Image_Name>+(Basic).txt
    """
    # Sanitize persona_name_or_basic for use in filename (e.g., if it was a path)
    safe_persona_name = os.path.splitext(os.path.basename(str(persona_name_or_basic)))[0]

    # Image_name is the sub-directory for outputs for that image
    image_output_dir = os.path.join(OUTPUT_DIR_ROOT, image_name)
    
    try:
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)
            logging.info(f"Created directory: {image_output_dir}")
    except Exception as e:
        logging.error(f"Could not create directory {image_output_dir}: {e}")
        return None

    output_filename = f"{image_name}+{safe_persona_name}.txt"
    full_output_path = os.path.join(image_output_dir, output_filename)
    return full_output_path

def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"Image file not found for base64 encoding: {image_path}")
        return None
    except Exception as e:
        logging.error(f"Error encoding image {image_path} to base64: {e}")
        return None

# Phase 3: Helper Functions - API Interaction
# ===========================================
def call_inference_api(image_paths, text_input):
    """
    使用新版 Hugging Face Hub OpenAI 兼容 API，图片以 base64 data URI 方式传递。
    image_paths: 图片文件路径列表
    text_input: 需要输入的文本（persona+prompt+knowledge）
    返回: 推理结果字符串或错误信息
    """
    global client
    # 构造 messages 格式
    messages = [
        {
            "role": "user",
            "content": []
        }
    ]
    # 添加文本
    messages[0]["content"].append({"type": "text", "text": text_input})
    # 添加图片（base64 data URI）
    for img_path in image_paths:
        img_b64 = encode_image_to_base64(img_path)
        if img_b64:
            ext = os.path.splitext(img_path)[1].lower().replace('.', '')
            mime = f"image/{'jpeg' if ext in ['jpg', 'jpeg'] else ext}"
            data_uri = f"data:{mime};base64,{img_b64}"
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": data_uri}
            })
    # API 调用重试机制
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            completion = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages
            )
            # 解析返回内容
            if hasattr(completion, "choices") and completion.choices:
                msg = completion.choices[0].message
                if hasattr(msg, "content"):
                    return msg.content
                elif isinstance(msg, dict) and "content" in msg:
                    return msg["content"]
                else:
                    return str(msg)
            else:
                return str(completion)
        except Exception as e:
            logging.error(f"API 调用异常: {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY_SECONDS)
    return "[ERROR] API 调用失败，已重试多次。"

# Phase 4: Core Experiment Logic
# ==============================
def run_experiment():
    global client
    print("初始化 Hugging Face InferenceClient ...")
    client = InferenceClient(model=MODEL_ID, token=HF_API_TOKEN)
    print("加载 persona、prompt、knowledge base ...")
    persona_files = get_persona_files()
    prompt_content = load_text_file(PROMPT_FILE_PATH)
    knowledge_content = load_text_file(KNOWLEDGE_BASE_FILE_PATH)
    monthly_folders = get_monthly_image_folders()
    if not monthly_folders:
        print("未找到任何月令图文件夹！")
        return
    total_tasks = len(monthly_folders) * (len(persona_files) + 1)
    pbar = tqdm(total=total_tasks, desc="批量推理进度", ncols=100)
    for image_name in monthly_folders:
        image_slices = get_image_slices(image_name)
        if not image_slices:
            logging.warning(f"未找到图片切片: {image_name}")
            pbar.update(len(persona_files) + 1)
            continue
        # Persona-based Q&A
        for persona_path in persona_files:
            persona_text = load_text_file(persona_path)
            persona_name = os.path.splitext(os.path.basename(persona_path))[0]
            input_text = (persona_text or "") + "\n" + (prompt_content or "") + "\n" + (knowledge_content or "")
            output_path = prepare_output_path_and_dir(image_name, persona_name)
            # 幂等性：已存在且非空则跳过
            if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                pbar.update(1)
                continue
            result = call_inference_api(image_slices, input_text)
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result)
            except Exception as e:
                logging.error(f"写入结果失败: {output_path}: {e}")
            pbar.update(1)
        # Basic Q&A
        input_text = (prompt_content or "") + "\n" + (knowledge_content or "")
        output_path = prepare_output_path_and_dir(image_name, "(Basic)")
        if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            pbar.update(1)
            continue
        result = call_inference_api(image_slices, input_text)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
        except Exception as e:
            logging.error(f"写入结果失败: {output_path}: {e}")
        pbar.update(1)
    pbar.close()
    print("全部批量推理任务完成。")

# Phase 5: Script Entry Point
# ===========================
def main():
    run_experiment()

if __name__ == "__main__":
    main() 