import os
import time
import logging
from tqdm import tqdm
from huggingface_hub import InferenceClient

# ========== 配置部分 ==========
PROJECT_ROOT = "I:/deeplearning/text collection/v1_lang_shining_project/"
IMAGES_DIR = os.path.join(PROJECT_ROOT, "experiment/MLLMS/images")
PERSONA_DIR = os.path.join(PROJECT_ROOT, "experiment/MLLMS/persona")
PROMPT_FILE_PATH = os.path.join(PROJECT_ROOT, "experiment/MLLMS/prompt/prompt.md")
KNOWLEDGE_BASE_FILE_PATH = os.path.join(PROJECT_ROOT, "experiment/MLLMS/knowledge_dataset/knowledge_base.json")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PROVIDER = "nebius"
MODEL_NAME_FOR_PATH = MODEL_ID.replace("/", "_")
OUTPUT_DIR_ROOT = os.path.join(PROJECT_ROOT, "experiment/MLLMS/feedbacks", MODEL_NAME_FOR_PATH)

HF_API_TOKEN = os.getenv("HF_TOKEN")  # Get token from environment variable
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY_SECONDS = 20

# ========== 日志 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler("run_hf_inference_llama3_8b.log"),
        logging.StreamHandler()
    ]
)

# ========== 工具函数 ==========
def load_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"读取文本文件失败: {file_path}: {e}")
        return None

def get_monthly_image_folders():
    if not os.path.isdir(IMAGES_DIR):
        logging.error(f"Images directory not found: {IMAGES_DIR}")
        return []
    return [d for d in os.listdir(IMAGES_DIR) if os.path.isdir(os.path.join(IMAGES_DIR, d))]

def get_persona_files():
    if not os.path.isdir(PERSONA_DIR):
        logging.error(f"Persona directory not found: {PERSONA_DIR}")
        return []
    return [os.path.join(PERSONA_DIR, f) for f in os.listdir(PERSONA_DIR) if f.endswith(".txt") and os.path.isfile(os.path.join(PERSONA_DIR, f))]

def prepare_output_path_and_dir(image_name, persona_name_or_basic):
    safe_persona_name = os.path.splitext(os.path.basename(str(persona_name_or_basic)))[0]
    image_output_dir = os.path.join(OUTPUT_DIR_ROOT, image_name)
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)
    output_filename = f"{image_name}+{safe_persona_name}.txt"
    return os.path.join(image_output_dir, output_filename)

# ========== API 调用 ==========
def call_inference_api(text_input):
    global client
    messages = [
        {
            "role": "user",
            "content": text_input
        }
    ]
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            completion = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages
            )
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

# ========== 主循环 ==========
def run_experiment():
    global client
    client = InferenceClient(provider=PROVIDER, api_key=HF_API_TOKEN)
    persona_files = get_persona_files()
    prompt_content = load_text_file(PROMPT_FILE_PATH)
    knowledge_content = load_text_file(KNOWLEDGE_BASE_FILE_PATH)
    monthly_folders = get_monthly_image_folders()
    if not monthly_folders:
        print("未找到任何月令图文件夹！")
        return
    total_tasks = len(monthly_folders) * (len(persona_files) + 1)
    pbar = tqdm(total=total_tasks, desc="Llama3-8B 批量推理进度", ncols=100)
    for image_name in monthly_folders:
        for persona_path in persona_files:
            persona_text = load_text_file(persona_path)
            persona_name = os.path.splitext(os.path.basename(persona_path))[0]
            input_text = (persona_text or "") + "\n" + (prompt_content or "") + "\n" + (knowledge_content or "")
            output_path = prepare_output_path_and_dir(image_name, persona_name)
            if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                pbar.update(1)
                continue
            result = call_inference_api(input_text)
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result)
            except Exception as e:
                logging.error(f"写入结果失败: {output_path}: {e}")
            pbar.update(1)
        input_text = (prompt_content or "") + "\n" + (knowledge_content or "")
        output_path = prepare_output_path_and_dir(image_name, "(Basic)")
        if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            pbar.update(1)
            continue
        result = call_inference_api(input_text)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
        except Exception as e:
            logging.error(f"写入结果失败: {output_path}: {e}")
        pbar.update(1)
    pbar.close()
    print("全部批量推理任务完成。")

# ========== 入口 ==========
def main():
    run_experiment()

if __name__ == "__main__":
    main() 