'''
脚本功能：从 PaddleOCR 输出的文本文件中提取纯文本内容，
去除每行的坐标 (BBox) 和置信度 (Confidence) 等元数据。
'''
import re
import os

def extract_text_from_ocr(input_file_path, output_file_path):
    '''
    读取OCR输出文件，提取纯文本并保存到输出文件。

    参数:
        input_file_path (str): 原始OCR文本文件的路径。
        output_file_path (str): 处理后纯文本的保存路径。
    '''
    extracted_lines = []
    # 正则表达式匹配包含 "Line (BBox: ...): \"text\" (Confidence: ...)" 格式的行
    # 并捕获双引号内的文本内容。
    line_regex = re.compile(r'^\s*Line \(BBox: .*?\): "(.*?)" \(Confidence: .*?\)$')

    print(f"开始处理文件: {input_file_path}")

    try:
        # 使用绝对路径读取输入文件
        with open(os.path.abspath(input_file_path), 'r', encoding='utf-8') as infile:
            for line in infile:
                match = line_regex.match(line.strip()) # 使用 strip() 移除行尾换行符和首尾空白
                if match:
                    extracted_text = match.group(1)
                    extracted_lines.append(extracted_text)

        # 确保输出目录存在 (使用绝对路径)
        abs_output_file_path = os.path.abspath(output_file_path)
        output_dir = os.path.dirname(abs_output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")

        with open(abs_output_file_path, 'w', encoding='utf-8') as outfile:
            outfile.write('\n'.join(extracted_lines))
        
        print(f"成功提取文本并保存到: {abs_output_file_path}")
        if not extracted_lines:
            print("警告: 未从输入文件中提取到任何文本行。请检查文件格式和内容。")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到 - {os.path.abspath(input_file_path)}")
    except IOError as e:
        print(f"错误: 发生文件读写错误 - {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    # 用户提供的绝对路径
    # 注意：在Windows上，原始字符串路径中的反斜杠 '\' 需要转义 '\\' 或者使用原始字符串 r"..."
    
    user_output_dir = r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\result"

    files_to_process = [
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\《清代宫廷绘画》_聂崇正_paddleocr.txt", # 保留原始文件以供完整处理，如果需要的话
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\海外中国画研究文选.txt",
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\山川悠远 中国山水画艺术=Symbols Eternity The Art Landscape Painting in China.txt",
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\石涛 清初中国的绘画与现代性.txt",
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\元明清绘画研究十论.txt",
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\中国绘画断代史 8 清代绘画.txt",
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\中国美术史：清代卷（上册）.txt",
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\中国美术史：清代卷（下册）.txt",
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\中国山水画史.txt",
    ]

    # 移除之前处理过的单个文件，或者根据用户意图选择是否重新处理所有文件
    # 根据用户说"把剩下的文本也处理掉"，我们假定聂崇正那个已经处理过了，这里列表可以只包含新的
    # 但为了脚本的通用性，这里我将包含所有文件，如果用户只想处理新的，可以注释掉第一个
    # 或者，更精确地，只处理用户新提供的列表：
    
    new_files_to_process = [
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\海外中国画研究文选.txt",
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\山川悠远 中国山水画艺术=Symbols Eternity The Art Landscape Painting in China.txt",
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\石涛 清初中国的绘画与现代性.txt",
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\元明清绘画研究十论.txt",
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\中国绘画断代史 8 清代绘画.txt",
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\中国美术史：清代卷（上册）.txt",
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\中国美术史：清代卷（下册）.txt",
        r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\中国山水画史.txt",
    ]

    # 首先处理用户之前指定的单个文件（如果需要，或者可以从下面的循环中移除）
    # original_file = r"I:\deeplearning\text collection\v1_lang_shining_project\data\text\processing\清代宫廷绘画\《清代宫廷绘画》_聂崇正_paddleocr.txt"
    # input_filename = os.path.basename(original_file)
    # output_filename = input_filename
    # final_output_path = os.path.join(user_output_dir, output_filename)
    # print(f"--- Processing original file: {original_file} ---")
    # extract_text_from_ocr(original_file, final_output_path)
    # print(f"--- Finished processing: {original_file} ---\n")


    print("--- Starting batch processing of new files ---")
    for user_input_file in new_files_to_process:
        input_filename = os.path.basename(user_input_file)
        # 按用户要求，输出文件名与输入文件名相同
        output_filename = input_filename 
        final_output_path = os.path.join(user_output_dir, output_filename)
        
        print(f"--- Processing file: {user_input_file} ---")
        extract_text_from_ocr(user_input_file, final_output_path)
        print(f"--- Finished processing: {user_input_file} ---\n")
    
    print("--- Batch processing finished ---") 