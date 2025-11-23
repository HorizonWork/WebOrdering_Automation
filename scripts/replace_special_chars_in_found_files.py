import os
import sys

# Thêm thư mục gốc của dự án vào Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.utils.special_char_processor import replace_special_characters, replace_emojis_with_text

def process_specific_files(file_list, mode='replace'):
    """
    Xử lý danh sách các file cụ thể
    
    Args:
        file_list (list): Danh sách các file cần xử lý
        mode (str): Chế độ xử lý - 'replace' để thay thế, 'remove' để xóa
    """
    files_processed = 0
    
    for file_path in file_list:
        try:
            # Kiểm tra file tồn tại
            if not os.path.exists(file_path):
                print(f"File không tồn tại: {file_path}")
                continue
            
            # Đọc nội dung file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Xử lý nội dung
            if mode == 'replace_emojis':
                processed_content = replace_emojis_with_text(original_content)
            else:
                processed_content = replace_special_characters(original_content, mode=mode)
            
            # Ghi lại file nếu có thay đổi
            if original_content != processed_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(processed_content)
                print(f"✅ Đã xử lý file: {file_path}")
                files_processed += 1
            else:
                print(f"⏭️  Bỏ qua (không có thay đổi): {file_path}")
        
        except UnicodeDecodeError:
            print(f"⏭️  Bỏ qua (file binary): {file_path}")
        except Exception as e:
            print(f"❌ Lỗi khi xử lý file {file_path}: {str(e)}")
    
    print(f"\nTổng cộng: {files_processed} file đã được xử lý")

if __name__ == "__main__":
    # Danh sách các file chứa ký tự đặc biệt đã được phát hiện
    files_with_special_chars = [
        '.\\scripts\\check_special_chars.py',
        '.\\src\\planning\\sub_agents\\base_agent.py',
        '.\\src\\utils\\special_char_processor.py',
        '.\\agent_orchestrator_backup.txt',
        '.\\00-START-HERE.md',
        '.\\CODEBASE-OVERVIEW.md',
        '.\\COMPLETE-DOCUMENTATION.md',
        '.\\PLAN.md',
        '.\\README.md',
        '.\\RUN-INSTRUCTIONS.md',
        '.\\SETUP.md',
        '.\\THEORY.md',
        '.\\docs\\CHROME_PROFILE_GUIDE.md',
        '.\\docs\\TEST_EXECUTION_GUIDE.md'
    ]
    
    mode = sys.argv[1] if len(sys.argv) > 1 else 'replace'
    print(f"Đang xử lý các file chứa ký tự đặc biệt với chế độ: {mode}")
    print("="*60)
    
    process_specific_files(files_with_special_chars, mode=mode)