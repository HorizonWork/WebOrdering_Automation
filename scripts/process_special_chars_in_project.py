import os
import sys
import glob

# Thêm thư mục gốc của dự án vào Python path
project_root = os.path.dirname(os.path.dirname(__file__))  # Lên một cấp từ scripts
sys.path.insert(0, project_root)

from src.utils.special_char_processor import replace_special_characters, replace_emojis_with_text

def process_files_in_directory(directory_path, file_extensions=None, mode='replace'):
    """
    Xử lý các ký tự đặc biệt trong tất cả các file trong thư mục
    
    Args:
        directory_path (str): Đường dẫn thư mục để quét
        file_extensions (list): Danh sách phần mở rộng file cần xử lý (ví dụ: ['.py', '.txt', '.md'])
        mode (str): Chế độ xử lý - 'replace' để thay thế, 'remove' để xóa
    """
    if file_extensions is None:
        file_extensions = ['.py', '.txt', '.md', '.json', '.yaml', '.yml', '.html', '.css', '.js', '.ts']
    
    # Tìm tất cả các file có phần mở rộng được chỉ định
    files_processed = 0
    for ext in file_extensions:
        pattern = os.path.join(directory_path, '**', f'*{ext}')
        for file_path in glob.glob(pattern, recursive=True):
            # Bỏ qua các thư mục ẩn và thư mục venv
            if any(ignore_dir in file_path for ignore_dir in ['/venv/', '\\venv\\', '/.git/', '\\.git\\', '/__pycache__/', '\\__pycache__\\', '/.pytest_cache/', '\\.pytest_cache\\']):
                continue
            
            try:
                # Đọc nội dung file
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                
                # Xử lý nội dung
                if mode == 'replace_emojis':
                    processed_content = replace_emojis_with_text(original_content)
                else:
                    processed_content = replace_special_characters(original_content, mode=mode)
                
                # Nếu nội dung thay đổi, ghi lại file
                if original_content != processed_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(processed_content)
                    print(f"Đã xử lý file: {file_path}")
                    files_processed += 1
                else:
                    # Bỏ qua in ra khi không có thay đổi để giảm output
                    pass
            
            except UnicodeDecodeError:
                # Bỏ qua các file không đọc được như hình ảnh, binary, v.v.
                continue
            except Exception as e:
                print(f"Lỗi khi xử lý file {file_path}: {str(e)}")
    
    print(f"\nĐã xử lý {files_processed} file trong thư mục {directory_path}")


def process_specific_file(file_path, mode='replace'):
    """
    Xử lý một file cụ thể
    
    Args:
        file_path (str): Đường dẫn đến file cần xử lý
        mode (str): Chế độ xử lý - 'replace' để thay thế, 'remove' để xóa
    """
    try:
        # Kiểm tra file tồn tại
        if not os.path.exists(file_path):
            print(f"File không tồn tại: {file_path}")
            return
        
        # Đọc nội dung file
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Xử lý nội dung
        if mode == 'replace_emojis':
            processed_content = replace_emojis_with_text(original_content)
        else:
            processed_content = replace_special_characters(original_content, mode=mode)
        
        # Nếu nội dung thay đổi, ghi lại file
        if original_content != processed_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            print(f"Đã xử lý file: {file_path}")
        else:
            print(f"Không có thay đổi trong file: {file_path}")
    
    except UnicodeDecodeError:
        print(f"File không thể đọc được (có thể là file binary): {file_path}")
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {str(e)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        target = sys.argv[1]
        mode = sys.argv[2] if len(sys.argv) > 2 else 'replace'
        
        if os.path.isfile(target):
            # Xử lý file cụ thể
            print(f"Đang xử lý file: {target}")
            process_specific_file(target, mode)
        elif os.path.isdir(target):
            # Xử lý thư mục
            print(f"Đang xử lý thư mục: {target}")
            process_files_in_directory(target, mode=mode)
        else:
            print(f"Đường dẫn không tồn tại: {target}")
    else:
        # Xử lý toàn bộ dự án
        project_path = os.path.join(os.path.dirname(__file__), '..')
        print(f"Đang xử lý toàn bộ dự án tại: {project_path}")
        process_files_in_directory(project_path, mode='replace')