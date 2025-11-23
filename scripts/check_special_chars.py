import os
import glob
import re

def check_special_characters_in_project(directory_path, file_extensions=None):
    """
    Kiểm tra các file trong dự án có chứa ký tự đặc biệt cần xử lý
    
    Args:
        directory_path (str): Đường dẫn thư mục để quét
        file_extensions (list): Danh sách phần mở rộng file cần kiểm tra
    """
    if file_extensions is None:
        file_extensions = ['.py', '.txt', '.md', '.json', '.yaml', '.yml', '.html', '.css', '.js', '.ts']
    
    # Regex pattern để tìm các ký tự đặc biệt phổ biến
    special_chars_pattern = r'[✅❌✔✗☀☁☂☃⚡❤🔥🌟💯🎉👏🙌👍👎👌🙏👀🐶🐱🐭🐹🐰🦊🐻🐼🐨🦁🐯🐮🐷🐸🐵🐔🐧🐦🦆🦅🦉🦇🐺🐗🐴🦄🐝🐛🦋🐌🐞🐜🦟🦗🕷🦂🐢🐍🦎🦖🦕🐙🦑🦐🦞🦀🐡🐠🐟🐬🐳🐋🦈🐊🐅🐆🦓🦍🐘🦏🦛🐪🐫🦒🦘🐃🐂🐄🐎🐖🐏🐑🦙🐐🦌🐕🐩🦮🐕‍🦺🐈🐓🦃🦚🦜🦢🦩🕊🐇🦝🦨🦡🦦🦥🐁🐀🐿🦔😊😌😍😏😒😞😔😟😕🙁☹😮😯😲😳🥺😦😧😨😰😥😢😭😱😖😣😓😩😫🥱😤😡😠🤬😈👿💀☠💩🤡👹👺👻👽👾🤖😺😸😹😻😼😽🙀😿😾🙈🙉🙊]'
    
    files_with_special_chars = []
    
    for ext in file_extensions:
        pattern = os.path.join(directory_path, '**', f'*{ext}')
        for file_path in glob.glob(pattern, recursive=True):
            # Bỏ qua các thư mục ẩn và thư mục venv
            if any(ignore_dir in file_path for ignore_dir in ['/venv/', '\\venv\\', '/.git/', '\\.git\\', '/__pycache__/', '\\__pycache__\\', '/.pytest_cache/', '\\.pytest_cache\\']):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                matches = re.findall(special_chars_pattern, content)
                if matches:
                    unique_matches = list(set(matches))
                    files_with_special_chars.append((file_path, unique_matches, len(matches)))
                    print(f"File: {file_path}")
                    print(f"  Có {len(matches)} ký tự đặc biệt: {unique_matches[:10]}{'...' if len(unique_matches) > 10 else ''}")
                    print()
            
            except UnicodeDecodeError:
                # Bỏ qua các file không đọc được
                continue
            except Exception as e:
                print(f"Lỗi khi đọc file {file_path}: {str(e)}")
    
    print(f"\nTổng cộng: {len(files_with_special_chars)} file chứa ký tự đặc biệt")
    return files_with_special_chars


if __name__ == "__main__":
    import sys
    directory_path = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Đang kiểm tra các ký tự đặc biệt trong: {directory_path}")
    print("="*60)
    
    results = check_special_characters_in_project(directory_path)
    
    if not results:
        print("\nKhông tìm thấy file nào chứa ký tự đặc biệt cần xử lý.")
    else:
        print("\nDanh sách các file chứa ký tự đặc biệt:")
        for file_path, unique_matches, total_count in results:
            print(f"- {file_path}: {total_count} ký tự, {len(unique_matches)} loại")