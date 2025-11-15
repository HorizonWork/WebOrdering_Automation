import json
import time
from playwright.sync_api import sync_playwright, Playwright, expect

# Tên file log sự kiện
EVENT_LOG_FILE = 'data\\flow\\flowWithoutImage\\event_log_500_v4.json'

def run_test_from_log(playwright: Playwright):
    """
    Chạy kịch bản kiểm thử Playwright dựa trên file log sự kiện JSON.
    """
    # --- 1. Đọc và phân tích file JSON ---
    try:
        with open(EVENT_LOG_FILE, 'r', encoding='utf-8') as f:
            events = json.load(f)
        print(f"Đã đọc thành công {len(events)} sự kiện từ '{EVENT_LOG_FILE}'.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{EVENT_LOG_FILE}'. Vui lòng đảm bảo file này nằm cùng thư mục.")
        return
    except json.JSONDecodeError:
        print(f"Lỗi: File '{EVENT_LOG_FILE}' không phải là một file JSON hợp lệ.")
        return

    # --- 2. Khởi tạo trình duyệt ---
    # Chạy ở chế độ có giao diện (headed) để dễ dàng quan sát
    browser = playwright.chromium.launch(headless=False, slow_mo=500) 
    context = browser.new_context()
    page = context.new_page()

    print("\nBắt đầu tái hiện các hành động của người dùng...")

    # --- 3. Lặp qua và thực thi các sự kiện ---
    for i, event in enumerate(events):
        event_type = event.get('event_type')
        
        # Bỏ qua các sự kiện rỗng hoặc không có event_type
        if not event_type:
            continue

        selector = event.get('selector')
        
        print(f"Bước {i+1}: Thực thi sự kiện '{event_type}'...")

        try:
            if event_type == 'navigate':
                url = event.get('url')
                if url:
                    print(f"  -> Điều hướng đến: {url}")
                    page.goto(url)
                else:
                    print("  -> Bỏ qua: Sự kiện 'navigate' thiếu URL.")

            elif event_type == 'click':
                if selector:
                    print(f"  -> Nhấp vào selector: '{selector}'")
                    page.locator(selector).click()
                else:
                    print("  -> Bỏ qua: Sự kiện 'click' thiếu selector.")

            elif event_type == 'change':
                value = event.get('value', '') # Mặc định là chuỗi rỗng nếu không có giá trị
                if selector:
                    print(f"  -> Điền giá trị '{value}' vào selector: '{selector}'")
                    page.locator(selector).fill(value)
                else:
                    print("  -> Bỏ qua: Sự kiện 'change' thiếu selector.")
            
            # Các loại sự kiện khác như 'submit' có thể được bỏ qua
            # vì chúng thường được kích hoạt bởi một sự kiện 'click' trước đó.
            else:
                print(f"  -> Bỏ qua loại sự kiện không xác định: '{event_type}'")

        except Exception as e:
            print(f"!!!!!! Lỗi ở bước {i+1} khi thực thi sự kiện {event} !!!!!!!")
            print(f"Lỗi chi tiết: {e}")
            # Chụp ảnh màn hình khi có lỗi để debug
            screenshot_path = f"error_step_{i+1}.png"
            page.screenshot(path=screenshot_path)
            print(f"Đã chụp ảnh màn hình lỗi tại: '{screenshot_path}'")
            break # Dừng kịch bản khi có lỗi

    # --- 4. Hoàn tất và dọn dẹp ---
    print("\nHoàn thành tất cả các hành động!")
    print("Trình duyệt sẽ đóng sau 5 giây...")
    time.sleep(5) # Giữ trình duyệt mở để xem kết quả

    context.close()
    browser.close()

def main():
    with sync_playwright() as playwright:
        run_test_from_log(playwright)

if __name__ == "__main__":
    main()