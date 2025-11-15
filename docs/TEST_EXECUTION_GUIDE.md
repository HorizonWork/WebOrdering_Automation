# How to Test Execution Layer

## 🎯 Mục Đích

Hướng dẫn này giúp bạn kiểm tra phần **Execution Layer** đã hoạt động tốt chưa.

---

## 📋 Execution Layer Bao Gồm

### 1. **BrowserManager** (`src/execution/browser_manager.py`)
- Quản lý Playwright browser lifecycle
- Launch/close browser
- Tạo và quản lý pages
- Screenshot, navigation
- Hỗ trợ Chrome profiles

### 2. **SkillExecutor** (`src/execution/skill_executor.py`)
- Dispatch actions tới skills
- Quản lý execution flow
- Error handling

### 3. **Skills** (`src/execution/skills/`)
- **NavigationSkills**: goto, back, forward, reload
- **InteractionSkills**: click, type, fill, hover, press
- **ObservationSkills**: screenshot, get_dom, get_text, get_url
- **ValidationSkills**: check_exists, check_visible, check_enabled
- **WaitSkills**: wait_for, wait_for_selector, wait_for_navigation

---

## 🧪 Cách Test

### Test 1: Import Test (Nhanh nhất)

Kiểm tra xem tất cả components có import được không:

```powershell
F:\WebOrdering_Automation\woa\python.exe -c "
from src.execution.browser_manager import BrowserManager
from src.execution.skill_executor import SkillExecutor
from src.execution.skills import NavigationSkills, InteractionSkills
print('✅ All imports successful!')
"
```

**Kết quả mong đợi:** 
```
✅ All imports successful!
```

---

### Test 2: BrowserManager Test

Kiểm tra BrowserManager cơ bản (KHÔNG dùng Chrome profile):

```powershell
F:\WebOrdering_Automation\woa\python.exe -c "
import asyncio
from src.execution.browser_manager import BrowserManager

async def test():
    manager = BrowserManager(headless=False, use_chrome_profile=False)
    await manager.launch()
    page = await manager.new_page()
    await page.goto('https://example.com')
    print(f'✅ URL: {page.url}')
    print(f'✅ Title: {await page.title()}')
    await manager.close()

asyncio.run(test())
"
```

**Kết quả mong đợi:**
```
✅ URL: https://example.com/
✅ Title: Example Domain
```

---

### Test 3: SkillExecutor Test

Kiểm tra SkillExecutor với các skills cơ bản:

```powershell
F:\WebOrdering_Automation\woa\python.exe -c "
import asyncio
from src.execution.browser_manager import BrowserManager
from src.execution.skill_executor import SkillExecutor

async def test():
    manager = BrowserManager(headless=False, use_chrome_profile=False)
    executor = SkillExecutor()
    
    await manager.launch()
    page = await manager.new_page()
    
    # Test goto skill
    result = await executor.execute(page, {
        'skill': 'goto',
        'params': {'url': 'https://google.com'}
    })
    print(f'✅ goto: {result[\"status\"]}')
    
    # Test get_title skill
    result = await executor.execute(page, {
        'skill': 'get_title',
        'params': {}
    })
    print(f'✅ get_title: {result[\"data\"]}')
    
    await manager.close()

asyncio.run(test())
"
```

**Kết quả mong đợi:**
```
✅ goto: success
✅ get_title: Google
```

---

### Test 4: Unit Tests (Đầy đủ)

Chạy unit tests có sẵn:

```powershell
# Test execution cơ bản
F:\WebOrdering_Automation\woa\python.exe tests/unit/test_execution.py
```

**Nếu muốn test kỹ hơn:**

```powershell
# Test step-by-step (từng bước)
F:\WebOrdering_Automation\woa\python.exe tests/test_execution_stepbystep.py

# Test quick (nhanh)
F:\WebOrdering_Automation\woa\python.exe tests/test_execution_quick.py

# Test suite đầy đủ (lâu nhất, test nhiều nhất)
F:\WebOrdering_Automation\woa\python.exe tests/unit/test_execution_suite.py
```

---

### Test 5: Chrome Profile Test (Tùy chọn)

**⚠️ LƯU Ý: PHẢI đóng tất cả cửa sổ Chrome trước khi chạy!**

```powershell
# Test với Chrome profile
F:\WebOrdering_Automation\woa\python.exe tests/test_chrome_profile.py
```

---

## ✅ Checklist Kiểm Tra

Đánh dấu ✅ khi test thành công:

### BrowserManager
- [ ] Import BrowserManager thành công
- [ ] Launch browser (standard Chromium)
- [ ] Create page
- [ ] Navigate to URL
- [ ] Get page title
- [ ] Take screenshot
- [ ] Close browser
- [ ] Launch with Chrome profile (optional)

### SkillExecutor
- [ ] Import SkillExecutor thành công
- [ ] Execute goto skill
- [ ] Execute get_title skill
- [ ] Execute get_url skill
- [ ] Execute wait_for_selector skill
- [ ] Error handling works

### Skills
#### NavigationSkills
- [ ] goto
- [ ] back
- [ ] forward
- [ ] reload

#### InteractionSkills
- [ ] click
- [ ] type
- [ ] fill
- [ ] hover (optional)
- [ ] press (optional)

#### ObservationSkills
- [ ] get_url
- [ ] get_title
- [ ] get_text
- [ ] screenshot

#### WaitSkills
- [ ] wait_for_selector
- [ ] wait_for (optional)

---

## 🐛 Troubleshooting

### Lỗi: "ModuleNotFoundError"

```
ModuleNotFoundError: No module named 'src.execution'
```

**Giải pháp:**
1. Chạy từ root directory của project
2. Hoặc set PYTHONPATH:
   ```powershell
   $env:PYTHONPATH="F:\WebOrdering_Automation"
   ```

### Lỗi: "playwright._impl._errors.TargetClosedError"

```
Target page, context or browser has been closed
```

**Giải pháp:**
- Đóng tất cả cửa sổ Chrome nếu đang test với Chrome profile
- Hoặc dùng standard browser: `use_chrome_profile=False`

### Lỗi: "Timeout waiting for selector"

```
TimeoutError: Timeout 30000ms exceeded
```

**Giải pháp:**
- Selector không đúng hoặc element chưa load
- Tăng timeout: `timeout=60000`
- Kiểm tra selector bằng browser DevTools (F12)

---

## 📊 Kết Quả Mong Đợi

Nếu **EXECUTION LAYER HOẠT ĐỘNG TỐT**, bạn sẽ thấy:

### ✅ Import Test
```
✅ All imports successful!
```

### ✅ BrowserManager Test
```
✅ Browser launched
✅ Page created
✅ URL: https://example.com/
✅ Title: Example Domain
✅ Browser closed
```

### ✅ SkillExecutor Test
```
✅ goto: success
✅ get_title: Google
✅ type: success
✅ click: success
```

### ✅ Full Test Suite
```
Total Tests: 10
✅ Passed: 10
❌ Failed: 0
Success Rate: 100%
```

---

## 🎯 Quick Start - Test Ngay

**3 lệnh test nhanh nhất:**

```powershell
# 1. Import test (5 giây)
F:\WebOrdering_Automation\woa\python.exe -c "from src.execution.browser_manager import BrowserManager; from src.execution.skill_executor import SkillExecutor; print('✅ OK')"

# 2. Browser test (15 giây)
F:\WebOrdering_Automation\woa\python.exe -c "import asyncio; from src.execution.browser_manager import BrowserManager; asyncio.run((lambda: __import__('asyncio').create_task(test()))()) async def test(): m = BrowserManager(headless=False, use_chrome_profile=False); await m.launch(); p = await m.new_page(); await p.goto('https://example.com'); print('✅', p.url); await m.close()"

# 3. Executor test (30 giây)
F:\WebOrdering_Automation\woa\python.exe tests/unit/test_execution.py
```

---

## 💡 Tips

1. **Luôn test với `use_chrome_profile=False`** để tránh conflict với Chrome đang chạy
2. **Chạy test từ terminal trong VS Code** để dễ debug
3. **Xem log** trong `logs/` folder nếu có lỗi
4. **Take screenshot** khi debug: `await page.screenshot(path='debug.png')`
5. **In ra HTML** khi cần: `print(await page.content())`

---

## 🚀 Next Steps

Sau khi execution layer hoạt động tốt:

1. ✅ Test execution layer
2. Test planning layer (`src/planning/`)
3. Test perception layer (`src/perception/`)
4. Test learning layer (`src/learning/`)
5. Test full orchestrator (`src/orchestrator/`)

---

## 📝 Summary

**Execution Layer bao gồm:**
- BrowserManager (browser lifecycle)
- SkillExecutor (dispatch actions)
- Skills (implement actions)

**Test nhanh nhất:**
```powershell
python tests/unit/test_execution.py
```

**Test đầy đủ nhất:**
```powershell
python tests/unit/test_execution_suite.py
```

**Nếu tất cả test PASS → Execution layer đã hoạt động tốt!** ✅
