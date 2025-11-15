# Chrome Profile Support - Usage Guide

## 🎯 Overview

BrowserManager now supports using your existing Chrome profiles, allowing you to:
- ✅ **Keep login sessions** (Shopee, Lazada, etc.)
- ✅ **Use saved cookies** and localStorage
- ✅ **Access Chrome extensions**
- ✅ **Use autofill credentials**
- ✅ **Keep browsing history and bookmarks**

---

## 📋 Prerequisites

1. **Chrome installed** on your system
2. **Know your profile directory** (see below)
3. **Close Chrome** before running with profile

---

## 🔍 Step 1: Find Your Chrome Profile

### Method 1: Using the script

```powershell
python list_chrome_profiles.py
```

Output example:
```
6. Profile 18
   Display Name: Your Chrome
   Path: C:\Users\Nekloyh\AppData\Local\Google\Chrome\User Data\Profile 18
   Has Cookies: ✓
   Has Bookmarks: ✗
```

### Method 2: Manually in Chrome

1. Open Chrome
2. Type in address bar: `chrome://version/`
3. Find "Profile Path" line:
   ```
   Profile Path: C:\Users\Nekloyh\AppData\Local\Google\Chrome\User Data\Profile 18
   ```
4. The profile directory is: **Profile 18**

---

## ⚙️ Step 2: Configure

### Option A: Using `.env` file (Recommended)

Edit `.env`:

```bash
# Chrome Profile Settings
USE_CHROME_PROFILE=true
CHROME_EXECUTABLE_PATH=C:\Program Files\Google\Chrome\Application\chrome.exe
CHROME_PROFILE_DIRECTORY=Profile 18
```

Then use with settings:

```python
from config.settings import settings
from src.execution.browser_manager import BrowserManager

# Automatically uses config from .env
manager = BrowserManager(**settings.browser_config)
await manager.launch()
```

### Option B: Direct code configuration

```python
from src.execution.browser_manager import BrowserManager

manager = BrowserManager(
    headless=False,
    use_chrome_profile=True,
    chrome_executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    chrome_profile_directory="Profile 18"
)

await manager.launch()
```

---

## 🧪 Step 3: Test

### Quick test with .env config:

```powershell
python test_browser_with_settings.py
```

### Full Chrome profile test:

```powershell
python test_chrome_profile.py
```

### Manual test in your code:

```python
import asyncio
from src.execution.browser_manager import BrowserManager

async def test():
    manager = BrowserManager(
        headless=False,
        use_chrome_profile=True,
        chrome_profile_directory="Profile 18"
    )
    
    await manager.launch()
    page = manager.pages[0] if manager.pages else await manager.new_page()
    
    await page.goto("https://shopee.vn")
    # You should be already logged in if you were logged in before!
    
    await asyncio.sleep(5)
    await manager.close()

asyncio.run(test())
```

---

## ⚠️ Important Notes

### 1. Close Chrome First!

**You MUST close all Chrome windows** before running with profile:

```
❌ Error: Target page, context or browser has been closed
```

**Solution:**
1. Close ALL Chrome windows
2. Run your script again

### 2. Profile is Locked

When using Chrome profile in automation, Chrome locks it. You cannot open regular Chrome with the same profile simultaneously.

### 3. Multiple Profiles

To avoid conflicts, you can:

**Option A:** Create dedicated automation profile:

```powershell
# Create new Chrome profile
chrome.exe --user-data-dir="F:\WebOrdering_Automation\chrome_profile" --profile-directory="Automation"

# Login to your accounts, setup extensions, etc.
# Then close Chrome and use this profile for automation
```

Update `.env`:
```bash
CHROME_PROFILE_DIRECTORY=Automation
```

**Option B:** Use different profile for automation vs personal use

---

## 📂 File Structure

```
WebOrdering_Automation/
├── .env                          # Configuration
├── list_chrome_profiles.py       # Find available profiles
├── test_chrome_profile.py        # Full profile test
├── test_browser_with_settings.py # Quick settings test
│
├── src/execution/
│   └── browser_manager.py        # BrowserManager with profile support
│
└── config/
    └── settings.py               # Settings with browser_config
```

---

## 🔧 Configuration Reference

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `USE_CHROME_PROFILE` | Enable Chrome profile | `true` / `false` |
| `CHROME_EXECUTABLE_PATH` | Chrome executable path | `C:\Program Files\Google\Chrome\Application\chrome.exe` |
| `CHROME_PROFILE_DIRECTORY` | Profile directory name | `Profile 18`, `Default` |
| `HEADLESS` | Run in headless mode | `false` (profile needs headed mode) |

### BrowserManager Parameters

```python
BrowserManager(
    browser_type="chromium",           # Browser type
    headless=False,                    # Must be False for profile
    viewport={"width": 1920, "height": 1080},  # Viewport size
    use_chrome_profile=True,           # Enable profile
    chrome_executable_path="...",      # Chrome path (auto-detect if None)
    chrome_profile_directory="Profile 18"  # Profile name
)
```

---

## 🐛 Troubleshooting

### Error: "Profile directory not found"

```
FileNotFoundError: Profile directory not found: C:\Users\...\Profile 18
Available profiles in C:\Users\...\User Data:
  - Default
  - Profile 1
  - Profile 18
```

**Solution:**
1. Check profile name spelling
2. Run `python list_chrome_profiles.py` to see available profiles
3. Use exact profile directory name (case-sensitive)

### Error: "Chrome executable not found"

**Solution:**
1. Install Chrome if not installed
2. Specify path in `.env`:
   ```bash
   CHROME_EXECUTABLE_PATH=C:\Program Files\Google\Chrome\Application\chrome.exe
   ```

### Error: "Target page has been closed" or "already in use"

**Solution:**
1. **Close ALL Chrome windows**
2. Check Task Manager for `chrome.exe` processes
3. Kill all Chrome processes if needed
4. Run script again

---

## 🎯 Common Use Cases

### 1. E-commerce Testing (Logged In)

```python
# Use profile with Shopee/Lazada already logged in
manager = BrowserManager(
    use_chrome_profile=True,
    chrome_profile_directory="Profile 18"
)

await manager.launch()
page = manager.pages[0]
await page.goto("https://shopee.vn")
# ✅ Already logged in!
```

### 2. Multiple Accounts Testing

```python
# Profile 1 - Account A
manager_a = BrowserManager(
    use_chrome_profile=True,
    chrome_profile_directory="Profile 1"
)

# Profile 2 - Account B  
manager_b = BrowserManager(
    use_chrome_profile=True,
    chrome_profile_directory="Profile 2"
)
```

### 3. Extension Testing

```python
# Use profile with extensions installed
manager = BrowserManager(
    use_chrome_profile=True,
    chrome_profile_directory="Profile 18"
)
# Extensions from profile will be loaded
```

---

## ✅ Best Practices

1. **Create Dedicated Profile** for automation
2. **Keep Regular Chrome Profile** separate for daily use
3. **Always close Chrome** before automation
4. **Use headed mode** (headless doesn't work well with profiles)
5. **Handle errors** gracefully (profile locked, etc.)

---

## 📚 Additional Resources

- [Playwright Documentation](https://playwright.dev/python/docs/api/class-browsertype#browser-type-launch-persistent-context)
- [Chrome Profiles Guide](https://support.google.com/chrome/answer/2364824)
- Project documentation: `COMPLETE-DOCUMENTATION.md`

---

## 🎉 Summary

You can now use Chrome profiles in your automation! This allows you to:

✅ Skip login steps (already logged in)
✅ Use real browser data (cookies, localStorage)
✅ Test with extensions
✅ Simulate real user behavior

**Remember:** Close Chrome before running! 🚀
