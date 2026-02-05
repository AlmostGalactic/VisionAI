# 👁️ Vision Agent AI

A robotic AI agent that can "see" your screen and control your mouse/keyboard to perform tasks. 
It uses **OpenAI GPT-4o** for reasoning, **YOLO-World** for visual object detection, and **EasyOCR** for reading text on the screen.

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

## 🚀 Features

* **Natural Language Control:** Tell it "Open Spotify and play jazz" or "Find the email from John and reply."
* **Visual Recognition:** Uses computer vision to find icons and buttons, not just coordinate guessing.
* **Universal Compatibility:** Designed to run on standard CPUs (laptops) but supports GPU acceleration if available.
* **Optimized Speed:** Uses smart image compression to reduce API latency.

---

## 🛠️ Prerequisites

1.  **Python 3.10 or higher** installed on your system.
2.  **An OpenAI API Key** with access to GPT-4o.

---

## 📦 Installation

### 1. Clone or Download the Repository
```bash
git clone [https://github.com/yourusername/vision-agent-ai.git](https://github.com/yourusername/vision-agent-ai.git)
cd vision-agent-ai

```

### 2. Set Up Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install PyTorch (Critical Step)

This project uses PyTorch for computer vision. You must install the version that matches your hardware.

**Option A: Standard (Safe for everyone, no GPU required)**
*Use this if you are on a standard laptop or don't know what GPU you have.*

```bash
pip install torch torchvision torchaudio

```

### 4. Install Dependencies

Once PyTorch is installed, install the rest of the libraries:

```bash
pip install -r requirements.txt

```

---

## 🔑 Configuration

1. Create a file named `.env` in the main folder.
2. Add your OpenAI API key inside it:

```env
OPENAI_API_KEY="sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx"

```

---

## ▶️ Usage

1. Run the application:
```bash
python agent.py

```


2. A GUI window will appear.
3. **Type a task** into the input box (e.g., *"Open Calculator and calculate 55 * 10"*).
4. Click **START AGENT**.
5. **Hands off!** The mouse will start moving automatically.

**To Stop:**
Click the **STOP** button in the GUI or slam your mouse into the corner of the screen (Failsafe trigger).

---

## 📝 Troubleshooting

**"It says 'No matching distribution found for torch'"**
You are likely using a very new or very old version of Python. Ensure you are using Python 3.10, 3.11, or 3.12.

**"The mouse is moving too slowly"**
Open `agent.py` and change `pyautogui.PAUSE = 0.5` to `0.1` for faster movement.

**"It clicks the wrong thing"**
The AI relies on visual descriptions. If it fails, try to be more specific (e.g., instead of "Click the button", say "Click the blue 'Submit' button in the bottom right").

---

## ⚠️ Disclaimer

This tool allows an AI to control your mouse and keyboard.

* **Do not** leave it unattended.
* **Do not** use it for banking or sensitive data entry.
* **Always** keep your hand near the mouse to intervene if necessary.

