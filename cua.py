import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import threading
import queue
import time
import base64
import json
import pyautogui
import cv2
import numpy as np
import easyocr
import io
import platform
import ctypes
from PIL import Image, ImageDraw
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. DPI SCALING FIX (CRITICAL FOR WINDOWS) ---
# This forces Python to recognize your monitor's real 4K/2K resolution
try:
    if platform.system() == "Windows":
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass # Not on Windows or failed, continue safely

# --- CONFIGURATION ---
load_dotenv()
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5

# --- KEY MAPPING ---
KEY_MAP = {
    "control": "ctrl", "ctl": "ctrl",
    "command": "command", "cmd": "command",
    "option": "option", "opt": "option",
    "windows": "win", "super": "win",
    "return": "enter", "esc": "escape",
    "space": "space"
}

# --- OPTIONAL OCR (Backup Only) ---
class TextBrain:
    def __init__(self, logger):
        self.log = logger
        self.log("📖 Loading OCR Model...", "system")
        self.reader = easyocr.Reader(['en'], gpu=False)

    def find_text(self, text_query, screenshot_path):
        img = cv2.imread(screenshot_path)
        if img is None: return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray)
        matches = []
        for (bbox, text, conf) in results:
            if text_query.lower() in text.lower():
                (tl, tr, br, bl) = bbox
                cx = int((tl[0] + br[0]) / 2)
                cy = int((tl[1] + br[1]) / 2)
                matches.append((cx, cy))
        return matches

# --- THE AGENT WORKER ---
class AgentWorker:
    def __init__(self, logger_func, text_brain):
        self.log = logger_func 
        self.client = OpenAI()
        self.ocr = text_brain
        self.stop_flag = False
        
        # Get Screen Size (Now DPI Aware)
        self.screen_w, self.screen_h = pyautogui.size()
        self.log(f"🖥️ Monitor Resolution: {self.screen_w}x{self.screen_h}", "system")

        self.system_prompt = {
            "role": "system",
            "content": """You are a VISION-BASED MOUSE AGENT.
            
            **INSTRUCTIONS:**
            1. You will see a SCREENSHOT with a RED GRID (0.0 - 1.0).
            2. To click, visually estimate the coordinate (x,y) of the target.
            3. **CRITICAL:** You MUST call `task_finished` when the goal is met.
            
            **TOOLS:**
            - `click_coordinate(x_pct, y_pct, description)`: MAIN TOOL.
            - `type_text(text)`: Type on keyboard.
            - `press_key(key)`: Press keys like 'enter', 'esc', 'ctrl+c'.
            - `scroll(amount)`: Scroll down (-500) or up (500).
            - `use_ocr_backup(text)`: Use ONLY if visual grid fails.
            - `task_finished(reason)`: Call this IMMEDIATELY when done.
            """
        }

    def take_screenshot_with_grid(self):
        screenshot = pyautogui.screenshot()
        img_w, img_h = screenshot.size
        
        # --- SCALING CHECK ---
        # If screenshot is bigger than screen size (Retina/HighDPI), we must know the ratio.
        self.scale_x = img_w / self.screen_w
        self.scale_y = img_h / self.screen_h
        
        draw = ImageDraw.Draw(screenshot)
        
        # Grid settings
        step_x = img_w / 10
        step_y = img_h / 10
        
        # Draw Red Grid
        for i in range(1, 10):
            x = i * step_x
            draw.line([(x, 0), (x, img_h)], fill="red", width=2)
            draw.text((x + 5, 10), f"{i/10:.1f}", fill="red", font_size=20)

        for i in range(1, 10):
            y = i * step_y
            draw.line([(0, y), (img_w, y)], fill="red", width=2)
            draw.text((10, y + 5), f"{i/10:.1f}", fill="red", font_size=20)

        screenshot.save("debug_grid.png")
        return screenshot

    def get_base64_image(self, pil_image):
        pil_image.thumbnail((1024, 1024))
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=70)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def run_task(self, user_task, max_steps):
        self.stop_flag = False
        messages = [self.system_prompt]
        messages.append({"role": "user", "content": f"Task: {user_task}"})
        
        step = 0
        self.log(f"🚀 Starting Task: {user_task}", "system")

        while not self.stop_flag:
            step += 1
            if step > max_steps:
                self.log(f"⚠️ Max steps ({max_steps}) reached. Stopping.", "error")
                break

            grid_img = self.take_screenshot_with_grid()
            b64_img = self.get_base64_image(grid_img)

            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Step {step}/{max_steps}. Screen state below."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }
            messages.append(user_msg)
            self.log(f"🧠 Thinking (Step {step})...", "normal")

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=300,
                    tools=[{
                        "type": "function",
                        "function": {
                            "name": "computer_tools",
                            "description": "Control computer",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "thought": {"type": "string"},
                                    "action": {"type": "string", "enum": ["click_coordinate", "type_text", "press_key", "scroll", "use_ocr_backup", "task_finished"]},
                                    "x_pct": {"type": "number"},
                                    "y_pct": {"type": "number"},
                                    "description": {"type": "string"},
                                    "text": {"type": "string"},
                                    "key": {"type": "string"},
                                    "amount": {"type": "integer"},
                                    "reason": {"type": "string"}
                                },
                                "required": ["thought", "action"]
                            }
                        }
                    }]
                )
            except Exception as e:
                self.log(f"API Error: {e}", "error")
                break

            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                content = msg.content if msg.content else "No content."
                self.log(f"💬 AI Message: {content}", "normal")
                messages.append({"role": "user", "content": "You did not trigger an action. If you are done, use the 'task_finished' tool."})
                continue 

            for tool in msg.tool_calls:
                if self.stop_flag: break
                
                args = json.loads(tool.function.arguments)
                act = args.get("action")
                thought = args.get("thought", "")
                
                self.log(f"💭 {thought}", "thought")
                self.log(f"⚡ {act.upper()}", "action")

                res_txt = "Action Done."

                if act == "task_finished":
                    self.log(f"✅ Finished: {args.get('reason')}", "success")
                    self.stop_flag = True
                    break

                elif act == "click_coordinate":
                    xp, yp = args.get("x_pct"), args.get("y_pct")
                    if xp is not None:
                        # --- SCALING FIX ---
                        # We multiply percentage by SCREEN size, not Image size.
                        real_x = int(xp * self.screen_w)
                        real_y = int(yp * self.screen_h)
                        
                        # VISUAL DEBUG: Move mouse slowly so you can see where it aims
                        self.log(f"🖱️ Aiming at {real_x},{real_y}...", "normal")
                        pyautogui.moveTo(real_x, real_y, duration=0.5) 
                        pyautogui.click()
                        self.log(f"🖱️ CLICKED!", "success")
                    else:
                        res_txt = "Error: Coordinates missing."

                elif act == "type_text":
                    pyautogui.write(args.get("text"), interval=0.05)
                    self.log(f"⌨️ Type: {args.get('text')}", "normal")

                elif act == "press_key":
                    k = args.get("key", "").lower()
                    if k in KEY_MAP: k = KEY_MAP[k]
                    pyautogui.press(k)
                    self.log(f"🎹 Press: {k}", "normal")

                elif act == "scroll":
                    pyautogui.scroll(args.get("amount", -500))

                elif act == "use_ocr_backup":
                    txt = args.get("text")
                    matches = self.ocr.find_text(txt, "debug_grid.png")
                    if matches:
                        # OCR matches are in Image Pixels. We must convert to Screen Pixels.
                        img_x, img_y = matches[0]
                        
                        # Convert Image -> Screen
                        final_x = int(img_x / self.scale_x)
                        final_y = int(img_y / self.scale_y)
                        
                        pyautogui.moveTo(final_x, final_y, duration=0.5)
                        pyautogui.click()
                        res_txt = f"OCR clicked '{txt}'"
                    else:
                        res_txt = "OCR found nothing."
                    self.log(res_txt, "normal")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool.id,
                    "content": res_txt
                })

        self.log("🏁 Agent Stopped.", "system")

# --- MAIN GUI ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Vision Grid Agent (DPI Fix)")
        self.root.geometry("600x700")
        
        self.log_q = queue.Queue()
        
        lbl_font = ("Arial", 12, "bold")
        btn_font = ("Arial", 12, "bold")
        
        top_frame = tk.Frame(root, padx=10, pady=10)
        top_frame.pack(fill=tk.X)
        
        tk.Label(top_frame, text="Task Instruction:", font=lbl_font).pack(anchor="w")
        self.entry = tk.Entry(top_frame, font=("Arial", 11), width=50)
        self.entry.pack(fill=tk.X, pady=(0, 10))
        
        step_frame = tk.Frame(top_frame)
        step_frame.pack(fill=tk.X, anchor="w")
        tk.Label(step_frame, text="Max Steps:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.steps_spin = tk.Spinbox(step_frame, from_=5, to=100, width=5, font=("Arial", 10))
        self.steps_spin.delete(0, "end")
        self.steps_spin.insert(0, 15)
        self.steps_spin.pack(side=tk.LEFT, padx=5)

        btn_frame = tk.Frame(root, padx=10, pady=5)
        btn_frame.pack(fill=tk.X)
        
        self.btn_start = tk.Button(btn_frame, text="▶ START AGENT", font=btn_font, bg="#ccffcc", height=2, command=self.start)
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.btn_stop = tk.Button(btn_frame, text="⏹ STOP", font=btn_font, bg="#ffcccc", height=2, command=self.stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.console = scrolledtext.ScrolledText(root, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 11), padx=10, pady=10)
        self.console.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.console.tag_config("thought", foreground="#569cd6")
        self.console.tag_config("action", foreground="#ce9178")
        self.console.tag_config("error", foreground="#f44747")
        self.console.tag_config("success", foreground="#b5cea8")
        self.console.tag_config("system", foreground="#808080")

        threading.Thread(target=self.init_ocr, daemon=True).start()
        self.root.after(100, self.process_logs)

    def init_ocr(self):
        def log_bridge(t, s="normal"): self.log_q.put((t, s))
        self.ocr = TextBrain(log_bridge)
        self.log_q.put(("✅ System Ready.", "success"))

    def log(self, text, style="normal"):
        self.log_q.put((text, style))

    def process_logs(self):
        while not self.log_q.empty():
            t, s = self.log_q.get()
            self.console.config(state='normal')
            self.console.insert(tk.END, f"{t}\n", s)
            self.console.see(tk.END)
            self.console.config(state='disabled')
        self.root.after(100, self.process_logs)

    def start(self):
        task = self.entry.get()
        if not task: return messagebox.showwarning("Error", "Enter a task!")
        try: steps = int(self.steps_spin.get())
        except: steps = 15
        self.worker = AgentWorker(self.log, self.ocr)
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.entry.config(state=tk.DISABLED)
        threading.Thread(target=self.run_worker_thread, args=(task, steps), daemon=True).start()

    def run_worker_thread(self, task, steps):
        try:
            self.worker.run_task(task, steps)
        except Exception as e:
            self.log(f"Critical Error: {e}", "error")
        finally:
            self.root.after(0, self.reset_ui)

    def stop(self):
        if hasattr(self, 'worker'):
            self.log("🛑 Stopping...", "error")
            self.worker.stop_flag = True

    def reset_ui(self):
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.entry.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()