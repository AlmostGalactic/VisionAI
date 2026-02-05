import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import queue
import os
import time
import base64
import json
import pyautogui
import cv2
import numpy as np
import easyocr
import io
from PIL import Image
from openai import OpenAI
from ultralytics import YOLOWorld
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5
MAX_RETRIES = 15

# --- KEY MAPPING ---
KEY_MAP = {
    "control": "ctrl", "ctl": "ctrl",
    "command": "command", "cmd": "command",
    "option": "option", "opt": "option",
    "windows": "win", "super": "win",
    "return": "enter", "esc": "escape"
}

# --- BRAINS (Vision & OCR) ---
class VisionBrain:
    def __init__(self, logger):
        self.log = logger
        # Switched to 's' (Small) model for CPU speed
        self.log("👁️ Loading YOLO Vision Model (CPU Mode)...", "system")
        self.model = YOLOWorld('yolov8s-worldv2.pt') 

    def find_all_items(self, description, screenshot_path):
        self.log(f"🔎 YOLO scanning for: '{description}'...", "normal")
        self.model.set_classes([description])
        
        # device='cpu' ensures it works on all computers
        results = self.model.predict(
            screenshot_path,
            conf=0.10,
            imgsz=1280, # Reduced size for CPU speed
            verbose=False,
            max_det=10,
            device='cpu'
        )
        
        matches = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                coords = box.xyxy[0].tolist()
                center_x = int((coords[0] + coords[2]) / 2)
                center_y = int((coords[1] + coords[3]) / 2)
                matches.append((center_x, center_y))
        
        self.log(f"🎯 YOLO found {len(matches)} matches.", "success" if matches else "error")
        return matches

class TextBrain:
    def __init__(self, logger):
        self.log = logger
        self.log("📖 Loading EasyOCR Model (CPU Mode)...", "system")
        # gpu=False ensures no errors if user lacks NVIDIA card
        self.reader = easyocr.Reader(['en'], gpu=False) 

    def find_all_text(self, text_query, screenshot_path):
        self.log(f"🔎 OCR searching for: '{text_query}'...", "normal")
        img = cv2.imread(screenshot_path)
        if img is None: return []
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        results = self.reader.readtext(gray_img)
        matches = []
        for (bbox, text, conf) in results:
            if text_query.lower() in text.lower():
                (tl, tr, br, bl) = bbox
                center_x = int((tl[0] + br[0]) / 2)
                center_y = int((tl[1] + br[1]) / 2)
                matches.append((center_x, center_y))
        
        self.log(f"🎯 OCR found {len(matches)} matches.", "success" if matches else "error")
        return matches

    def read_all_text(self, screenshot_path):
        self.log("📖 Reading ALL text on screen...", "normal")
        img = cv2.imread(screenshot_path)
        if img is None: return []
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray_img)
        return [text for (_, text, _) in results]

# --- THE AGENT WORKER ---
class AgentWorker:
    def __init__(self, logger_func, vision_brain, text_brain):
        self.log = logger_func 
        self.client = OpenAI()
        self.vision = vision_brain
        self.ocr = text_brain
        self.stop_flag = False
        
        # Screen Calc
        self.screen_w, self.screen_h = pyautogui.size()
        test_shot = pyautogui.screenshot()
        self.img_w, self.img_h = test_shot.size
        self.scale_x = self.img_w / self.screen_w
        self.scale_y = self.img_h / self.screen_h
        
        self.log(f"📏 Screen: {self.screen_w}x{self.screen_h} (Scale: {self.scale_x:.2f})", "system")

        self.system_prompt = {
            "role": "system",
            "content": """You are a ROBOTIC ACTION AGENT.
            
            PROTOCOL:
            1. **AMBIGUITY:** If multiple matches found, use `match_index`.
            2. **COORDINATES:** Always prefer 0.0-1.0 range for x_pct/y_pct.
            
            TOOLS:
            - find_coordinates(description, type='icon'/'text', match_index=0)
            - click_mouse(x_pct, y_pct, num_clicks=1, button='left')
            - drag_mouse(start_x_pct, start_y_pct, end_x_pct, end_y_pct)
            - scroll_screen(amount) (-500 = DOWN)
            - read_screen_text()
            - type_text(text_content)
            - press_key(key_name)
            - task_finished(success=True, reason="...")
            """
        }

    def take_screenshot(self):
        # We save high-res for LOCAL vision/ocr
        path = "screen_temp.png"
        screenshot = pyautogui.screenshot()
        screenshot.save(path)
        return path, screenshot

    def get_compressed_base64(self, pil_image):
        # 1. Resize to max 1024 width for speed
        max_size = (1024, 1024)
        pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # 2. Save as JPEG (Fast upload)
        buffered = io.BytesIO()
        pil_image.convert("RGB").save(buffered, format="JPEG", quality=70)
        
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def resolve_coords(self, x_in, y_in):
        if x_in is None or y_in is None: return 0, 0
        def convert(val, max_dim):
            if val <= 1.0: return int(max_dim * val)
            elif val <= 100.0: return int(max_dim * (val / 100.0))
            else: return int(val)
        final_x = convert(x_in, self.screen_w)
        final_y = convert(y_in, self.screen_h)
        return final_x, final_y

    def run_task(self, user_task):
        self.stop_flag = False
        messages = [self.system_prompt]
        messages.append({"role": "user", "content": user_task})
        
        loop_count = 0
        task_active = True
        
        self.log(f"🚀 Starting Task: {user_task}", "system")

        while task_active and not self.stop_flag:
            loop_count += 1
            if loop_count > MAX_RETRIES:
                self.log("⚠️ Max retries reached. Stopping.", "error")
                break
            
            img_path, pil_img = self.take_screenshot()
            
            # Compress for OpenAI
            base64_img = self.get_compressed_base64(pil_img)
            
            content_msg = [
                {"type": "text", "text": f"Step {loop_count}: Screen state below. OUTPUT A TOOL CALL."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]
            messages.append({"role": "user", "content": content_msg})

            self.log(f"🧠 Thinking (Step {loop_count})...", "normal")
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tool_choice="required",
                    tools=[{
                        "type": "function",
                        "function": {
                            "name": "computer_tools",
                            "description": "Control tools",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "thought": {"type": "string"},
                                    "action": {"type": "string", "enum": ["click_mouse", "drag_mouse", "find_coordinates", "read_screen_text", "type_text", "press_key", "scroll_screen", "task_finished"]},
                                    "description": {"type": "string"},
                                    "type": {"type": "string", "enum": ["icon", "text"]},
                                    "match_index": {"type": "integer"},
                                    "text_content": {"type": "string"},
                                    "num_clicks": {"type": "integer"},
                                    "amount": {"type": "integer"},
                                    "button": {"type": "string"},
                                    "x_pct": {"type": "number"},
                                    "y_pct": {"type": "number"},
                                    "start_x_pct": {"type": "number"},
                                    "start_y_pct": {"type": "number"},
                                    "end_x_pct": {"type": "number"},
                                    "end_y_pct": {"type": "number"},
                                    "key_name": {"type": "string"},
                                    "reason": {"type": "string"}
                                },
                                "required": ["thought", "action"]
                            }
                        }
                    }]
                )
            except Exception as e:
                self.log(f"❌ API Error: {e}", "error")
                break

            msg = response.choices[0].message
            messages.append(msg)
            
            if msg.tool_calls:
                for tool in msg.tool_calls:
                    if self.stop_flag: break
                    
                    args = json.loads(tool.function.arguments)
                    action = args.get("action")
                    thought = args.get("thought", "No thought provided.")
                    
                    self.log(f"💭 {thought}", "thought")
                    self.log(f"⚡ ACTION: {action.upper()}", "action")
                    
                    time.sleep(0.1) 
                    result_message = "Action executed."

                    if action == "task_finished":
                        self.log(f"✅ Success: {args.get('reason')}", "success")
                        task_active = False
                    
                    elif action == "scroll_screen":
                        amt = args.get("amount", -500)
                        pyautogui.scroll(amt)
                        result_message = f"Scrolled {amt}"
                        self.log(result_message, "normal")

                    elif action == "find_coordinates":
                        desc = args.get("description")
                        stype = args.get("type")
                        idx = args.get("match_index", 0)
                        
                        matches = []
                        if stype == "text": matches = self.ocr.find_all_text(desc, img_path)
                        else: matches = self.vision.find_all_items(desc, img_path)
                        
                        if len(matches) > 0:
                            if idx < len(matches):
                                found_x, found_y = matches[idx]
                                x_pct = (found_x / self.scale_x) / self.screen_w
                                y_pct = (found_y / self.scale_y) / self.screen_h
                                result_message = f"FOUND {len(matches)} matches. Using #{idx}. Coords: x={x_pct:.4f}, y={y_pct:.4f}."
                            else:
                                result_message = f"ERROR: match_index {idx} out of range ({len(matches)} found)."
                        else:
                            result_message = f"FAILED to find '{desc}'."
                        self.log(result_message, "normal")

                    elif action == "read_screen_text":
                        all_txt = self.ocr.read_all_text(img_path)
                        preview = str(all_txt[:50])
                        result_message = f"VISIBLE: {preview}"
                        self.log(f"📄 Read Screen: {len(all_txt)} items found.", "normal")

                    elif action == "click_mouse":
                        tx, ty = self.resolve_coords(args.get("x_pct"), args.get("y_pct"))
                        pyautogui.click(x=tx, y=ty, clicks=args.get("num_clicks", 1), button=args.get("button", "left"))
                        result_message = f"Clicked {tx},{ty}"
                        self.log(result_message, "normal")

                    elif action == "drag_mouse":
                         sx, sy = self.resolve_coords(args.get("start_x_pct"), args.get("start_y_pct"))
                         ex, ey = self.resolve_coords(args.get("end_x_pct"), args.get("end_y_pct"))
                         pyautogui.moveTo(sx, sy)
                         pyautogui.dragTo(ex, ey, duration=0.8)
                         result_message = f"Dragged {sx},{sy} -> {ex},{ey}"
                         self.log(result_message, "normal")

                    elif action == "type_text":
                        text = args.get("text_content")
                        pyautogui.write(text, interval=0.02)
                        result_message = "Typed text."
                        self.log(f"⌨️ Typed: {text}", "normal")

                    elif action == "press_key":
                        raw_keys = args.get("key_name", "").lower().replace(" ", "").split('+')
                        clean_keys = [KEY_MAP.get(k, k) for k in raw_keys]
                        if len(clean_keys) > 1: pyautogui.hotkey(*clean_keys, interval=0.1)
                        else: pyautogui.press(clean_keys[0])
                        result_message = f"Pressed {clean_keys}"
                        self.log(result_message, "normal")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool.id,
                        "content": result_message
                    })

                    if not task_active: break
        
        self.log("🏁 Task ended.", "system")

# --- MAIN GUI CLASS ---
class AgentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vision Agent AI (Standard)")
        self.root.geometry("600x700")
        
        self.log_queue = queue.Queue()
        self._setup_ui()
        
        self.log("⏳ Initializing CPU Models... Please wait...", "system")
        threading.Thread(target=self._init_brains, daemon=True).start()
        
        self.root.after(100, self._process_log_queue)

    def _setup_ui(self):
        top_frame = tk.Frame(self.root, pady=10, padx=10)
        top_frame.pack(fill=tk.X)
        
        tk.Label(top_frame, text="Task:").pack(side=tk.LEFT)
        self.task_entry = tk.Entry(top_frame, width=40)
        self.task_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.task_entry.bind('<Return>', lambda event: self.start_agent())
        
        btn_frame = tk.Frame(self.root, pady=5)
        btn_frame.pack(fill=tk.X, padx=10)
        
        self.start_btn = tk.Button(btn_frame, text="▶ START AGENT", bg="#ccffcc", command=self.start_agent, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.stop_btn = tk.Button(btn_frame, text="⏹ STOP", bg="#ffcccc", command=self.stop_agent, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.console = scrolledtext.ScrolledText(self.root, state='disabled', bg="black", fg="white", font=("Consolas", 10))
        self.console.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.console.tag_config("system", foreground="white")
        self.console.tag_config("thought", foreground="cyan")
        self.console.tag_config("action", foreground="yellow")
        self.console.tag_config("success", foreground="#00ff00") 
        self.console.tag_config("error", foreground="#ff3333")   
        self.console.tag_config("normal", foreground="#cccccc") 

    def _init_brains(self):
        try:
            def logger(text, tag="normal"): self.log_queue.put((text, tag))
            self.vision = VisionBrain(logger)
            self.text = TextBrain(logger)
            self.agent_worker = None
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
            self.log("✅ Ready. Type a task and click Start.", "success")
        except Exception as e:
            self.log(f"CRITICAL ERROR LOADING MODELS: {e}", "error")

    def log(self, text, tag="normal"):
        self.log_queue.put((text, tag))

    def _process_log_queue(self):
        while not self.log_queue.empty():
            text, tag = self.log_queue.get()
            self.console.config(state='normal')
            self.console.insert(tk.END, text + "\n", tag)
            self.console.see(tk.END) 
            self.console.config(state='disabled')
        self.root.after(100, self._process_log_queue)

    def start_agent(self):
        task = self.task_entry.get()
        if not task:
            messagebox.showwarning("Input Error", "Please enter a task.")
            return

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.task_entry.config(state=tk.DISABLED)
        
        def thread_logger(text, tag="normal"): self.log_queue.put((text, tag))
        self.agent_worker = AgentWorker(thread_logger, self.vision, self.text)
        self.thread = threading.Thread(target=self._run_thread_logic, args=(task,))
        self.thread.start()

    def _run_thread_logic(self, task):
        try:
            self.agent_worker.run_task(task)
        except Exception as e:
            self.log(f"Thread Error: {e}", "error")
        finally:
            self.root.after(0, self._reset_ui)

    def stop_agent(self):
        if self.agent_worker:
            self.log("🛑 Stopping agent...", "error")
            self.agent_worker.stop_flag = True

    def _reset_ui(self):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.task_entry.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = AgentApp(root)
    root.mainloop()