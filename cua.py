import tkinter as tk
from tkinter import scrolledtext, messagebox, Toplevel, Label
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
from PIL import Image, ImageDraw, ImageTk, ImageFont
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. DPI SCALING FIX ---
try:
    if platform.system() == "Windows":
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass 

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
    "space": "space", "delete": "delete"
}

# --- OCR ENGINE (TextBrain) ---
class TextBrain:
    def __init__(self, logger):
        self.log = logger
        self.log("📖 Loading OCR Model...", "system")
        self.reader = easyocr.Reader(['en'], gpu=False)

    def find_text(self, text_query, screenshot_path):
        """
        Reads text from the CLEAN screenshot (no grid lines).
        """
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
    def __init__(self, logger_func, debug_updater, text_brain):
        self.log = logger_func
        self.update_debug_view = debug_updater
        self.client = OpenAI()
        self.ocr = text_brain
        self.stop_flag = False
        
        # Get Screen Size (DPI Aware)
        self.screen_w, self.screen_h = pyautogui.size()
        self.log(f"🖥️ Monitor Resolution: {self.screen_w}x{self.screen_h}", "system")

        self.system_prompt = {
            "role": "system",
            "content": """You are a VISION-BASED MOUSE AGENT.
            
            **INSTRUCTIONS:**
            1. You will see a SCREENSHOT with a FINE RED GRID (30x30).
            2. The grid steps are approx 0.033.
            3. A BLUE CIRCLE represents your CURRENT MOUSE POSITION.
            4. To click, visually estimate the coordinate (x,y) of the target.
            5. **CRITICAL:** You MUST call `task_finished` when the goal is met.
            
            **TOOLS:**
            - `click_coordinate(x_pct, y_pct, description)`: MAIN TOOL.
            - `drag_mouse(start_x, start_y, end_x, end_y)`: Drag from A to B.
            - `type_text(text)`: Type on keyboard.
            - `press_key(key)`: Press keys. Supports combos like "ctrl+z".
            - `scroll(amount)`: Scroll down (-500) or up (500).
            - `use_ocr_backup(text)`: Use ONLY if visual grid fails. It reads a CLEAN image without grid lines.
            - `task_finished(reason)`: Call this IMMEDIATELY when done.
            """
        }

    def capture_screen_data(self):
        """
        Captures the screen ONCE, then creates two versions:
        1. Clean version (for OCR)
        2. Grid version + Mouse Cursor (for AI Vision)
        """
        clean_screenshot = pyautogui.screenshot()
        img_w, img_h = clean_screenshot.size
        
        self.scale_x = img_w / self.screen_w
        self.scale_y = img_h / self.screen_h
        
        # 1. Save CLEAN version for OCR
        clean_screenshot.save("debug_clean.png")

        # 2. Create GRID version for AI
        grid_img = clean_screenshot.copy()
        draw = ImageDraw.Draw(grid_img)
        
        # --- ULTRA PRECISION GRID (30x30) ---
        divisions = 30
        step_x = img_w / divisions
        step_y = img_h / divisions
        
        # Vertical Lines
        for i in range(1, divisions):
            x = i * step_x
            draw.line([(x, 0), (x, img_h)], fill=(255, 0, 0, 100), width=1)
            if i % 3 == 0: 
                draw.text((x + 2, 5), f"{i/divisions:.2f}", fill="red", font_size=15)

        # Horizontal Lines
        for i in range(1, divisions):
            y = i * step_y
            draw.line([(0, y), (img_w, y)], fill=(255, 0, 0, 100), width=1)
            if i % 3 == 0:
                draw.text((5, y + 2), f"{i/divisions:.2f}", fill="red", font_size=15)

        # --- DRAW CURRENT MOUSE POSITION ---
        try:
            # Get current mouse position
            curr_x, curr_y = pyautogui.position()
            
            # Scale it to image coordinates (if DPI scaling exists)
            img_mx = curr_x * self.scale_x
            img_my = curr_y * self.scale_y
            
            r = 15 # Radius of mouse marker
            
            # Draw Blue Circle
            draw.ellipse((img_mx - r, img_my - r, img_mx + r, img_my + r), outline="blue", width=4)
            # Draw Crosshair
            draw.line((img_mx - r*1.5, img_my, img_mx + r*1.5, img_my), fill="blue", width=2)
            draw.line((img_mx, img_my - r*1.5, img_mx, img_my + r*1.5), fill="blue", width=2)
            
            # Label
            draw.text((img_mx + 20, img_my - 20), "MOUSE", fill="blue", font_size=20)
            
        except Exception as e:
            self.log(f"Warning: Could not draw mouse cursor: {e}", "error")

        # Save Grid version for visual debugging
        grid_img.save("debug_grid.png")
        
        return clean_screenshot, grid_img

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

            # --- CAPTURE PHASE ---
            clean_img, grid_img = self.capture_screen_data()
            
            # Send the GRID image to the Debug UI (so user sees what AI sees)
            self.update_debug_view(grid_img, "Thinking...", None)
            
            # Send the GRID image to OpenAI
            b64_img = self.get_base64_image(grid_img)

            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Step {step}/{max_steps}. Red Grid = Coords. Blue Circle = Your Mouse."},
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
                                    "action": {"type": "string", "enum": ["click_coordinate", "drag_mouse", "type_text", "press_key", "scroll", "use_ocr_backup", "task_finished"]},
                                    "x_pct": {"type": "number"},
                                    "y_pct": {"type": "number"},
                                    "start_x": {"type": "number"},
                                    "start_y": {"type": "number"},
                                    "end_x": {"type": "number"},
                                    "end_y": {"type": "number"},
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
                
                act = "unknown"
                res_txt = "Action failed."
                
                try:
                    args = json.loads(tool.function.arguments)
                    act = args.get("action")
                    thought = args.get("thought", "")
                    
                    self.log(f"💭 {thought}", "thought")
                    self.log(f"⚡ {act.upper()}", "action")

                    # --- DEBUG VIEW UPDATE ---
                    debug_coords = None
                    debug_label = f"ACTION: {act}"
                    if act == "click_coordinate":
                        xp, yp = args.get("x_pct"), args.get("y_pct")
                        if xp: debug_coords = (xp, yp)
                    elif act == "drag_mouse":
                        sx, sy = args.get("start_x"), args.get("start_y")
                        if sx: debug_coords = (sx, sy)

                    if debug_coords:
                        self.update_debug_view(grid_img, debug_label, debug_coords)
                        time.sleep(0.5)

                    # --- EXECUTE ACTIONS ---
                    if act == "task_finished":
                        self.log(f"✅ Finished: {args.get('reason')}", "success")
                        self.stop_flag = True
                        res_txt = "Task finished."

                    elif act == "click_coordinate":
                        xp, yp = args.get("x_pct"), args.get("y_pct")
                        if xp is not None:
                            real_x = int(xp * self.screen_w)
                            real_y = int(yp * self.screen_h)
                            self.log(f"🖱️ Aiming at {real_x},{real_y}...", "normal")
                            pyautogui.moveTo(real_x, real_y, duration=0.5) 
                            pyautogui.click()
                            res_txt = "Clicked."
                        else:
                            res_txt = "Error: Coordinates missing."

                    elif act == "drag_mouse":
                        sx, sy = args.get("start_x"), args.get("start_y")
                        ex, ey = args.get("end_x"), args.get("end_y")
                        if None not in [sx, sy, ex, ey]:
                            real_sx, real_sy = int(sx * self.screen_w), int(sy * self.screen_h)
                            real_ex, real_ey = int(ex * self.screen_w), int(ey * self.screen_h)
                            
                            self.log(f"✊ Dragging {real_sx},{real_sy} -> {real_ex},{real_ey}", "normal")
                            pyautogui.moveTo(real_sx, real_sy)
                            pyautogui.dragTo(real_ex, real_ey, duration=1.0, button='left')
                            res_txt = "Dragged."
                        else:
                            res_txt = "Error: Missing drag coordinates."

                    elif act == "type_text":
                        text_to_type = args.get("text", "")
                        pyautogui.write(text_to_type, interval=0.05)
                        self.log(f"⌨️ Type: {text_to_type}", "normal")
                        res_txt = "Typed text."

                    elif act == "press_key":
                        raw_key = args.get("key", "").lower()
                        keys = raw_key.split('+')
                        final_keys = []
                        for k in keys:
                            k = k.strip()
                            if k in KEY_MAP: k = KEY_MAP[k]
                            final_keys.append(k)
                        
                        if len(final_keys) > 1:
                            pyautogui.hotkey(*final_keys)
                            self.log(f"🎹 Combo: {'+'.join(final_keys)}", "normal")
                        else:
                            pyautogui.press(final_keys[0])
                            self.log(f"🎹 Press: {final_keys[0]}", "normal")
                        res_txt = f"Pressed {raw_key}."

                    elif act == "scroll":
                        pyautogui.scroll(args.get("amount", -500))
                        res_txt = "Scrolled."

                    elif act == "use_ocr_backup":
                        txt = args.get("text")
                        # Use clean image for OCR
                        matches = self.ocr.find_text(txt, "debug_clean.png")
                        
                        if matches:
                            img_x, img_y = matches[0]
                            final_x = int(img_x / self.scale_x)
                            final_y = int(img_y / self.scale_y)
                            
                            self.log(f"🔍 OCR found '{txt}' at {final_x},{final_y}", "success")
                            
                            # Visual Debug for OCR Click
                            vis_x = img_x / (self.screen_w * self.scale_x)
                            vis_y = img_y / (self.screen_h * self.scale_y)
                            self.update_debug_view(grid_img, f"OCR CLICK: {txt}", (vis_x, vis_y))
                            
                            pyautogui.moveTo(final_x, final_y, duration=0.5)
                            pyautogui.click()
                            res_txt = f"OCR clicked '{txt}'"
                        else:
                            self.log(f"❌ OCR could not find '{txt}'", "error")
                            res_txt = "OCR found nothing."
                
                except json.JSONDecodeError:
                    self.log(f"⚠️ JSON Parse Error on tool call.", "error")
                    res_txt = "Error: Invalid JSON format generated by AI. Please retry."
                except Exception as e:
                    self.log(f"⚠️ Action Error: {e}", "error")
                    res_txt = f"Error executing action: {e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool.id,
                    "content": res_txt
                })
                
                if self.stop_flag: break

        self.log("🏁 Agent Stopped.", "system")

# --- MAIN GUI ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Vision Agent")
        self.root.geometry("600x750")
        
        self.log_q = queue.Queue()
        self.debug_q = queue.Queue()
        
        lbl_font = ("Arial", 12, "bold")
        btn_font = ("Arial", 12, "bold")
        
        top_frame = tk.Frame(root, padx=10, pady=10)
        top_frame.pack(fill=tk.X)
        
        tk.Label(top_frame, text="Task Instruction:", font=lbl_font).pack(anchor="w")
        self.entry = tk.Entry(top_frame, font=("Arial", 11), width=50)
        self.entry.pack(fill=tk.X, pady=(0, 10))
        
        # --- CONTROL ROW ---
        ctrl_frame = tk.Frame(top_frame)
        ctrl_frame.pack(fill=tk.X, anchor="w")
        
        tk.Label(ctrl_frame, text="Max Steps:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.steps_spin = tk.Spinbox(ctrl_frame, from_=5, to=100, width=5, font=("Arial", 10))
        self.steps_spin.delete(0, "end")
        self.steps_spin.insert(0, 15)
        self.steps_spin.pack(side=tk.LEFT, padx=5)

        # DEBUG CHECKBOX
        self.debug_var = tk.IntVar(value=1) # Default ON
        self.chk_debug = tk.Checkbutton(ctrl_frame, text="Show Vision Debugger", variable=self.debug_var, font=("Arial", 10))
        self.chk_debug.pack(side=tk.LEFT, padx=20)

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

        self.debug_window = None
        self.debug_label = None

        threading.Thread(target=self.init_ocr, daemon=True).start()
        self.root.after(100, self.process_queues)

    def init_ocr(self):
        def log_bridge(t, s="normal"): self.log_q.put((t, s))
        self.ocr = TextBrain(log_bridge)
        self.log_q.put(("✅ System Ready.", "success"))

    def log(self, text, style="normal"):
        self.log_q.put(("log", text, style))

    # --- DEBUG VIEW UPDATER ---
    def update_debug_image(self, pil_img, status_text, coords):
        if self.debug_var.get() == 1:
            self.debug_q.put((pil_img, status_text, coords))

    def process_queues(self):
        # Process Logs
        while not self.log_q.empty():
            item = self.log_q.get()
            if item[0] == "log":
                _, text, style = item
                self.console.config(state='normal')
                self.console.insert(tk.END, f"{text}\n", style)
                self.console.see(tk.END)
                self.console.config(state='disabled')
            else:
                self.console.config(state='normal')
                self.console.insert(tk.END, f"{item[0]}\n", item[1])
                self.console.see(tk.END)
                self.console.config(state='disabled')

        # Process Debug Images
        while not self.debug_q.empty():
            pil_img, text, coords = self.debug_q.get()
            self.show_debug_window(pil_img, text, coords)

        self.root.after(100, self.process_queues)

    def show_debug_window(self, pil_img, text, coords):
        if self.debug_window is None or not self.debug_window.winfo_exists():
            self.debug_window = Toplevel(self.root)
            self.debug_window.title("AI Vision Stream")
            self.debug_window.geometry("600x400")
            self.debug_label = Label(self.debug_window)
            self.debug_label.pack(fill=tk.BOTH, expand=True)

        img_copy = pil_img.copy()
        draw = ImageDraw.Draw(img_copy)
        w, h = img_copy.size
        
        # Draw Coordinate Target (Green)
        if coords:
            cx, cy = coords[0] * w, coords[1] * h
            r = 10
            draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline="green", width=3)
            draw.line((cx-r*2, cy, cx+r*2, cy), fill="green", width=2)
            draw.line((cx, cy-r*2, cx, cy+r*2), fill="green", width=2)
        
        # Draw Status Text
        if text:
            draw.rectangle((0, 0, w, 40), fill="black")
            draw.text((10, 10), text, fill="white", font_size=20)

        # Resize for the window
        win_w = self.debug_window.winfo_width()
        win_h = self.debug_window.winfo_height()
        if win_w < 50: win_w = 600
        if win_h < 50: win_h = 400
        
        img_copy.thumbnail((win_w, win_h))
        self.tk_img = ImageTk.PhotoImage(img_copy) 
        
        self.debug_label.config(image=self.tk_img)

    def start(self):
        task = self.entry.get()
        if not task: return messagebox.showwarning("Error", "Enter a task!")
        try: steps = int(self.steps_spin.get())
        except: steps = 15
        
        self.worker = AgentWorker(self.log, self.update_debug_image, self.ocr)
        
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