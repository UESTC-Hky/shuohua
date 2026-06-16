"""
GUI for Speaker Recognition System (MFCC + VQ/LBG)
Tkinter-based graphical interface.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys
import os

import train
import test

# ---------------------------------------------------------------------------
# Default parameters (matching main.py)
# ---------------------------------------------------------------------------
DEFAULT = {
    "frame_size": 256,
    "frame_shift": 128,
    "num_filters": 26,
    "num_ceps": 12,
    "fft_size": 512,
    "codebook_size": 16,
    "train_dir": "D:/data/TRAIN",
    "test_dir": "D:/data/TEST",
    "model_dir": "D:/JIEDAN/models",
}


# ---------------------------------------------------------------------------
# stdout redirector — threads safe via root.after()
# ---------------------------------------------------------------------------
class _OutputRedirector:
    """Writes strings to a tkinter Text widget (thread-safe via after())."""

    def __init__(self, widget, root):
        self._widget = widget
        self._root = root

    def write(self, text):
        if not text:
            return
        self._root.after(0, self._write_to_widget, text)

    def _write_to_widget(self, text):
        self._widget.configure(state="normal")
        self._widget.insert(tk.END, text)
        self._widget.see(tk.END)
        self._widget.configure(state="disabled")

    def flush(self):
        pass  # not needed for tkinter


# ---------------------------------------------------------------------------
# Main GUI Application
# ---------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("说话人识别系统 (MFCC + VQ/LBG)")
        self.root.geometry("920x620")
        self.root.minsize(820, 500)

        # ----- split pane (left | right) -----
        paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left = ttk.Frame(paned, width=340)
        right = ttk.Frame(paned, width=580)
        paned.add(left, weight=1)
        paned.add(right, weight=2)

        self._build_params(left)
        self._build_files(left)
        self._build_actions(left)
        self._build_output(right)
        self._build_status(root)

        # stdout redirection
        self._redirector = _OutputRedirector(self.output_text, root)
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        # threading guard
        self._running = False

    # =======================================================================
    # Left Panel — Parameters
    # =======================================================================
    def _build_params(self, parent):
        frame = ttk.LabelFrame(parent, text="参数设置", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=(5, 3))

        param_specs = [
            ("帧长 (Frame Size):",     "frame_size"),
            ("帧移 (Frame Shift):",    "frame_shift"),
            ("Mel 滤波器数:",          "num_filters"),
            ("MFCC 系数数:",           "num_ceps"),
            ("FFT 点数:",              "fft_size"),
            ("码本大小 (Codebook):",   "codebook_size"),
        ]

        self.param_vars = {}
        for i, (label, key) in enumerate(param_specs):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=3)
            var = tk.StringVar(value=str(DEFAULT[key]))
            self.param_vars[key] = var
            ttk.Entry(frame, textvariable=var, width=12).grid(
                row=i, column=1, sticky=tk.W, pady=3, padx=(8, 0))

        ttk.Button(frame, text="恢复默认值", command=self._reset_params).grid(
            row=len(param_specs), column=0, columnspan=2, pady=(10, 0))

    def _reset_params(self):
        for key, val in DEFAULT.items():
            if key in self.param_vars:
                self.param_vars[key].set(str(val))

    def _get_params(self):
        """Read parameter values. Returns (dict, error_message)."""
        p = {}
        int_keys = [
            "frame_size", "frame_shift", "num_filters",
            "num_ceps", "fft_size", "codebook_size",
        ]
        for key in int_keys:
            try:
                p[key] = int(self.param_vars[key].get())
            except ValueError:
                return None, f"参数 '{key}' 必须为整数"
        return p, None

    # =======================================================================
    # Left Panel — File Paths
    # =======================================================================
    def _build_files(self, parent):
        frame = ttk.LabelFrame(parent, text="文件路径", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=3)

        # Train directory
        ttk.Label(frame, text="训练目录:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.train_dir_var = tk.StringVar(value=DEFAULT["train_dir"])
        ttk.Entry(frame, textvariable=self.train_dir_var).grid(
            row=0, column=1, sticky=tk.EW, pady=3, padx=(8, 4))
        ttk.Button(frame, text="浏览...", command=self._browse_train_dir).grid(
            row=0, column=2, pady=3)

        # Model directory
        ttk.Label(frame, text="模型目录:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.model_dir_var = tk.StringVar(value=DEFAULT["model_dir"])
        ttk.Entry(frame, textvariable=self.model_dir_var).grid(
            row=1, column=1, sticky=tk.EW, pady=3, padx=(8, 4))
        ttk.Button(frame, text="浏览...", command=self._browse_model_dir).grid(
            row=1, column=2, pady=3)

        # Test file (single)
        ttk.Label(frame, text="测试文件:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.test_file_var = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.test_file_var).grid(
            row=2, column=1, sticky=tk.EW, pady=3, padx=(8, 4))
        ttk.Button(frame, text="浏览...", command=self._browse_test_file).grid(
            row=2, column=2, pady=3)

        # Test directory (batch)
        ttk.Label(frame, text="批量测试目录:").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.test_dir_var = tk.StringVar(value=DEFAULT["test_dir"])
        ttk.Entry(frame, textvariable=self.test_dir_var).grid(
            row=3, column=1, sticky=tk.EW, pady=3, padx=(8, 4))
        ttk.Button(frame, text="浏览...", command=self._browse_test_dir).grid(
            row=3, column=2, pady=3)

        frame.columnconfigure(1, weight=1)

    def _browse_train_dir(self):
        d = filedialog.askdirectory(title="选择训练数据目录")
        if d:
            self.train_dir_var.set(d)

    def _browse_model_dir(self):
        d = filedialog.askdirectory(title="选择模型保存目录")
        if d:
            self.model_dir_var.set(d)

    def _browse_test_file(self):
        f = filedialog.askopenfilename(
            title="选择测试音频文件",
            filetypes=[("WAV 文件", "*.wav"), ("所有文件", "*.*")])
        if f:
            self.test_file_var.set(f)

    def _browse_test_dir(self):
        d = filedialog.askdirectory(title="选择批量测试目录")
        if d:
            self.test_dir_var.set(d)

    # =======================================================================
    # Left Panel — Action Buttons
    # =======================================================================
    def _build_actions(self, parent):
        frame = ttk.LabelFrame(parent, text="操作", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=3)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack()

        self.btn_train = ttk.Button(btn_frame, text="🔧 训练模型", command=self._on_train)
        self.btn_train.grid(row=0, column=0, padx=3, pady=4)

        self.btn_recog = ttk.Button(btn_frame, text="🎤 识别说话人", command=self._on_recognize)
        self.btn_recog.grid(row=0, column=1, padx=3, pady=4)

        self.btn_full = ttk.Button(btn_frame, text="🔄 一键全流程", command=self._on_full_pipeline)
        self.btn_full.grid(row=1, column=0, padx=3, pady=4)

        self.btn_clear = ttk.Button(btn_frame, text="🗑 清空输出", command=self._on_clear)
        self.btn_clear.grid(row=1, column=1, padx=3, pady=4)

    # =======================================================================
    # Right Panel — Output
    # =======================================================================
    def _build_output(self, parent):
        frame = ttk.LabelFrame(parent, text="输出结果", padding=6)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.output_text = scrolledtext.ScrolledText(
            frame, state="disabled", wrap=tk.WORD,
            font=("Consolas", 10), bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white",
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

    # =======================================================================
    # Status Bar
    # =======================================================================
    def _build_status(self, parent):
        self.status_var = tk.StringVar(value="就绪")
        bar = ttk.Label(
            parent, textvariable=self.status_var,
            relief=tk.SUNKEN, anchor=tk.W, padding=(8, 3),
        )
        bar.pack(side=tk.BOTTOM, fill=tk.X)

    # =======================================================================
    # Helpers
    # =======================================================================
    def _set_buttons_state(self, enabled):
        state = "normal" if enabled else "disabled"
        self.btn_train.configure(state=state)
        self.btn_recog.configure(state=state)
        self.btn_full.configure(state=state)

    def _set_status(self, text):
        self.root.after(0, self.status_var.set, text)

    def _run_in_thread(self, target):
        """Start target() in a daemon thread with button guard."""
        if self._running:
            messagebox.showinfo("提示", "正在运行中，请等待当前任务完成。")
            return
        self._running = True
        self._set_buttons_state(False)

        def wrapper():
            try:
                sys.stdout = self._redirector
                sys.stderr = self._redirector
                target()
            except Exception as exc:
                self.root.after(0, lambda: self._append_line(
                    f"[错误] {exc}\n"))
            finally:
                sys.stdout = self._original_stdout
                sys.stderr = self._original_stderr
                self._running = False
                self._set_buttons_state(True)
                self._set_status("就绪")

        threading.Thread(target=wrapper, daemon=True).start()

    def _append_line(self, text):
        self._redirector.write(text)

    # =======================================================================
    # Actions
    # =======================================================================
    def _on_train(self):
        params, err = self._get_params()
        if err:
            messagebox.showwarning("参数错误", err)
            return
        train_dir = self.train_dir_var.get()
        model_dir = self.model_dir_var.get()

        if not os.path.isdir(train_dir):
            self._append_line(f"[错误] 训练目录不存在: {train_dir}\n")
            return

        self._set_status("训练中...")
        self._append_line("=" * 50 + "\n")
        self._append_line("PHASE 1: 训练\n")
        self._append_line("-" * 40 + "\n")

        def do_train():
            train.train_all(
                train_dir, model_dir,
                frame_size=params["frame_size"],
                frame_shift=params["frame_shift"],
                num_filters=params["num_filters"],
                num_ceps=params["num_ceps"],
                fft_size=params["fft_size"],
                codebook_size=params["codebook_size"],
            )
            self._append_line("\n✅ 训练完成\n")

        self._run_in_thread(do_train)

    def _on_recognize(self):
        params, err = self._get_params()
        if err:
            messagebox.showwarning("参数错误", err)
            return
        test_file = self.test_file_var.get()
        model_dir = self.model_dir_var.get()

        if not test_file:
            messagebox.showwarning("提示", "请先选择测试音频文件。")
            return
        if not os.path.isfile(test_file):
            self._append_line(f"[错误] 测试文件不存在: {test_file}\n")
            return

        self._set_status("识别中...")
        self._append_line("\n--- 识别 ---\n")

        def do_recognize():
            speaker, distortion = test.recognize(
                test_file, model_dir,
                frame_size=params["frame_size"],
                frame_shift=params["frame_shift"],
                num_filters=params["num_filters"],
                num_ceps=params["num_ceps"],
                fft_size=params["fft_size"],
            )
            name = os.path.basename(test_file)
            self._append_line(
                f"文件: {name}\n"
                f"识别结果: 说话人 S{speaker}\n"
                f"失真度: {distortion:.4f}\n"
                f"✅ 识别完成\n"
            )

        self._run_in_thread(do_recognize)

    def _on_full_pipeline(self):
        params, err = self._get_params()
        if err:
            messagebox.showwarning("参数错误", err)
            return
        train_dir = self.train_dir_var.get()
        test_dir = self.test_dir_var.get()
        model_dir = self.model_dir_var.get()

        if not os.path.isdir(train_dir):
            self._append_line(f"[错误] 训练目录不存在: {train_dir}\n")
            return

        self._set_status("全流程运行中...")
        self._append_line("=" * 50 + "\n")
        self._append_line("说话人识别系统 (MFCC + VQ/LBG)\n")
        self._append_line("=" * 50 + "\n")

        def do_all():
            # Train
            self._append_line("\n[Phase 1: 训练]\n")
            self._append_line("-" * 40 + "\n")
            train.train_all(
                train_dir, model_dir,
                frame_size=params["frame_size"],
                frame_shift=params["frame_shift"],
                num_filters=params["num_filters"],
                num_ceps=params["num_ceps"],
                fft_size=params["fft_size"],
                codebook_size=params["codebook_size"],
            )

            # Recognize all
            self._append_line("\n[Phase 2: 识别]\n")
            self._append_line("-" * 40 + "\n")
            correct, total = test.test_all(
                test_dir, model_dir,
                frame_size=params["frame_size"],
                frame_shift=params["frame_shift"],
                num_filters=params["num_filters"],
                num_ceps=params["num_ceps"],
                fft_size=params["fft_size"],
            )
            self._append_line("\n" + "=" * 50 + "\n")
            self._append_line(
                f"最终结果: {correct}/{total} ({correct/total*100:.1f}%)\n")
            self._append_line("=" * 50 + "\n")
            self._append_line("✅ 全流程完成\n")

        self._run_in_thread(do_all)

    def _on_clear(self):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.configure(state="disabled")
        self._set_status("就绪")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
