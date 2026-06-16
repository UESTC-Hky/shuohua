# Speaker Recognition System (MFCC + VQ/LBG)

## 1. 环境依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.14.2 | |
| NumPy | 2.4.6 | 数组运算、FFT |
| SciPy | 1.17.1 | DCT 变换 |
| Tkinter | 内置（无需安装） | GUI 界面 |

Tkinter 随 Python 一起安装，**无需额外下载任何包**。

---

## 2. 如何运行

### 数据准备

音频数据需按以下目录结构存放：

```
D:\data\
├── TRAIN\           # 训练集：s1.wav ~ s8.wav
└── TEST\            # 测试集：s1.wav ~ s8.wav
```

### 方式一：图形界面（推荐）

```bash
cd D:\JIEDAN
D:\uestc\python\python.exe(后续换成你的python编译器位置) gui.py
```

界面提供参数调节、文件选择、一键训练/识别/全流程，结果实时显示。

### 方式二：命令行

```bash
cd D:\JIEDAN

# 完整流程（训练 + 识别）
D:\HuangKeYuan\python\python.exe(后续换成你的python编译器位置) main.py

# 仅训练，模型保存到 models/
D:\HuangKeYuan\python\python.exe(后续换成你的python编译器位置) train.py

# 仅识别（需已完成训练）
D:\HuangKeYuan\python\python.exe(后续换成你的python编译器位置) test.py
```

---

## 3. 每个文件的主要作用

| 文件 | 作用 |
|------|------|
| `gui.py` | **图形界面入口** — Tkinter GUI，左侧参数设置 + 文件选择 + 操作按钮，右侧实时输出结果 |
| `main.py` | **命令行入口** — 整合训练 + 识别全流程，定义全局参数，格式化输出结果 |
| `mfcc.py` | **MFCC 特征提取** — 读取 WAV → 预加重 → 分帧/加窗 → FFT → Mel 滤波器 → 对数能量 → DCT |
| `vq.py` | **矢量量化** — LBG 算法训练码本（`lbg_train`）、计算失真度匹配（`compute_distortion`） |
| `train.py` | **训练模块** — 遍历 S1~S8 训练文件，提取 MFCC，LBG 生成码本，保存为 `models/s*.pkl` |
| `test.py` | **识别模块** — 对测试文件提取 MFCC，与所有码本匹配，输出识别结果和准确率 |
| `models/` | **模型目录** — 存放训练生成的 8 个码本文件（`s1.pkl` ~ `s8.pkl`） |

---

## 4. 移植到其他电脑

### 4.1 需要拷贝的文件

```
D:\JIEDAN\          ← 整个项目目录拷贝过去
D:\data\            ← 音频数据目录拷贝过去（或在目标电脑重新准备）
```

### 4.2 目标电脑需要安装

```bash
pip install numpy scipy
```

Tkinter 随 Python 自带，无需额外安装。

### 4.3 需要修改的路径

如果目标电脑的目录结构不同，需修改以下文件中的路径：

| 文件 | 需修改的内容 |
|------|-------------|
| `gui.py` | `DEFAULT` 字典里的 `train_dir`、`test_dir`、`model_dir`（界面里也可以直接浏览选择） |
| `main.py` | `TRAIN_DIR`、`TEST_DIR`、`MODEL_DIR` |
| `train.py` | `__main__` 底部的默认路径 |
| `test.py` | `__main__` 底部的默认路径 |

### 4.4 运行

```bash
cd D:\JIEDAN
python gui.py        # 图形界面
python main.py       # 命令行
```
