# Vibe Coding for DJI Mic

通过敲击 DJI Mic 2 麦克风外壳，实现免手操作的快捷控制。

> DJI Mic 2 的实体按键无法被 macOS 识别为 HID/键盘事件。本方案通过音频暗号（敲击冲击脉冲）绕过这个限制。

## 操作方式

| 敲击次数 | 动作 |
|---------|------|
| 1 下 | 输入 "Yes" |
| 2 下 | 切换豆包语音输入（按住/松开右 Option） |
| 3 下 | 输入 "No" |

## 快速开始

### 1. 安装依赖

```bash
cd dji-mic-vibecoding
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 查看音频设备

```bash
python vibecoding_controller.py --list
```

### 3. 运行

```bash
python vibecoding_controller.py --device <设备号>
```

## 技术原理

```
DJI Mic 2 音频流 (16kHz PCM)
        ↓
   信号处理 (NumPy)
   ├─ RMS → 音量基线
   ├─ PEAK → 峰值检测
   └─ RATIO = PEAK / RMS → 区分敲击与说话
        ↓
   敲击检测
   ├─ 绝对死线：PEAK > 15000, RATIO > 8.0
   ├─ 训练阈值：PEAK > 18000, RATIO > 10.0
   ├─ 去抖动：130ms
   └─ 时间窗：400ms
        ↓
   动作执行 (pynput)
   ├─ 剪贴板粘贴（绕过中文输入法）
   └─ 模拟按键（Option 键控制语音）
```

## 语音输入状态保护

- 语音输入期间，只有 2 下敲击能关闭语音
- 开启语音后 1.5 秒内禁止关闭（防止说话时误触）
- 1 下/3 下在语音期间视为爆音，直接忽略

## 环境要求

- macOS（Apple Silicon + macOS 14+）
- DJI Mic 2 接收器通过 USB-C 连接 Mac
- 终端需要辅助功能权限

## 文件结构

```
dji-mic-vibecoding/
├── vibecoding_controller.py    # 主程序
├── requirements.txt            # Python 依赖
├── run.sh                      # 启动脚本
└── README.md                   # 项目文档
```

## License

MIT
