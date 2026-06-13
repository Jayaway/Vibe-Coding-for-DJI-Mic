#!/usr/bin/env python3
"""
DJI Mic 2 Vibecoding Controller

因为 DJI Mic 2 的三个实体按键不会被 macOS 识别为 HID/键盘事件，
所以本脚本采用"音频暗号"方案：通过识别敲击麦克风外壳产生的冲击脉冲
来实现免手操作：
    - 单次敲击 -> 输入 Yes
    - 连续两次敲击 -> 切换豆包语音输入（自动按住/松开右 Option）
    - 连续三次敲击 -> 输入 No

首次使用请先运行训练模式：python vibecoding_controller.py --train

使用前请先把 DJI Mic 2 接收器通过 USB-C 接到 Mac，并在系统设置里
把"DJI Mic 2"设为默认输入设备。
"""

import argparse
import json
import time
import subprocess
import threading
from pathlib import Path
from collections import deque

import numpy as np
import pyaudio
from pynput.keyboard import Controller, Key


# ========================== 默认参数（训练后会被覆盖） ==========================
SAMPLE_RATE = 16000
CHUNK = 512
FORMAT = pyaudio.paInt16

DEFAULT_TAP_PEAK_THRESHOLD = 18000
DEFAULT_TAP_RATIO_MIN = 10.0

# 绝对死线：低于此值的声音一律视为噪音
HARD_PEAK_FLOOR = 15000
HARD_RATIO_FLOOR = 8.0

DEBOUNCE_MS = 130            # 单次敲击防抖，防止外壳余震算成 2 下
TAP_WINDOW_MS = 400          # 组合拳收集时间窗，敲完 3 下的黄金窗口
VOICE_TOGGLE_TAPS = 2
YES_TAPS = 1
NO_TAPS = 3

AUTO_STOP_VOICE_MS = 3000

TEXT_YES = "Yes"
TEXT_NO = "No"
VOICE_KEY = Key.alt_r

CONFIG_FILE = Path(__file__).parent / "tap_profile.json"
# =============================================================


def load_config():
    """加载训练好的配置"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
            print(f"[加载配置] 从 {CONFIG_FILE.name} 读取训练数据")
            return cfg
        except Exception as e:
            print(f"[警告] 配置文件损坏: {e}")
    return None


def save_config(profile):
    """保存训练配置"""
    with open(CONFIG_FILE, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"[保存配置] 已保存到 {CONFIG_FILE.name}")


class TapTrainer:
    """训练模式：收集用户敲击数据，计算最佳阈值"""

    def __init__(self, device_index=None):
        self.device_index = device_index
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def find_device_index(self):
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            name = info.get("name", "")
            if "DJI" in name or "Mic" in name or "mic" in name:
                if info.get("maxInputChannels", 0) > 0:
                    return i
        default = self.audio.get_default_input_device_info()
        return default["index"]

    def open_stream(self):
        idx = self.device_index if self.device_index is not None else self.find_device_index()
        self.stream = self.audio.open(
            format=FORMAT, channels=1, rate=SAMPLE_RATE,
            input=True, input_device_index=idx, frames_per_buffer=CHUNK,
        )

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def collect_samples(self, duration_sec=5):
        """收集指定时长的音频数据，返回 (rms_list, peak_list, ratio_list)"""
        rms_list = []
        peak_list = []
        ratio_list = []
        rms_window = deque(maxlen=20)

        start = time.time()
        while time.time() - start < duration_sec:
            data = np.frombuffer(self.stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
            peak = float(np.max(np.abs(data)))

            rms_window.append(rms)
            baseline = np.median(list(rms_window)) if rms_window else 1.0
            ratio = peak / max(baseline, 1.0)

            rms_list.append(rms)
            peak_list.append(peak)
            ratio_list.append(ratio)

            time.sleep(0.03)

        return rms_list, peak_list, ratio_list

    def collect_tap_samples(self, tap_count, rounds=3):
        """让用户练习敲 N 下，收集多轮数据"""
        all_peaks = []
        all_ratios = []

        for r in range(rounds):
            input(f"\n  第 {r+1}/{rounds} 轮：请准备好，按回车后开始...")
            print(f"  >>> 请连续敲 {tap_count} 下 <<<")
            _, peaks, ratios = self.collect_samples(duration_sec=4)

            # 提取敲击峰值（取最高的几个）
            sorted_peaks = sorted(enumerate(peaks), key=lambda x: x[1], reverse=True)
            # 取 top 20% 作为敲击样本
            top_n = max(tap_count, len(peaks) // 5)
            tap_peaks = [peaks[i] for i, _ in sorted_peaks[:top_n]]
            tap_ratios = [ratios[i] for i, _ in sorted_peaks[:top_n]]

            avg_peak = np.mean(tap_peaks)
            avg_ratio = np.mean(tap_ratios)
            print(f"  结果：平均 PEAK={avg_peak:.0f}, 平均 RATIO={avg_ratio:.1f}")

            all_peaks.extend(tap_peaks)
            all_ratios.extend(tap_ratios)

        return all_peaks, all_ratios

    def collect_silence_samples(self, rounds=3):
        """收集静默/握持噪音样本"""
        all_peaks = []
        all_ratios = []

        for r in range(rounds):
            input(f"\n  第 {r+1}/{rounds} 轮：请保持安静，正常握持麦克风，按回车开始...")
            print("  >>> 保持 4 秒不动 <<<")
            _, peaks, ratios = self.collect_samples(duration_sec=4)
            all_peaks.extend(peaks)
            all_ratios.extend(ratios)

        return all_peaks, all_ratios

    def train(self):
        """完整的训练流程"""
        print("=" * 60)
        print("  DJI Mic 2 Vibecoding 训练模式")
        print("=" * 60)
        print()
        print("训练目标：根据你的敲击力度，自动计算最佳检测阈值")
        print()
        print("注意事项：")
        print("  1. 保持和正常使用时一样的握持姿势")
        print("  2. 用和正常使用时一样的力度敲击")
        print("  3. 每轮训练前会提示你准备")
        print()

        self.open_stream()

        try:
            # 第一步：收集静默/握持噪音
            print("-" * 60)
            print("步骤 1/4：收集握持噪音基线")
            print("-" * 60)
            silence_peaks, silence_ratios = self.collect_silence_samples(rounds=3)
            silence_peak_p95 = np.percentile(silence_peaks, 95)
            silence_ratio_p95 = np.percentile(silence_ratios, 95)
            print(f"\n  握持噪音：PEAK P95={silence_peak_p95:.0f}, RATIO P95={silence_ratio_p95:.1f}")

            # 第二步：收集单次敲击
            print("\n" + "-" * 60)
            print("步骤 2/4：训练单次敲击")
            print("-" * 60)
            tap1_peaks, tap1_ratios = self.collect_tap_samples(tap_count=1, rounds=3)
            tap1_peak_min = np.percentile(tap1_peaks, 10)
            tap1_ratio_min = np.percentile(tap1_ratios, 10)
            print(f"\n  单次敲击：PEAK 范围={np.min(tap1_peaks):.0f}-{np.max(tap1_peaks):.0f}, "
                  f"RATIO 范围={np.min(tap1_ratios):.1f}-{np.max(tap1_ratios):.1f}")

            # 第三步：收集连续两次敲击
            print("\n" + "-" * 60)
            print("步骤 3/4：训练连续两次敲击")
            print("-" * 60)
            tap2_peaks, tap2_ratios = self.collect_tap_samples(tap_count=2, rounds=3)
            print(f"\n  两次敲击：PEAK 范围={np.min(tap2_peaks):.0f}-{np.max(tap2_peaks):.0f}, "
                  f"RATIO 范围={np.min(tap2_ratios):.1f}-{np.max(tap2_ratios):.1f}")

            # 第四步：收集连续三次敲击
            print("\n" + "-" * 60)
            print("步骤 4/4：训练连续三次敲击")
            print("-" * 60)
            tap3_peaks, tap3_ratios = self.collect_tap_samples(tap_count=3, rounds=3)
            print(f"\n  三次敲击：PEAK 范围={np.min(tap3_peaks):.0f}-{np.max(tap3_peaks):.0f}, "
                  f"RATIO 范围={np.min(tap3_ratios):.1f}-{np.max(tap3_ratios):.1f}")

            # 计算阈值
            print("\n" + "=" * 60)
            print("  计算最佳阈值...")
            print("=" * 60)

            # 所有敲击样本
            all_tap_peaks = tap1_peaks + tap2_peaks + tap3_peaks
            all_tap_ratios = tap1_ratios + tap2_ratios + tap3_ratios

            # 阈值 = 握持噪音的上限 和 敲击样本的下限 之间取值
            # PEAK 阈值：取噪音 P99 和 最小敲击值 的中间
            peak_threshold = (silence_peak_p95 * 1.5 + np.percentile(all_tap_peaks, 5)) / 2
            # RATIO 阈值：取噪音 P99 和 最小敲击值 的中间
            ratio_threshold = (silence_ratio_p95 * 1.5 + np.percentile(all_tap_ratios, 5)) / 2

            # 确保阈值不低于最小值
            peak_threshold = max(peak_threshold, 500)
            ratio_threshold = max(ratio_threshold, 3.0)

            print(f"\n  训练结果：")
            print(f"    PEAK 阈值: {peak_threshold:.0f}")
            print(f"    RATIO 阈值: {ratio_threshold:.1f}")
            print()

            # 保存配置
            profile = {
                "tap_peak_threshold": round(peak_threshold),
                "tap_ratio_min": round(ratio_threshold, 1),
                "silence_peak_p95": round(silence_peak_p95),
                "silence_ratio_p95": round(silence_ratio_p95, 1),
                "tap1_peak_range": [round(np.min(tap1_peaks)), round(np.max(tap1_peaks))],
                "tap2_peak_range": [round(np.min(tap2_peaks)), round(np.max(tap2_peaks))],
                "tap3_peak_range": [round(np.min(tap3_peaks)), round(np.max(tap3_peaks))],
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            save_config(profile)

            print("\n" + "=" * 60)
            print("  训练完成！")
            print("  现在可以运行：python vibecoding_controller.py --device <设备号>")
            print("=" * 60)

        finally:
            self.close()


class VibecodingController:
    def __init__(self, device_index=None, debug=False, calibrate=False):
        self.device_index = device_index
        self.debug = debug
        self.calibrate = calibrate
        self.keyboard = Controller()

        self.audio = pyaudio.PyAudio()
        self.stream = None
        self._lock = threading.Lock()

        # 暴力硬编码阈值，彻底断开 tap_profile.json
        self.tap_peak_threshold = DEFAULT_TAP_PEAK_THRESHOLD
        self.tap_ratio_min = DEFAULT_TAP_RATIO_MIN

        # 敲击状态
        self._tap_count = 0
        self._last_tap_time = 0.0
        self._debounce_until = 0.0

        # 语音输入状态
        self._voice_active = False
        self._last_sound_time = 0.0
        self._voice_toggle_allowed_after = 0.0  # 语音切换冷冻期

        # 近期 RMS 滑动窗口
        self._rms_window = deque(maxlen=20)

    def find_device_index(self):
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            name = info.get("name", "")
            if "DJI" in name or "Mic" in name or "mic" in name:
                if info.get("maxInputChannels", 0) > 0:
                    return i
        default = self.audio.get_default_input_device_info()
        return default["index"]

    def open_stream(self):
        idx = self.device_index if self.device_index is not None else self.find_device_index()
        self.stream = self.audio.open(
            format=FORMAT, channels=1, rate=SAMPLE_RATE,
            input=True, input_device_index=idx, frames_per_buffer=CHUNK,
        )
        print("\n音频流已打开，开始监听敲击暗号...")
        print(f"  当前阈值：PEAK>{self.tap_peak_threshold}, RATIO>{self.tap_ratio_min}")
        print("  单次敲击 = Yes")
        print("  连续两次敲击 = 切换豆包语音输入")
        print("  连续三次敲击 = No")
        print("按 Ctrl+C 停止\n")

    def _rms(self, data):
        return float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))

    def _peak(self, data):
        return float(np.max(np.abs(data)))

    def _paste_text(self, text):
        """通过剪贴板粘贴文字，绕过输入法"""
        process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
        process.communicate(text.encode('utf-8'))
        time.sleep(0.15)
        self.keyboard.release(Key.cmd)
        time.sleep(0.02)
        self.keyboard.press(Key.cmd)
        self.keyboard.press('v')
        time.sleep(0.02)
        self.keyboard.release('v')
        self.keyboard.release(Key.cmd)

    def _perform_action(self, taps):
        # 【防误触锁】语音输入期间，只允许 2 下切换语音
        if self._voice_active:
            if taps == VOICE_TOGGLE_TAPS:
                self._toggle_voice()
            else:
                print(f"[忽略] 语音输入中，{taps} 次敲击视为爆音，就地销毁")
            return

        # 未开启语音状态下的正常逻辑
        if taps == YES_TAPS:
            print(f"[动作] 输入 {TEXT_YES}")
            self._paste_text(TEXT_YES)
        elif taps == VOICE_TOGGLE_TAPS:
            self._toggle_voice()
        elif taps == NO_TAPS:
            print(f"[动作] 输入 {TEXT_NO}")
            self._paste_text(TEXT_NO)
        else:
            print(f"[忽略] 未知暗号：{taps} 次敲击")

    def _toggle_voice(self):
        with self._lock:
            now = time.time()
            if self._voice_active:
                # 冷冻期内禁止关闭语音
                if now < self._voice_toggle_allowed_after:
                    print("[忽略] 还在说话保护期内，忽略疑似误触的关闭动作")
                    return
                print("[动作] 停止语音输入（松开右 Option）")
                self.keyboard.release(VOICE_KEY)
                self._voice_active = False
            else:
                print("[动作] 开始语音输入（按住右 Option）")
                self.keyboard.press(VOICE_KEY)
                self._voice_active = True
                self._last_sound_time = now
                # 开启语音后，未来 1.5 秒处于物理免疫状态，禁止关闭
                self._voice_toggle_allowed_after = now + 1.5

    def _process_audio_chunk(self, in_data):
        data = np.frombuffer(in_data, dtype=np.int16)
        rms = self._rms(data)
        peak = self._peak(data)

        now = time.time()

        self._rms_window.append(rms)
        baseline_rms = np.median(list(self._rms_window)) if self._rms_window else 1.0
        ratio = peak / max(baseline_rms, 1.0)

        if self.calibrate:
            print(f"RMS={rms:7.1f}  PEAK={peak:7.1f}  RATIO={ratio:5.1f}")
            return

        if self.debug:
            print(f"RMS={rms:7.1f}  PEAK={peak:7.1f}  RATIO={ratio:5.1f}  taps={self._tap_count}")

        # 语音输入期间跳过敲击检测
        if self._voice_active:
            if rms > 200:
                self._last_sound_time = now
            elif AUTO_STOP_VOICE_MS > 0 and (now - self._last_sound_time) * 1000 > AUTO_STOP_VOICE_MS:
                print("[自动] 长时间静音，停止语音输入")
                self._toggle_voice()

        # 敲击检测：绝对死线 + 训练阈值
        # 低于 HARD_PEAK_FLOOR 的声音一律视为噪音，看都不看
        is_tap = (
            peak > HARD_PEAK_FLOOR
            and peak > self.tap_peak_threshold
            and ratio > HARD_RATIO_FLOOR
            and ratio > self.tap_ratio_min
            and now > self._debounce_until
        )

        if is_tap:
            if now - self._last_tap_time > TAP_WINDOW_MS / 1000.0 + 0.05:
                if self._tap_count > 0:
                    self._perform_action(self._tap_count)
                self._tap_count = 1
            else:
                self._tap_count += 1

            self._last_tap_time = now
            self._debounce_until = now + DEBOUNCE_MS / 1000.0
            if not self.calibrate:
                print(f"[检测到敲击] 当前计数 {self._tap_count}")

        else:
            # 超过时间窗没有新敲击，结算上一组
            if self._tap_count > 0 and now - self._last_tap_time > TAP_WINDOW_MS / 1000.0 + 0.05:
                self._perform_action(self._tap_count)
                self._tap_count = 0

    def run(self):
        self.open_stream()
        try:
            while True:
                in_data = self.stream.read(CHUNK, exception_on_overflow=False)
                self._process_audio_chunk(in_data)
        except KeyboardInterrupt:
            print("\n正在停止...")
        finally:
            if self._voice_active:
                self.keyboard.release(VOICE_KEY)
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()


def list_audio_devices():
    pa = pyaudio.PyAudio()
    print("\n可用输入设备：")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            print(f"  {i}: {info['name']}")
    pa.terminate()


def main():
    parser = argparse.ArgumentParser(description="DJI Mic 2 音频暗号 Vibecoding 控制器")
    parser.add_argument("--device", type=int, default=None, help="指定音频设备索引")
    parser.add_argument("--list", action="store_true", help="列出可用输入设备")
    parser.add_argument("--debug", action="store_true", help="打印每帧 RMS/PEAK")
    parser.add_argument("--calibrate", action="store_true", help="校准模式：只打印 RMS/PEAK")
    parser.add_argument("--train", action="store_true", help="训练模式：根据你的敲击力度校准阈值")
    args = parser.parse_args()

    if args.list:
        list_audio_devices()
        return

    if args.train:
        trainer = TapTrainer(device_index=args.device)
        trainer.train()
        return

    controller = VibecodingController(
        device_index=args.device,
        debug=args.debug,
        calibrate=args.calibrate,
    )
    controller.run()


if __name__ == "__main__":
    main()
