"""跨平台命令行输出设置。"""

import sys


def configure_utf8_console() -> None:
    """避免 Windows 中文终端使用旧编码时打印失败或乱码。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
