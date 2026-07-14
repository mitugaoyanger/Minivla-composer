"""配置文件读取工具。"""

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def _parse_scalar(value: str) -> Any:
    """解析简易 YAML 标量，支持 bool、int、float 和字符串。"""
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text.strip("\"'")


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """在未安装 PyYAML 时读取本项目使用的平铺键值配置。"""
    cfg: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        cfg[key.strip()] = _parse_scalar(value)
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置并返回字典。"""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    if yaml is None:
        return _load_simple_yaml(text)
    return yaml.safe_load(text) or {}
