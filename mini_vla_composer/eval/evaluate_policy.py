"""兼容旧导入路径；评估实现位于 :mod:`evaluator_v2`。"""

from mini_vla_composer.eval.evaluator_v2 import evaluate_policy

__all__ = ["evaluate_policy"]
