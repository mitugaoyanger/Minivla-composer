"""根据任务生成固定子目标列表。"""

def generate_subgoals(_instruction: str) -> list[str]:
    """返回专家执行时使用的高层子目标名称。"""
    return ["move_to_object", "grasp_object", "move_to_target", "release_object"]
