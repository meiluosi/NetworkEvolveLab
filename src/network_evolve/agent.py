from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Any, Optional

@dataclass
class EvolvingAgent:
    """
    具备语义信念与异质原型的可演化智能体
    """
    agent_id: int
    archetype: str  # "hub_broadcaster", "active_bridger", "long_tail_responder"
    belief: float   # 连续信念区间 [-1.0, 1.0]
    tolerance: float = 0.5  # 认知失调容忍度
    damping: float = 0.1    # 信念更新阻尼系数
    memory: List[Dict[str, Any]] = field(default_factory=list)

    def perceive_and_react(self, speaker_belief: float, speaker_archetype: str) -> Dict[str, Any]:
        """
        根据发言者信念做出微观反应：
        1. 计算语义差异与共鸣度
        2. 产生信念位移（带阻尼）
        3. 做出拓扑关系决策 (reinforce, decay, unfollow)
        """
        diff = abs(self.belief - speaker_belief)
        agreement = 1.0 - (diff / 2.0)  # [0, 1] 共鸣度
        
        # 信念更新（往发言者方向位移，但受自身容忍度和原型约束）
        if diff <= self.tolerance * 2.0:
            # 正常拉近
            shift = (speaker_belief - self.belief) * self.damping * agreement
            # 逐位等价于 float(np.clip(x, -1.0, 1.0))，但避免标量 numpy 调用开销
            # （在大 N 下 np.clip 的调度开销占引擎单轮耗时约 20%）。
            # 写法保留 NaN 传播语义（NaN 与 -1.0/1.0 的比较均为 False）。
            new_b = self.belief + shift
            if new_b > 1.0:
                new_b = 1.0
            elif new_b < -1.0:
                new_b = -1.0
            self.belief = float(new_b)
            action = "reinforce"
            weight_delta = 0.05
        else:
            # 认知失调激化（逆火或疏离）
            action = "decay"
            weight_delta = -0.15

        return {
            "agent_id": self.agent_id,
            "speaker_belief": speaker_belief,
            "new_belief": self.belief,
            "agreement": agreement,
            "action": action,
            "weight_delta": weight_delta
        }
