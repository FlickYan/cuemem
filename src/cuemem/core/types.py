"""几处共用的取值范围。放在不依赖向量库、模型客户端的模块里：命令行检查参数时只导入这里。

原来定义它们的模块（memory、read.controller、api.service、llm.client）从这里导入，原来的导入
路径照旧可用。
"""

from __future__ import annotations

from typing import Literal

AskMode = Literal["l0", "l1", "l2", "auto", "agent", "hatch", "facts", "naive"]
RecallMode = Literal["l0", "l1", "l2", "auto"]
Router = Literal["learned", "hybrid", "rule", "llm"]
Scenario = Literal["shop", "plain"]
PeakPolicy = Literal["wait", "warn", "allow"]
