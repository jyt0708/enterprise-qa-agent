| 文档 | 内容安排 | 验证目的 |
| ----- | ----- | ----- |
| D01 |	“报销申请应在费用发生后 30 天内提交。” | 冲突的一方 |
| D02 |  “报销申请应在费用发生后 15 天内提交。” | 冲突的另一方 |
| D03 |	“年假申请应提前 3 个工作日提交。” | 明确答案 |
| D04 | “年假申请须由直属主管审批。” | 上下文追问 |
| D05 | 远程办公申请规则 | 其他正常问题 |
| D06 | 设备申领流程 | 检索干扰项 |


## 说明
D01 和 D02 都有版本日期，但**故意不提供适用范围、替代关系或优先级。**

这样可以检查 Agent 是否错误地推断：D02 日期上更新，所以一定采用 15 天。


## 放入项目

将本包的 data/ 和 examples/cases/ 合并到项目同名目录。

```python
import json
from pathlib import Path

documents = json.loads(
    Path("data/documents.json").read_text(encoding="utf-8")
)
print(len(documents))  # 6
```