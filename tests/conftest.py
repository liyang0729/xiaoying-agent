"""pytest 的约定文件：pytest 启动时会自动先执行它。

本项目代码在 src/ 目录，而 pytest 默认只把"测试文件所在目录"
加入模块搜索路径。这里显式把项目根目录下的 src/ 插入 sys.path，
测试文件里才能直接 `from agent import ...`。
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))