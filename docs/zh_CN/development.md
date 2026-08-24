# 开发与文档维护

<div class="language-switcher"><a href="../en/development.html">English</a> · 简体中文</div>

## 仓库结构

- 公共头文件：`include/holistic_motion/`
- C++ 实现：`src/`
- Python 绑定：`pybind/module.cpp`
- Python 包：`python/holistic_motion/`
- C++ 测试：`tests/cpp/`
- Python 测试：`tests/python/`
- Conan 消费端测试：`test_package/`

## 构建文档

```bash
python -m pip install '.[docs]'
./scripts/docs.sh
```

构建会把 Sphinx 警告和无效 API 引用视为错误。生成文件统一位于
`docs/_build/`，不得提交。

修改公共行为时：

1. 更新相关公共头文件注释或 Python docstring。
2. 添加或更新聚焦测试。
3. 同步修改英文和中文指南。
4. 运行 `./scripts/docs.sh` 及受影响的测试套件。
