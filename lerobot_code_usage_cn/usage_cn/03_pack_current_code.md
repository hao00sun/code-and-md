# 打包当前代码和中文文档为 zip

本文说明如何把当前 LeRobot 代码和中文 md 文档压缩成一个 zip，方便备份或迁移到另一台机器。

目标：

- 包含当前仓库代码。
- 包含 `usage_cn/` 中文文档。
- 不包含 `.venv`。
- 不包含 `.git`。
- 不包含训练输出、大数据集、大权重、缓存、日志等重文件。
- 不包含敏感 token 文件。

## 1. 推荐打包命令

在仓库根目录执行：

```bash
cd /home/wu/lerobot_space/lerobot
```

创建 zip：

```bash
ZIP_NAME="/tmp/lerobot_code_docs_$(date +%Y%m%d_%H%M%S).zip"

zip -r "$ZIP_NAME" . \
  -x ".git/*" \
  -x ".venv/*" \
  -x "__pycache__/*" \
  -x "*/__pycache__/*" \
  -x ".pytest_cache/*" \
  -x ".mypy_cache/*" \
  -x ".ruff_cache/*" \
  -x "outputs/*" \
  -x "tests/outputs/*" \
  -x "*.pyc" \
  -x "*.pyo" \
  -x "*.log" \
  -x "token_hf.json"

echo "$ZIP_NAME"
```

生成文件类似：

```text
/tmp/lerobot_code_docs_20260814_160000.zip
```

## 2. 为什么要排除这些内容

`.venv/`：

```text
本地 Python 虚拟环境，当前约 6.9G，可以在新机器上重新创建。
```

`.git/`：

```text
git 历史目录，体积较大；如果只迁移代码快照，不需要它。
```

`outputs/`、`tests/outputs/`：

```text
训练或测试输出，不适合作为代码包的一部分。
```

`token_hf.json`：

```text
可能包含 Hugging Face token，属于敏感文件，不应打包分享。
```

大数据集和权重一般不在仓库目录内，例如：

```text
/data/SUN_ht/datasets
/data/SUN_ht/pi/pretrained_weights
/data/SUN_ht/pi/cache
/mnt/bigdata/SUN_ht/runs
/media/wu/data/SUN_ht
```

这些路径不会被上述 zip 命令包含，因为它们不在当前仓库目录下。

## 3. 检查 zip 里是否误包含 .venv

查看 zip 内容：

```bash
unzip -l "$ZIP_NAME" | head -50
```

确认没有 `.venv`：

```bash
unzip -l "$ZIP_NAME" | grep -E '(^|/)\.venv/' || echo "OK: no .venv"
```

确认没有 `.git`：

```bash
unzip -l "$ZIP_NAME" | grep -E '(^|/)\.git/' || echo "OK: no .git"
```

确认没有 token：

```bash
unzip -l "$ZIP_NAME" | grep 'token_hf.json' || echo "OK: no token_hf.json"
```

确认中文文档已包含：

```bash
unzip -l "$ZIP_NAME" | grep 'usage_cn'
```

## 4. 解压使用方法

在新机器上：

```bash
mkdir -p /home/wu/lerobot_space/lerobot_from_zip
cd /home/wu/lerobot_space/lerobot_from_zip
unzip /path/to/lerobot_code_docs_*.zip
```

进入目录：

```bash
cd /home/wu/lerobot_space/lerobot_from_zip
```

先读中文说明：

```bash
less usage_cn/README.md
```

然后按从零环境文档搭环境：

```bash
less usage_cn/01_from_zero_setup.md
```

## 5. 如果想把 zip 放到当前目录

也可以直接输出到仓库上一级：

```bash
cd /home/wu/lerobot_space/lerobot
ZIP_NAME="../lerobot_code_docs_$(date +%Y%m%d_%H%M%S).zip"

zip -r "$ZIP_NAME" . \
  -x ".git/*" \
  -x ".venv/*" \
  -x "__pycache__/*" \
  -x "*/__pycache__/*" \
  -x ".pytest_cache/*" \
  -x ".mypy_cache/*" \
  -x ".ruff_cache/*" \
  -x "outputs/*" \
  -x "tests/outputs/*" \
  -x "*.pyc" \
  -x "*.pyo" \
  -x "*.log" \
  -x "token_hf.json"
```

生成位置类似：

```text
/home/wu/lerobot_space/lerobot_code_docs_20260814_160000.zip
```

## 6. 如果要包含 git 历史

如果接收方需要完整 git 历史，不建议用普通 zip，而建议：

```bash
git remote -v
git status
```

然后把当前分支 push 到远程仓库，让对方：

```bash
git clone <仓库地址>
```

如果必须离线迁移完整 git 仓库，可以用：

```bash
cd /home/wu/lerobot_space
tar --exclude='lerobot/.venv' \
    --exclude='lerobot/__pycache__' \
    -czf lerobot_with_git_$(date +%Y%m%d_%H%M%S).tar.gz \
    lerobot
```

但这种方式会包含 `.git`，包会更大，也可能包含本地 git 信息。

## 7. 打包前建议检查

查看当前改动：

```bash
git status --short
```

查看仓库一层大小：

```bash
du -sh -- * .[!.]* 2>/dev/null | sort -hr
```

查看 zip 大小：

```bash
du -sh "$ZIP_NAME"
```

如果 zip 仍然很大，可以列出最大文件：

```bash
unzip -l "$ZIP_NAME" | sort -k1,1nr | head -50
```

根据结果继续在 `zip -x` 中增加排除规则。
