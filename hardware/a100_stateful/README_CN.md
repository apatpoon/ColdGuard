# ColdGuard A100四卡有状态多模型竞争实验包 v2

四张A100各自运行一个独立的六模型竞争四槽位资源池。五种策略采用5×4循环拉丁矩形轮换，三个时间块、四个请求顺序种子，共60次正式回放。四卡会并行使用，但单次资源竞争绝不跨卡拆分。

## 启动

```bash
cd /workspace/ColdGuard_A100_stateful_competition_v2

tmux new-session -d -s coldguard-stateful-v2 \
  'bash /workspace/ColdGuard_A100_stateful_competition_v2/scripts/run_all.sh'
```

## 状态

```bash
watch -n 5 \
  'bash /workspace/ColdGuard_A100_stateful_competition_v2/scripts/status.sh'
```

```bash
tmux attach -t coldguard-stateful-v2
```

退出tmux但保持运行：`Ctrl+B`，松开后按`D`。

## 优雅停止

```bash
tmux send-keys -t coldguard-stateful-v2 C-c
```

脚本收到信号后通知四个工作线程清理各自的vLLM进程。确认显存释放后再关闭会话：

```bash
nvidia-smi
tmux kill-session -t coldguard-stateful-v2 2>/dev/null || true
```

## 完成标准

```bash
RESULT="$(cat /workspace/coldguard-a100-stateful-results-v2/LAST_RESULT_DIR.txt)"
jq . "$RESULT/summary/evidence_audit.json"
```

必须为`"complete": true`，且`planned_runs`与`observed_runs`均为60。脚本不会自动停止Pod。
