uv run main.py common-benchmark \
  --config_file_name=agent_gaia-validation-text-only \
  benchmark.execution.max_tasks=30 \
  benchmark.execution.max_concurrent=10 \
  output_dir="logs/gaia-validation-text-only/gpt-4o_30tasks_$(date +"%Y%m%d_%H%M")"