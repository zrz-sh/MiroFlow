export http_proxy=http://127.0.0.1:7981
export https_proxy=http://127.0.0.1:7981
uv run main.py common-benchmark \
  --config_file_name=agent_gaia-validation-text-only \
  benchmark.execution.max_concurrent=20 \
  output_dir="logs/gaia-validation-text-only/$(date +"%Y%m%d_%H%M")" \
  # benchmark.execution.max_tasks=1 \