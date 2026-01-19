import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from dotenv import load_dotenv

async def run_job(python_exe, runner_script, item_id, log_file, json_path=None, label_key=None):
    """Run a single evaluation job."""
    print(f"🚀 提交任务: item {item_id}")
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    cmd = [python_exe, runner_script, "--item_id", str(item_id), ""]
    if json_path:
        cmd.extend(["--json_path", str(json_path)])
    if label_key:
        cmd.extend(["--label_key", str(label_key)])
    
    env = os.environ.copy()
    
    with open(log_file, "w") as f:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env
        )
        await process.wait()
    return item_id

async def main():
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment
    max_jobs = int(os.getenv("MAX_JOBS", 3))
    agent_cases_dir_str = os.getenv("AGENT_CASES_DIR", "cases/agent_cases")
    agent_scripts_dir_str = os.getenv("AGENT_SCRIPTS_DIR", "src/agent_evaluator/experiments/threads")
    log_base_dir_str = os.getenv("LOG_BASE_DIR", "cases/logs")
    
    # Get evaluation type from command line
    if len(sys.argv) < 2:
        # Try to list available types from cases dir
        agent_cases_dir = Path(agent_cases_dir_str)
        if agent_cases_dir.exists():
            available = [f.stem for f in agent_cases_dir.glob("*.json")]
            print(f"Please specify evaluation type...[{', '.join(available)}]")
        else:
            print("Please specify evaluation type.")
        sys.exit(1)
        
    eval_type = sys.argv[1]
    agent_cases_dir = Path(agent_cases_dir_str)
    log_base_dir = Path(log_base_dir_str)
    
    # 使用通用的 runner 脚本，而不是每个 agent 文件夹下的 _bash.py
    runner_script = Path(__file__).parent / "base" / "runner.py"
    
    json_path = agent_cases_dir / f"{eval_type}.json"
    # 日志输出路径：LOG_BASE_DIR / eval_type / logs
    logs_dir = log_base_dir / eval_type
    
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        sys.exit(1)
    if not runner_script.exists():
        print(f"Error: Runner script not found at {runner_script}")
        sys.exit(1)
        
    # Read total count from JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            total = len(dataset)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)
        
    print(f"总数据量: {total}")
    
    # Get python executable (prefer virtual env)
    root_dir_str = os.getenv("ROOT_DIR", ".")
    root_dir = Path(root_dir_str).absolute()
    if os.name == "nt": # Windows
        python_exe = root_dir / ".venv" / "Scripts" / "python.exe"
    else: # Unix
        python_exe = root_dir / ".venv" / "bin" / "python"
        
    if not python_exe.exists():
        python_exe = sys.executable # Fallback to current python
        
    # Run jobs with concurrency limit
    semaphore = asyncio.Semaphore(max_jobs)
    
    async def sem_run_job(item_id):
        async with semaphore:
            log_file = logs_dir / f"item_{item_id}.log"
            await run_job(
                str(python_exe), 
                str(runner_script), 
                item_id, 
                str(log_file), 
                json_path=str(json_path),
                label_key=eval_type
            )
            # Sleep a bit between submissions as in run.sh
            await asyncio.sleep(3)

    tasks = [sem_run_job(i) for i in range(total)]
    await asyncio.gather(*tasks)
    
    print("✅ 所有任务完成")

if __name__ == "__main__":
    asyncio.run(main())
