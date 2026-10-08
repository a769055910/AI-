"""Upload tamper-model integration files and start remote setup jobs safely.

Connection defaults are read from the existing local deployment configuration;
credentials are never printed or accepted on the command line.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import paramiko


PROJECT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/forge-detector"


def _deployment_settings() -> dict[str, object]:
    local_config = PROJECT / "gpu.remote.local.json"
    if local_config.is_file():
        config = json.loads(local_config.read_text(encoding="utf-8"))
        settings = {name: os.environ.get(f"AUTODL_SSH_{name}", config.get(name)) for name in ("HOST", "PORT", "USER", "PASSWORD")}
        missing = [name for name, value in settings.items() if value is None or value == ""]
        if missing:
            raise RuntimeError(f"Missing deployment settings: {', '.join(missing)}")
        settings["PORT"] = int(settings["PORT"])
        return settings
    tree = ast.parse((PROJECT / "_fix_restart.py").read_text(encoding="utf-8"))
    settings: dict[str, object] = {}
    wanted = {"HOST", "PORT", "USER", "PASSWORD"}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if not isinstance(node.targets[0], ast.Name) or node.targets[0].id not in wanted:
            continue
        name = node.targets[0].id
        value = node.value
        # PORT is wrapped in int(os.environ.get(...)); unwrap it without
        # evaluating arbitrary source code.
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "int" and value.args:
            value = value.args[0]
        if isinstance(value, ast.Call) and len(value.args) >= 2 and isinstance(value.args[0], ast.Constant) and isinstance(value.args[1], ast.Constant):
            settings[name] = os.environ.get(str(value.args[0].value), value.args[1].value)
        elif isinstance(value, ast.Constant):
            settings[name] = value.value
    missing = wanted - settings.keys()
    if missing:
        raise RuntimeError(f"Missing deployment settings: {', '.join(sorted(missing))}")
    settings["PORT"] = int(settings["PORT"])
    return settings


def _connect() -> paramiko.SSHClient:
    config = _deployment_settings()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=str(config["HOST"]), port=int(config["PORT"]),
        username=str(config["USER"]), password=str(config["PASSWORD"]), timeout=20,
    )
    return client


def _run(client: paramiko.SSHClient, command: str, stdin_text: str | None = None) -> str:
    stdin, stdout, stderr = client.exec_command(command, timeout=30)
    if stdin_text is not None:
        stdin.write(stdin_text)
        stdin.flush()
        stdin.channel.shutdown_write()
    output = stdout.read().decode("utf-8", "replace").strip()
    error = stderr.read().decode("utf-8", "replace").strip()
    if error:
        output = f"{output}\n{error}".strip()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-install", action="store_true", help="start background source/model setup jobs")
    parser.add_argument("--start-adaifl", action="store_true", help="start only the AdaIFL setup job")
    parser.add_argument("--start-trufor", action="store_true", help="start only the TruFor setup job")
    parser.add_argument("--status", action="store_true", help="show non-sensitive remote setup status")
    parser.add_argument("--probe", action="store_true", help="inspect remote runtime prerequisites")
    parser.add_argument("--adaifl-disk", action="store_true", help="inspect AdaIFL install disk usage")
    parser.add_argument("--clean-adaifl-cache", action="store_true", help="remove only the failed AdaIFL PyTorch cache archive")
    parser.add_argument("--stop-adaifl-setup", action="store_true", help="stop only the stalled AdaIFL setup job")
    parser.add_argument("--inspect-trufor", action="store_true", help="inspect TruFor inference imports")
    parser.add_argument("--stop-trufor-setup", action="store_true", help="stop only a running TruFor setup job")
    parser.add_argument("--finalize-trufor", action="store_true", help="normalize the official TruFor weight path")
    parser.add_argument("--verify-trufor", action="store_true", help="run one remote TruFor inference smoke test")
    parser.add_argument("--verify-adaifl", action="store_true", help="exercise the deployed AdaIFL HTTP endpoint")
    parser.add_argument("--restart-gpu", action="store_true", help="restart GPU service with the uploaded integration")
    parser.add_argument("--health", action="store_true", help="query GPU service health after deployment")
    parser.add_argument("--verify-service", action="store_true", help="exercise deployed TruFor HTTP endpoint")
    parser.add_argument("--gpu-log", action="store_true", help="show recent GPU service log lines")
    parser.add_argument("--port-status", action="store_true", help="show the process holding GPU port 6006")
    parser.add_argument("--bind-test", action="store_true", help="test whether the GPU namespace can bind port 6006")
    parser.add_argument("--foreground-gpu-smoke", action="store_true", help="run the GPU service briefly in foreground for startup diagnosis")
    parser.add_argument("--service-procs", action="store_true", help="list likely process supervisors and Python services")
    parser.add_argument("--storage-audit", action="store_true", help="report large model/cache locations before migration")
    parser.add_argument("--migrate-data-disk", action="store_true", help="move models/caches to AutoDL data disk and install AdaIFL")
    args = parser.parse_args()
    client = _connect()
    try:
        _run(client, f"mkdir -p {REMOTE_ROOT}/scripts {REMOTE_ROOT}/logs")
        with client.open_sftp() as sftp:
            sftp.put(str(PROJECT / "gpu_server.py"), f"{REMOTE_ROOT}/gpu_server.py")
            sftp.put(str(PROJECT / "trufor_worker.py"), f"{REMOTE_ROOT}/trufor_worker.py")
            for name in ("setup_trufor_remote.sh", "setup_adaifl_remote.sh"):
                sftp.put(str(PROJECT / "scripts" / name), f"{REMOTE_ROOT}/scripts/{name}")
        _run(client, f"chmod 700 {REMOTE_ROOT}/scripts/setup_trufor_remote.sh {REMOTE_ROOT}/scripts/setup_adaifl_remote.sh")
        print("Uploaded gpu_server.py and model setup scripts.")
        if args.status:
            print(_run(client, f"ps -ef | grep -E 'setup_(trufor|adaifl)_remote' | grep -v grep || true"))
            print(_run(client, "ps -ef | grep -E 'conda|pip|wget|gdown' | grep -v grep || true"))
            print("--- TruFor setup log ---")
            print(_run(client, f"tail -n 20 {REMOTE_ROOT}/logs/trufor_setup.log 2>/dev/null || true"))
            print("--- AdaIFL setup log ---")
            print(_run(client, f"tail -n 20 {REMOTE_ROOT}/logs/adaifl_setup.log 2>/dev/null || true"))
        if args.probe:
            print(_run(client, "command -v docker || true; command -v conda || true; command -v python || true; ls -d /root/miniconda3 /opt/conda 2>/dev/null || true; /root/miniconda3/bin/python -c 'import sys,torch,cv2,torchvision; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available(), cv2.__version__, torchvision.__version__)' 2>/dev/null || true; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true"))
        if args.adaifl_disk:
            print(_run(client, "df -h /root/miniconda3 /root/forge-detector; ls -lh /root/miniconda3/pkgs/pytorch-2.0.1-py3.8_cuda11.8_cudnn8.7.0_0.tar.bz2 2>/dev/null || true"))
        if args.clean_adaifl_cache:
            failed_archive = "/root/miniconda3/pkgs/pytorch-2.0.1-py3.8_cuda11.8_cudnn8.7.0_0.tar.bz2"
            print(_run(client, f"rm -f {failed_archive}; df -h /root/miniconda3"))
        if args.stop_adaifl_setup:
            print(_run(client, "pkill -f '[s]etup_adaifl_remote.sh|[g]down.*187SJ_O0YHP0DVBXgfob_o2BofzCf0TMP' || true"))
            print("Stopped the stalled AdaIFL setup job.")
        if args.inspect_trufor:
            print(_run(client, f"grep -R -nE '^(import|from) (mmcv|jpegio)' {REMOTE_ROOT}/vendor/TruFor/TruFor_train_test {REMOTE_ROOT}/vendor/TruFor/lib 2>/dev/null | head -n 50 || true"))
        if args.stop_trufor_setup:
            print(_run(client, "pkill -f 'setup_trufor_remote.sh|conda run -n trufor python -m pip' || true"))
            print("Stopped the active TruFor setup job.")
        if args.finalize_trufor:
            print(_run(client, f"cd {REMOTE_ROOT}/vendor/TruFor/TruFor_train_test/pretrained_models && [ -f weights/trufor.pth.tar ] && ln -sfn weights/trufor.pth.tar trufor.pth.tar && ls -l trufor.pth.tar"))
        if args.verify_trufor:
            remote_sample = f"{REMOTE_ROOT}/tmp_trufor_smoke.jpg"
            remote_output = f"{REMOTE_ROOT}/tmp_trufor_smoke.npz"
            with client.open_sftp() as sftp:
                sample = next((PROJECT / "验收测试集" / "05_篡改真实_CASIA1").glob("*.jpg"))
                sftp.put(str(sample), remote_sample)
            command = (
                f"cd {REMOTE_ROOT}/vendor/TruFor/TruFor_train_test && "
                f"/root/miniconda3/envs/trufor/bin/python test.py -g 0 -in {remote_sample} -out {remote_output} "
                f"-exp trufor_ph3 TEST.MODEL_FILE pretrained_models/trufor.pth.tar && "
                f"/root/miniconda3/envs/trufor/bin/python -c \"import numpy as n; x=n.load('{remote_output}'); print('score=',float(x['score']), 'map=',x['map'].shape, 'conf=',x['conf'].shape)\""
            )
            print(_run(client, command))
        if args.restart_gpu:
            restart = (
                "screen -S forge_gpu -X quit >/dev/null 2>&1 || true; "
                "pkill -f '[t]rufor_worker.py' >/dev/null 2>&1 || true; "
                "sleep 2; "
                f"cd {REMOTE_ROOT} && screen -dmS forge_gpu bash -lc "
                f"'exec env PRELOAD_MODELS=0 HF_HOME=/root/autodl-tmp/forge-detector-data/huggingface "
                f"ADAIFL_MODEL_PATH=/root/autodl-tmp/forge-detector-data/models/AdaIFL/AdaIFL_v0.pth "
                f"/root/miniconda3/bin/python gpu_server.py "
                f"--host 0.0.0.0 --port 6006 >> logs/gpu_server.log 2>&1'"
            )
            _run(client, restart)
            print("Restart requested; use --status after a short delay to inspect setup logs.")
        if args.health:
            print(_run(client, "curl -fsS --max-time 10 http://127.0.0.1:6006/health || true"))
        if args.verify_service:
            response = f"{REMOTE_ROOT}/tmp_trufor_http.json"
            command = (
                f"curl -fsS --max-time 240 -F file=@{REMOTE_ROOT}/tmp_trufor_smoke.jpg "
                f"http://127.0.0.1:6006/trufor_analyze > {response} && "
                f"/root/miniconda3/bin/python -c \"import json; x=json.load(open('{response}'))['data']; "
                f"print('available=',x['available'],'score=',x['score'],'reliability=',x['reliability'],'map=',bool(x['map_png']),'confidence=',bool(x['confidence_png']))\""
            )
            print(_run(client, command))
        if args.verify_adaifl:
            response = f"{REMOTE_ROOT}/tmp_adaifl_http.json"
            command = (
                f"test -f {REMOTE_ROOT}/tmp_trufor_smoke.jpg || exit 2; "
                f"status=$(curl -sS --max-time 300 -o {response} -w '%{{http_code}}' -F file=@{REMOTE_ROOT}/tmp_trufor_smoke.jpg "
                f"http://127.0.0.1:6006/adaifl_analyze); "
                f"/root/miniconda3/bin/python -c \"import json; x=json.load(open('{response}'))['data']; "
                f"print('http=', '$status', 'available=',x['available'],'detail=',x['detail'])\""
            )
            print(_run(client, command))
        if args.gpu_log:
            print(_run(client, f"tail -n 50 {REMOTE_ROOT}/logs/gpu_server.log 2>/dev/null || true"))
        if args.port_status:
            command = (
                "inode=$(awk '$2 ~ /:1776$/ && $4 == \"0A\" {print $10}' /proc/net/tcp /proc/net/tcp6); "
                "echo listener_inode=$inode; "
                "for proc in /proc/[0-9]*; do for fd in $proc/fd/*; do "
                "[ \"$(readlink $fd 2>/dev/null)\" = \"socket:[$inode]\" ] && "
                "echo PID=${proc##*/} CMD=$(tr '\\0' ' ' < $proc/cmdline 2>/dev/null); "
                "done; done; ps -ef | grep '[g]pu_server.py' || true"
            )
            print(_run(client, command))
        if args.bind_test:
            print(_run(client, "/root/miniconda3/bin/python -c \"import socket; s=socket.socket(); s.bind(('0.0.0.0', 6006)); print('bind_ok'); s.close()\""))
        if args.foreground_gpu_smoke:
            print(_run(client, f"cd {REMOTE_ROOT} && timeout 12s env PYTHONUNBUFFERED=1 PRELOAD_MODELS=0 /root/miniconda3/bin/python gpu_server.py --host 0.0.0.0 --port 6006 2>&1 || true"))
        if args.service_procs:
            print(_run(client, "ps -eo pid,ppid,stat,etime,args | grep -E '[p]ython|[s]upervis|[u]vicorn|[g]unicorn|[s]creen'; screen -ls 2>&1 || true; echo '--- launcher tools ---'; command -v tmux || true; command -v screen || true; command -v supervisorctl || true; echo '--- supervisor config ---'; sed -n '1,240p' /init/supervisor/supervisor.ini 2>/dev/null || true"))
        if args.storage_audit:
            print(_run(client, (
                "df -h / /root/autodl-tmp 2>&1; echo '--- candidate storage ---'; "
                "du -sh /root/autodl-tmp/forge-detector-data /root/autodl-tmp/forge-detector-data/models "
                "/root/autodl-tmp/forge-detector-data/huggingface 2>/dev/null || true; "
                "echo '--- compatibility links ---'; readlink /root/.cache/huggingface "
                "/root/forge-detector/models /root/forge-detector/vendor/TruFor/TruFor_train_test/pretrained_models 2>/dev/null || true; "
                "echo '--- largest files ---'; "
                "find /root/autodl-tmp/forge-detector-data -type f -printf '%s %p\\n' 2>/dev/null "
                "| sort -nr | head -n 25 | numfmt --field=1 --to=iec 2>/dev/null || true"
            )))
        if args.migrate_data_disk:
            adaifl_url = os.environ.get("ADAIFL_DOWNLOAD_URL")
            if not adaifl_url:
                raise RuntimeError("ADAIFL_DOWNLOAD_URL is required for --migrate-data-disk")
            # The signed URL travels over SSH stdin only, so it is absent from
            # shell history, process arguments, project files and server logs.
            migration = r'''set -euo pipefail
DATA_ROOT=/root/autodl-tmp/forge-detector-data
MODEL_ROOT="$DATA_ROOT/models"
mkdir -p "$MODEL_ROOT/AdaIFL" "$DATA_ROOT"

# Download directly to the persistent data disk and publish atomically.
ADAIFL_TARGET="$MODEL_ROOT/AdaIFL/AdaIFL_v0.pth"
if [ ! -s "$ADAIFL_TARGET" ]; then
  curl -fsSL --retry 3 --connect-timeout 30 --max-time 1800 "$ADAIFL_URL" -o "$ADAIFL_TARGET.part"
  test -s "$ADAIFL_TARGET.part"
  mv "$ADAIFL_TARGET.part" "$ADAIFL_TARGET"
fi

# Stop only this application while paths are switched, avoiding a partial read.
screen -S forge_gpu -X quit >/dev/null 2>&1 || true
sleep 2

move_and_link() {
  source_path="$1"
  target_path="$2"
  if [ -L "$source_path" ]; then
    [ "$(readlink -f "$source_path")" = "$target_path" ] || { echo "unexpected symlink: $source_path" >&2; exit 1; }
    return
  fi
  if [ -e "$target_path" ]; then
    echo "target already exists: $target_path" >&2
    exit 1
  fi
  mv "$source_path" "$target_path"
  ln -s "$target_path" "$source_path"
}

move_and_link /root/.cache/huggingface "$DATA_ROOT/huggingface"
move_and_link /root/forge-detector/models "$MODEL_ROOT/runtime"
move_and_link /root/forge-detector/vendor/TruFor/TruFor_train_test/pretrained_models "$DATA_ROOT/trufor_pretrained_models"

echo "migration_complete"
du -sh "$DATA_ROOT" /root/.cache/huggingface /root/forge-detector/models /root/forge-detector/vendor/TruFor/TruFor_train_test/pretrained_models
'''
            print(_run(client, "bash -c 'read -r ADAIFL_URL; export ADAIFL_URL; bash -s'", adaifl_url + "\n" + migration))
        if args.start_install:
            trufor = _run(client, f"cd {REMOTE_ROOT} && nohup bash scripts/setup_trufor_remote.sh > logs/trufor_setup.log 2>&1 & echo $!")
            adaifl = _run(client, f"cd {REMOTE_ROOT} && nohup bash scripts/setup_adaifl_remote.sh > logs/adaifl_setup.log 2>&1 & echo $!")
            print(f"Started TruFor setup job: {trufor}")
            print(f"Started AdaIFL setup job: {adaifl}")
        elif args.start_adaifl:
            adaifl = _run(client, f"cd {REMOTE_ROOT} && nohup bash scripts/setup_adaifl_remote.sh > logs/adaifl_setup.log 2>&1 & echo $!")
            print(f"Started AdaIFL setup job: {adaifl}")
        elif args.start_trufor:
            trufor = _run(client, f"cd {REMOTE_ROOT} && nohup bash scripts/setup_trufor_remote.sh > logs/trufor_setup.log 2>&1 & echo $!")
            print(f"Started TruFor setup job: {trufor}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
