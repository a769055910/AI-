"""Forward localhost:6006 to the configured GPU server and reconnect on SSH loss.

Uses the ignored gpu.remote.local.json file; credentials never enter command lines
or logs. Launch this instead of another tunnel bound to the same local port.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import select
import socketserver
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import paramiko
except ImportError:
    sys.path.insert(0, str(ROOT / "tmp/ssh_dependencies"))
    import paramiko

from deploy_tamper_models import _deployment_settings

LOG = logging.getLogger("gpu_tunnel")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-port", type=int, default=6006)
    parser.add_argument("--remote-port", type=int, default=6006)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    runtime = ROOT / "tmp"
    runtime.mkdir(exist_ok=True)
    known_hosts = runtime / "gpu_tunnel_known_hosts"
    state_path = runtime / ("gpu_tunnel_state.json" if args.local_port == 6006 else f"gpu_tunnel_state_{args.local_port}.json")

    class ForwardServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = False
        daemon_threads = True

    class ForwardHandler(socketserver.BaseRequestHandler):
        def handle(self):
            transport = self.server.transport
            if transport is None or not transport.is_active():
                return
            channel = None
            try:
                channel = transport.open_channel(
                    "direct-tcpip", ("127.0.0.1", args.remote_port),
                    self.request.getpeername(), timeout=10,
                )
                while transport.is_active():
                    ready, _, _ = select.select([self.request, channel], [], [], 2)
                    if self.request in ready:
                        data = self.request.recv(65536)
                        if not data:
                            break
                        channel.sendall(data)
                    if channel in ready:
                        data = channel.recv(65536)
                        if not data:
                            break
                        self.request.sendall(data)
            except (OSError, EOFError, paramiko.SSHException):
                LOG.warning("Forwarded connection closed")
            finally:
                if channel is not None:
                    channel.close()

    # Keep one listener across reconnections; Windows may refuse an immediate
    # rebind when forwarded TCP connections are still closing.
    try:
        server = ForwardServer(("127.0.0.1", args.local_port), ForwardHandler)
    except OSError:
        LOG.error("Cannot bind local port %d; use only one tunnel", args.local_port)
        return
    server.transport = None
    threading.Thread(target=server.serve_forever, daemon=True).start()

    while True:
        client = paramiko.SSHClient()
        try:
            config = _deployment_settings()
            client.load_host_keys(str(known_hosts))
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            client.connect(
                hostname=str(config["HOST"]), port=int(config["PORT"]),
                username=str(config["USER"]), password=str(config["PASSWORD"]),
                timeout=15, auth_timeout=15, banner_timeout=15,
                look_for_keys=False, allow_agent=False,
            )
            transport = client.get_transport()
            transport.set_keepalive(15)
            server.transport = transport
            state_path.write_text(json.dumps({
                "pid": os.getpid(), "status": "connected",
                "local_port": args.local_port, "remote_port": args.remote_port,
            }), encoding="utf-8")
            LOG.info("Tunnel ready: 127.0.0.1:%d -> GPU 127.0.0.1:%d", args.local_port, args.remote_port)
            while transport.is_active():
                time.sleep(1)
            LOG.warning("SSH connection lost; reconnecting in 5 seconds")
        except KeyboardInterrupt:
            server.shutdown()
            server.server_close()
            return
        except OSError as exc:
            LOG.warning("Tunnel unavailable (%s); retrying in 5 seconds", type(exc).__name__)
        except Exception as exc:
            LOG.warning("Tunnel unavailable (%s); retrying in 5 seconds", type(exc).__name__)
        finally:
            server.transport = None
            client.close()
            if state_path.exists():
                state_path.write_text(json.dumps({"pid": os.getpid(), "status": "disconnected"}), encoding="utf-8")
        time.sleep(5)


if __name__ == "__main__":
    main()
