#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os

def check_remote_gpus(ip, user):
    ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no"]
    if user:
        ssh_cmd.append(f"{user}@{ip}")
    else:
        ssh_cmd.append(ip)

    # Command to count NVIDIA GPUs
    cmd_nvidia = "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader | wc -l || echo 0"
    
    # Command to count OpenCL GPUs (excluding CPU)
    cmd_dri = "command -v clinfo >/dev/null 2>&1 && clinfo -l | grep -i 'Device #' | grep -vi 'cpu' | grep -iE 'intel|amd|radeon' | wc -l || echo 0"

    try:
        res_nv = subprocess.run(ssh_cmd + [cmd_nvidia], capture_output=True, text=True, check=True)
        nvidia_count = int(res_nv.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"  [!] SSH/Command error on {ip} (NVIDIA check). Exit code {e.returncode}. {e.stderr.strip() if e.stderr else ''}")
        nvidia_count = 0
    except Exception as e:
        print(f"  [!] Error connecting to {ip}: {e}")
        nvidia_count = 0

    try:
        res_dri = subprocess.run(ssh_cmd + [cmd_dri], capture_output=True, text=True, check=True)
        dri_count = int(res_dri.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"  [!] SSH/Command error on {ip} (OpenCL check). Exit code {e.returncode}. {e.stderr.strip() if e.stderr else ''}")
        dri_count = 0
    except Exception as e:
        print(f"  [!] Error connecting to {ip}: {e}")
        dri_count = 0

    opencl_count = dri_count
    total_gpus = nvidia_count + opencl_count
    
    if total_gpus == 0:
        total_gpus = 1
        print(f"  [!] No GPUs detected on {ip}. Assigning 1 slot for CPU fallback.")
    else:
        print(f"  [+] Node {ip}: {nvidia_count} NVIDIA GPUs, {opencl_count} Integrated GPUs -> {total_gpus} hardware slots.")

    return total_gpus

def main():
    parser = argparse.ArgumentParser(description="Auto-launcher for Rescue Video Analyzer Cluster")
    parser.add_argument("--ips", required=True, help="Comma-separated list of IP addresses (e.g. localhost,192.168.1.10)")
    parser.add_argument("--user", help="SSH username (optional)")
    parser.add_argument("--exec", default="./build/rescue_video_analyzer", dest="executable", help="Path to executable")
    parser.add_argument("--args", default="--input dataset/videoset2.mp4 --output-dir output --batch-size 8 --scheduler dynamic --gpu-backend auto", help="Arguments for the executable")
    
    args = parser.parse_args()
    ips = [ip.strip() for ip in args.ips.split(",") if ip.strip()]

    if not ips:
        print("Error: No IPs provided.")
        sys.exit(1)

    print("--- Rescue Video Analyzer: Auto Cluster Discovery ---")
    slots_per_ip = {}
    
    for i, ip in enumerate(ips):
        print(f"[*] Probing node {ip}...")
        if ip in ("localhost", "127.0.0.1"):
            cmd_nvidia = "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader | wc -l || echo 0"
            cmd_dri = "command -v clinfo >/dev/null 2>&1 && clinfo -l | grep -i 'Device #' | grep -vi 'cpu' | grep -iE 'intel|amd|radeon' | wc -l || echo 0"
            try:
                nv = int(subprocess.run(cmd_nvidia, shell=True, capture_output=True, text=True).stdout.strip())
            except Exception: nv = 0
            try:
                dri = int(subprocess.run(cmd_dri, shell=True, capture_output=True, text=True).stdout.strip())
            except Exception: dri = 0
            
            opencl = dri
            total_gpus = nv + opencl
            if total_gpus == 0:
                total_gpus = 1
            
            # Master needs 1 extra slot for the orchestrator rank
            if i == 0:
                total_gpus += 1
                print(f"  [+] Master Node (Local): {nv} NVIDIA GPUs, {opencl} Integrated GPUs. (+1 Master Orchestrator) -> {total_gpus} slots.")
            else:
                print(f"  [+] Worker Node (Local): {nv} NVIDIA GPUs, {opencl} Integrated GPUs -> {total_gpus} slots.")
                
            slots_per_ip[ip] = total_gpus
        else:
            slots = check_remote_gpus(ip, args.user)
            if i == 0:
                slots += 1
                print(f"  [+] Master Node (Remote). (+1 Master Orchestrator) -> {slots} slots.")
            slots_per_ip[ip] = slots

    hostfile_path = "hosts_cluster.txt"
    print(f"\n[*] Generating hostfile: {hostfile_path}")
    import socket
    local_hostname = socket.gethostname()

    with open(hostfile_path, "w") as f:
        for ip, slots in slots_per_ip.items():
            if ip in ("localhost", "127.0.0.1"):
                f.write(f"{local_hostname} slots={slots}\n")
            elif args.user:
                f.write(f"{args.user}@{ip} slots={slots}\n")
            else:
                f.write(f"{ip} slots={slots}\n")

    total_slots = sum(slots_per_ip.values())

    mpi_cmd = f"mpiexec --mca plm_rsh_args \"-o StrictHostKeyChecking=no\" --mca btl_tcp_disable_family IPv6 --hostfile {hostfile_path} -n {total_slots} {args.executable} {args.args}"
    print(f"[*] Launching MPI Cluster with {total_slots} total processes...")
    print(f"    $ {mpi_cmd}\n")
    
    # Give control to mpiexec
    os.system(mpi_cmd)

if __name__ == "__main__":
    main()
