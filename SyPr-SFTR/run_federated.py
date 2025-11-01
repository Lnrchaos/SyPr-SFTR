import os
import subprocess
import time
import argparse
from typing import List
import torch
import numpy as np

def start_server(num_rounds: int = 10, num_clients: int = 5):
    """Start the federated learning server."""
    print(f"Starting federated learning server for {num_rounds} rounds with {num_clients} clients...")
    
    # Start the server in a separate process
    server_process = subprocess.Popen(
        ["python", "train_federated.py"],
        env={
            **os.environ,
            "NUM_CLIENTS": str(num_clients),
            "NUM_ROUNDS": str(num_rounds)
        }
    )
    
    # Give the server some time to start
    time.sleep(5)
    
    return server_process

def start_clients(num_clients: int):
    """Start multiple federated learning clients."""
    client_processes = []
    
    for client_id in range(num_clients):
        print(f"Starting client {client_id}...")
        process = subprocess.Popen(
            ["python", "client.py", f"--client-id={client_id}"],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(client_id % torch.cuda.device_count()) if torch.cuda.is_available() else ""
            }
        )
        client_processes.append(process)
        time.sleep(1)  # Stagger client starts
    
    return client_processes

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run federated learning experiment.")
    parser.add_argument("--num-clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of federated learning rounds")
    args = parser.parse_args()
    
    try:
        # Start the server
        server_process = start_server(
            num_rounds=args.num_rounds,
            num_clients=args.num_clients
        )
        
        # Start clients
        client_processes = start_clients(args.num_clients)
        
        # Wait for all clients to finish
        for process in client_processes:
            process.wait()
        
        # Wait for server to finish
        server_process.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure all processes are terminated
        try:
            server_process.terminate()
            for process in client_processes:
                process.terminate()
        except:
            pass

if __name__ == "__main__":
    main()
