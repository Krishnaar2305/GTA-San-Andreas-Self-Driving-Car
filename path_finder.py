import subprocess
import sys
import os
import time
import colorama  # <--- NEW: Import colorama
from colorama import Fore, Style # <--- NEW: Import Fore and Style

# --- Configuration ---
# The *absolute path* to the directory containing demo.py
# This is crucial so it can find 'utils' and 'data'
YOLOP_DIR = r"D:\Coding\pygta5\original_project\YOLOPv2\path_123\YOLOPv2"
# ---------------------

def run_demo():
    """
    Runs the demo.py script as a subprocess and captures its output in real-time.
    Only prints inference lines (in green) or errors (in red).
    """
    
    colorama.init() # <--- NEW: Initialize colorama

    # 1. Define the command to run
    command = [
        sys.executable, 
        'main.py',
        '--weights', 'data/weights/yolopv2.pt',
        '--device', '0',
        # '--nosave'
    ]

    inferences_log = []
    print(f"--- Starting YOLOPv2 in: {YOLOP_DIR} ---")
    print(f"--- Command: {' '.join(command)} ---")
    
    process = None
    try:
        # 2. Start the subprocess
        process = subprocess.Popen(
            command,
            cwd=YOLOP_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1 
        )

        print(f"--- YOLOPv2 Process Started (PID: {process.pid}) ---")
        print("--- Press Ctrl+C in this terminal to stop --- \n")

        # 3. Read output in real-time
        while True:
            output_line = process.stdout.readline()
            
            if output_line == '' and process.poll() is not None:
                break
                
            if output_line:
                line = output_line.strip()
                
                # <--- NEW: Filter output and print with color ---
                # Check for keywords that indicate a successful inference frame
                if 'Done.' in line or 'inf :' in line or 'screen capture' in line:
                    # A. Print the *inference line* to this console in GREEN
                    print(f"{Fore.GREEN}[YOLOPv2]: {line}{Style.RESET_ALL}")
                
                # Also print errors from the script in RED
                elif 'ERROR' in line or 'Error:' in line or 'Traceback' in line:
                    print(f"{Fore.RED}[YOLOPv2 ERROR]: {line}{Style.RESET_ALL}")
                # --- END OF MODIFICATION ---

                # B. Save *all* lines to our log, regardless of color
                inferences_log.append(line)

        # 4. Process finished, check for errors
        print("\n--- YOLOPv2 Process Finished ---")
        
        stderr_output = process.stderr.read()
        if process.returncode != 0:
            print(f"*** {Fore.RED}ERROR (Return Code: {process.returncode}){Style.RESET_ALL} ***")
            print(stderr_output)
        else:
            print(f"{Fore.GREEN}Process completed successfully.{Style.RESET_ALL}")

    except KeyboardInterrupt:
        # 5. Handle user pressing Ctrl+C
        print("\n--- User interrupted! Sending stop signal to YOLOPv2... ---")
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("--- Process did not stop, forcing kill. ---")
                process.kill()
        print("--- Process Terminated ---")

    except Exception as e:
        print(f"\n*** {Fore.RED}An unexpected error occurred: {e}{Style.RESET_ALL} ***")
        if process:
            process.kill()

    finally:
        # 6. Show summary of captured inferences
        print("\n--- Summary of Captured Inference Timings ---")
        if not inferences_log:
            print("No output was captured.")
        else:
            timing_lines = [l for l in inferences_log if 'Done' in l or 'inf' in l or 'screen capture' in l]
            if not timing_lines:
                print("No performance lines were captured.")
            else:
                for line in timing_lines:
                    # Print summary in green as well
                    print(f"{Fore.GREEN}{line}{Style.RESET_ALL}")
        
        print("\n--- Script finished ---")
        return inferences_log

if __name__ == "__main__":
    all_output = run_demo()