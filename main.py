import os
import sys
import json
import random
import argparse
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils import (
    load_data, create_question, determine_difficulty,
    process_basic_query, process_intermediate_query, process_advanced_query,
    Agent, SampleAPICallTracker
)

# Thread-safe lock for results and file operations
results_lock = threading.Lock()
file_lock = threading.Lock()

# Logger class for logging to both console and file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def _atomic_json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, indent=4)
    os.replace(tmp_path, path)

def make_log_func(multithread, log_lines=None):
    """
    Returns a log function:
      - multithread=True: appends to log_lines list (no terminal output)
      - multithread=False: acts like print() (terminal output captured by Logger)
    """
    if multithread:
        def log(msg):
            log_lines.append(str(msg))
        return log
    else:
        return print

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='medqa')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--difficulty', type=str, default='adaptive')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for LLM sampling (default: 0.0)')
    parser.add_argument('--multithread', action='store_true', help='Enable multithreaded execution.')
    args = parser.parse_args()

    file_name = f"{args.dataset}_{args.model}_{args.difficulty}_{args.num_samples}{'_' + str(args.seed) if args.seed is not None else ''}_{args.temperature}"

    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    if args.multithread:
        # Create per-sample logs directory for multithreaded mode
        sample_logs_dir = os.path.join('logs', f"{file_name}_samples")
        os.makedirs(sample_logs_dir, exist_ok=True)
    else:
        # Single-threaded: redirect stdout to terminal + single file
        main_log_path = f"logs/{file_name}.log"
        sys.stdout = Logger(main_log_path)

    # Output + resume paths
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{file_name}.json")
    progress_path = os.path.join('logs', f"{file_name}.progress.json")

    # Load previous results if present (resume)
    results = []
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
            if not isinstance(results, list):
                print(f"[WARN] Existing output is not a list. Starting fresh: {output_path}")
                results = []
        except Exception as e:
            print(f"[WARN] Failed to load existing output ({output_path}): {e}. Starting fresh.")
            results = []

    start_no = len(results)
    if start_no > 0:
        print(f"[INFO] Resuming from sample index {start_no} (already saved {start_no} results).")

    # Keep a tiny progress file too (helpful if output gets edited)
    _atomic_json_dump({"next_index": start_no}, progress_path)

    test_qa, examplers = load_data(args.dataset)

    # Randomly select test samples for quicker testing (remove this part for full eval)
    if args.seed is not None:
        random.seed(args.seed)
        
    if args.num_samples is not None and args.num_samples < len(test_qa):
        test_qa = random.sample(test_qa, args.num_samples)

    # Prepare samples to process (skip already completed)
    samples_to_process = list(enumerate(test_qa[start_no:], start=start_no))
    if args.num_samples is not None:
        samples_to_process = [(no, s) for no, s in samples_to_process if no < args.num_samples]

    if args.multithread:
        # ==================== MULTITHREADED MODE ====================
        def process_sample(no, sample):
            """Process a single sample - thread worker function"""
            sample_log_path = os.path.join(sample_logs_dir, f"sample_{no:04d}.log")
            log_lines = []
            log = make_log_func(multithread=True, log_lines=log_lines)
            
            try:
                log(f"[INFO] Processing sample {no}")
                tracker = SampleAPICallTracker()
                question, img_path = create_question(sample, args.dataset)
                
                difficulty, moderator = determine_difficulty(question, args.difficulty, log=log, tracker=tracker)

                if difficulty == 'basic':
                    final_decision = process_basic_query(question, examplers, args, log=log, tracker=tracker)
                elif difficulty == 'intermediate':
                    final_decision = process_intermediate_query(question, examplers, moderator, args, log=log, tracker=tracker)
                elif difficulty == 'advanced':
                    final_decision = process_advanced_query(question, args, log=log, tracker=tracker)
                else:
                    raise ValueError(f"Unknown difficulty: {difficulty}")

                sample_api_calls = tracker.total_calls()
                
                log(f"\n[INFO] API calls for this sample: {sample_api_calls}")

                if args.dataset == 'medqa':
                    result = {
                        'index': no,
                        'question': question,
                        'label': sample['answer_idx'],
                        'answer': sample['answer'],
                        'options': sample['options'],
                        'response': final_decision,
                        'difficulty': difficulty,
                        'api_calls': sample_api_calls
                    }
                else:
                    result = {
                        'index': no,
                        'question': question,
                        'response': final_decision,
                        'difficulty': difficulty,
                        'api_calls': sample_api_calls
                    }

                log(f"\n[INFO] Sample {no} completed successfully")
                
                # Write log to file
                with open(sample_log_path, 'w') as f:
                    f.write('\n'.join(log_lines))
                
                return no, result, None

            except Exception as e:
                error_info = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                log(f"[ERROR] Exception at sample {no}: {error_info}")
                
                # Write log to file even on error
                with open(sample_log_path, 'w') as f:
                    f.write('\n'.join(log_lines))
                
                return no, None, error_info

        # Thread-safe function to save results
        def save_results_thread_safe():
            with file_lock:
                _atomic_json_dump(results, output_path)
                _atomic_json_dump({"next_index": len(results)}, progress_path)

        # Main multithreaded loop
        errors = []
        shutdown_flag = threading.Event()

        NUM_WORKERS = 20  # Fixed number of worker threads
        print(f"[INFO] Starting processing with {NUM_WORKERS} worker threads...")
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(process_sample, no, sample): (no, sample) 
                for no, sample in samples_to_process
            }
            
            # Progress bar for completed tasks
            pbar = tqdm(total=len(test_qa), initial=start_no, desc="Processing samples")
            
            try:
                for future in as_completed(future_to_sample):
                    if shutdown_flag.is_set():
                        break
                        
                    no, result, error = future.result()
                    
                    if error:
                        errors.append((no, error))
                    else:
                        # Thread-safe result storage
                        with results_lock:
                            results.append(result)
                            # Sort by index to maintain order
                            results.sort(key=lambda x: x.get('index', 0))
                        
                        # Save progress periodically (thread-safe)
                        save_results_thread_safe()
                    
                    pbar.update(1)
                    
            except KeyboardInterrupt:
                print("\n[WARN] Interrupted by user (KeyboardInterrupt). Shutting down workers...")
                shutdown_flag.set()
                executor.shutdown(wait=False, cancel_futures=True)
                
            finally:
                pbar.close()

        # Remove the 'index' field from results before final save
        with results_lock:
            for r in results:
                r.pop('index', None)
        
        # Final save
        save_results_thread_safe()
        
        if errors:
            print(f"\n[WARN] {len(errors)} samples failed with errors")
            for no, err in errors:
                print(f"  - Sample {no}: {err.split(chr(10))[0]}")
        
        print(f"[INFO] Done. Saved {len(results)} samples to: {output_path}")
        print(f"[INFO] Total API calls: {Agent.get_total_api_calls()}")

    else:
        # ==================== SINGLE-THREADED MODE ====================
        # log = print (stdout is already redirected to Logger)
        log = make_log_func(multithread=False)
        
        for no, sample in enumerate(
            tqdm(test_qa[start_no:], total=len(test_qa), initial=start_no),
            start=start_no 
        ):
            if args.num_samples is not None and no >= args.num_samples:
                break

            if no == 0:
                log(f"[INFO] no: {no}")
            else:    
                log(f"\n\n[INFO] no: {no}")

            try:
                tracker = SampleAPICallTracker()
                question, img_path = create_question(sample, args.dataset)
                difficulty, moderator = determine_difficulty(question, args.difficulty, log=log, tracker=tracker)

                log(f"difficulty: {difficulty}")

                if difficulty == 'basic':
                    final_decision = process_basic_query(question, examplers, args, log=log, tracker=tracker)
                elif difficulty == 'intermediate':
                    final_decision = process_intermediate_query(question, examplers, moderator, args, log=log, tracker=tracker)
                elif difficulty == 'advanced':
                    final_decision = process_advanced_query(question, args, log=log, tracker=tracker)
                else:
                    raise ValueError(f"Unknown difficulty: {difficulty}")

                sample_api_calls = tracker.total_calls()

                if args.dataset == 'medqa':
                    results.append({
                        'question': question,
                        'label': sample['answer_idx'],
                        'answer': sample['answer'],
                        'options': sample['options'],
                        'response': final_decision,
                        'difficulty': difficulty,
                        'api_calls': sample_api_calls
                    })
                else:
                    results.append({
                        'question': question,
                        'response': final_decision,
                        'difficulty': difficulty,
                        'api_calls': sample_api_calls
                    })

                # Save after each successful sample
                _atomic_json_dump(results, output_path)
                _atomic_json_dump({"next_index": len(results)}, progress_path)
                log(f"\n[INFO] API calls for this sample: {sample_api_calls}")

            except KeyboardInterrupt:
                log("\n[WARN] Interrupted by user (KeyboardInterrupt). Saving progress and exiting...")
                _atomic_json_dump(results, output_path)
                _atomic_json_dump({"next_index": len(results)}, progress_path)
                break

            except Exception as e:
                log(f"\n[ERROR] Exception at sample index {no}: {type(e).__name__}: {e}")
                traceback.print_exc()
                log("[INFO] Saving progress up to last completed sample and exiting...")
                _atomic_json_dump(results, output_path)
                _atomic_json_dump({"next_index": len(results)}, progress_path)
                break

        # Final save (in case loop ends normally)
        _atomic_json_dump(results, output_path)
        _atomic_json_dump({"next_index": len(results)}, progress_path)
        log(f"\n[INFO] Done. Saved {len(results)} samples to: {output_path}")
        log(f"[INFO] Total API calls: {Agent.get_total_api_calls()}")

if __name__ == "__main__":
    main()