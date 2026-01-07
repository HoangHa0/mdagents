import os
import sys
import json
import random
import traceback
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from utils import setup_model, load_data, create_question, Agent, SampleAPICallTracker
from main import Logger, _atomic_json_dump, make_log_func

from baselines.single.zero_few_shot import zero_few_shot_query
from baselines.single.cot import cot_query
from baselines.single.cot_sc import cot_sc_query

# Thread-safe locks
results_lock = threading.Lock()
file_lock = threading.Lock()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('--dataset', type=str, default='medqa')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--method', type=str, default='cot', choices=['zero_shot', 'few_shot', 'cot', 'cot_sc'], help='Method to use')
    parser.add_argument('--fewshot', type=int, default=3, help='Number of few-shot examples to use. If method is zero-shot, this is ignored.')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for response generation.')
    parser.add_argument('--multithread', action='store_true', help='Enable multithreaded execution.')
    args = parser.parse_args()
    
    file_name = f"{args.dataset}_{args.model}_{args.method}_{args.num_samples}{'_' + str(args.seed) if args.seed is not None else ''}_{args.temperature}"

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
        pass

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
                
                if args.method == 'zero_shot' or args.method == 'few_shot':
                    final_decision = zero_few_shot_query(question, examplers, args, fewshot_num=args.fewshot, log=log, tracker=tracker)
                elif args.method == 'cot':
                    final_decision = cot_query(question, examplers, args, fewshot_num=args.fewshot, log=log, tracker=tracker)
                elif args.method == 'cot_sc':
                    final_decision = cot_sc_query(question, examplers, args, fewshot_num=args.fewshot, log=log, tracker=tracker)
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
                        'api_calls': sample_api_calls
                    }
                else:
                    result = {
                        'index': no,
                        'question': question,
                        'response': final_decision,
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
        
        NUM_WORKERS = 20
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

                if args.method == 'zero_shot' or args.method == 'few_shot':
                    final_decision = zero_few_shot_query(question, examplers, args, fewshot_num=args.fewshot, log=log, tracker=tracker)
                elif args.method == 'cot':
                    final_decision = cot_query(question, examplers, args, fewshot_num=args.fewshot, log=log, tracker=tracker)
                elif args.method == 'cot_sc':
                    final_decision = cot_sc_query(question, examplers, args, fewshot_num=args.fewshot, log=log, tracker=tracker)
                sample_api_calls = tracker.total_calls()

                if args.dataset == 'medqa':
                    results.append({
                        'question': question,
                        'label': sample['answer_idx'],
                        'answer': sample['answer'],
                        'options': sample['options'],
                        'response': final_decision,
                        'api_calls': sample_api_calls
                    })
                else:
                    results.append({
                        'question': question,
                        'response': final_decision,
                        'api_calls': sample_api_calls
                    })

                # Save after each successful sample
                _atomic_json_dump(results, output_path)
                _atomic_json_dump({"next_index": len(results)}, progress_path)
                log(f"\n[INFO] API calls for this sample: {sample_api_calls}")

            except KeyboardInterrupt:
                log("[WARN] Interrupted by user (KeyboardInterrupt). Saving progress and exiting...")
                _atomic_json_dump(results, output_path)
                _atomic_json_dump({"next_index": len(results)}, progress_path)
                break

            except Exception as e:
                log(f"[ERROR] Exception at sample index {no}: {type(e).__name__}: {e}")
                traceback.print_exc()
                log("[INFO] Saving progress up to last completed sample and exiting...")
                _atomic_json_dump(results, output_path)
                _atomic_json_dump({"next_index": len(results)}, progress_path)
                break

        # Final save
        _atomic_json_dump(results, output_path)
        _atomic_json_dump({"next_index": len(results)}, progress_path)
        log(f"\n[INFO] Done. Saved {len(results)} samples to: {output_path}")
        log(f"[INFO] Total API calls: {Agent.get_total_api_calls()}")
