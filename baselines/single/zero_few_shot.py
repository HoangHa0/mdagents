import random
from utils import Agent, _noop_log, SampleAPICallTracker

def zero_few_shot_query(question, examplers, args, log=None, tracker=None):
    if log is None:
        log = _noop_log
    
    if args.method == 'zero_shot':
        log(f"\n[INFO] Generating Zero-Shot response.")
        single_agent = Agent(
            instruction="You are a helpful assistant that answers multiple choice questions about medical knowledge.", 
            role='medical expert',
            model_info=args.model,
            tracker=tracker
        )
        response_dict = single_agent.temp_responses(
            f"The following is a multiple choice question (with answer choices) about medical knowledge. Please provide the final answer directly.\n\nQuestion: {question}\nAnswer: ", 
            temperatures=[0.0], 
            img_path=None
        )
        final_decision = list(response_dict.values())[0]
        log(final_decision)

        return final_decision
    
    elif args.method == 'few_shot':
        log(f"\n[INFO] Generating Few-Shot response with {args.fewshot} few-shot examplers.")
        if args.dataset == 'medqa':
            random.shuffle(examplers)
            fewshot_examplers = []
            for i, exampler in enumerate(examplers[:args.fewshot]):
                tmp_exampler = {}
                exampler_question = f"Question: {exampler['question']}"
                options = [f"({k}) {v}" for k, v in exampler['options'].items()]
                random.shuffle(options)
                exampler_question += " " + ' '.join(options)
                exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"

                tmp_exampler['question'] = exampler_question
                tmp_exampler['answer'] = exampler_answer
                fewshot_examplers.append(tmp_exampler)
                log(f"\nFewshot exampler #{i + 1}:\n{tmp_exampler['question']}\n{tmp_exampler['answer']}")

        single_agent = Agent(
            instruction="You are a helpful assistant that answers multiple choice questions about medical knowledge.", 
            role='medical expert', 
            examplers=fewshot_examplers,
            model_info=args.model,
            tracker=tracker
        )

        response_dict = single_agent.temp_responses(
            f"The following is a multiple choice question (with answer choices) about medical knowledge. Please provide the final answer directly.\n\nQuestion: {question}\nAnswer: ", 
            temperatures=[0.0], 
            img_path=None
        )
        final_decision = list(response_dict.values())[0]
        log(final_decision)
        
        return final_decision