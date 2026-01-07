import random
from utils import Agent, _noop_log, SampleAPICallTracker

def cot_query(question, examplers, args, fewshot_num=3, log=None, tracker=None):
    if log is None:
        log = _noop_log
    
    log(f"\n[INFO] Generating CoT response with {fewshot_num} few-shot examplers.")
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=args.model, tracker=tracker)
    fewshot_examplers = []
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        for i, exampler in enumerate(examplers[:fewshot_num]):
            tmp_exampler = {}
            exampler_question = f"Question: {exampler['question']}"
            options = [f"({k}) {v}" for k, v in exampler['options'].items()]
            random.shuffle(options)
            exampler_question += " " + ' '.join(options)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"
            exampler_reason = "Reason: " + medical_agent.chat(f"You are a helpful medical agent. Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, provide 1-2 sentences of reason that support the answer suppose you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}")

            tmp_exampler['question'] = exampler_question
            tmp_exampler['reason'] = exampler_reason
            tmp_exampler['answer'] = exampler_answer
            fewshot_examplers.append(tmp_exampler)
            log(f"Fewshot exampler #{i + 1}:\n{tmp_exampler['question']}\n{tmp_exampler['reason']}\n{tmp_exampler['answer']}")
    
    single_agent = Agent(instruction="You are a helpful assistant that answers multiple choice questions about medical knowledge.", role='medical expert', examplers=fewshot_examplers, model_info=args.model, tracker=tracker)
    final_decision = single_agent.temp_responses(f"The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n\nQuestion: {question}\nAnswer: ", img_path=None)
    
    log(f"[INFO] CoT response generated.")
    log(final_decision[0.0])

    return final_decision