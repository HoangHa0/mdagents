import random
from utils import Agent, _noop_log, SampleAPICallTracker

def cot_examplers(examplers, args, log=None, tracker=None):
    if log is None:
        log = _noop_log
    
    log(f"\n[INFO] Generating CoT examplers with {args.fewshot} few-shot examplers.")
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=args.model, tracker=tracker)
    fewshot_examplers = []
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        for i, exampler in enumerate(examplers[:args.fewshot]):
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
            log(f"\nFewshot exampler #{i + 1}:\n{tmp_exampler['question']}\n{tmp_exampler['reason']}\n{tmp_exampler['answer']}")

    return fewshot_examplers


def cot_sc_query(question, examplers, args, sampling_num=10, log=None, tracker=None):
    if log is None:
        log = _noop_log
    
    sampling_responses = []

    log(f"\n[INFO] Generating CoT-SC responses with {args.fewshot} few-shot examplers and {sampling_num} sampling paths.")
    for i in range(sampling_num):
        single_agent = Agent(
            instruction="You are a helpful assistant that answers multiple choice questions about medical knowledge.", 
            role='medical expert', 
            examplers=cot_examplers(examplers, args, log=log, tracker=tracker), 
            model_info=args.model,
            tracker=tracker
        )

        response_dict = single_agent.temp_responses(
            f"The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n\nQuestion: {question}\nAnswer: ", 
            temperatures=[0.7], 
            img_path=None
        )
        sampling_responses.append(list(response_dict.values())[0])
        log(f"[INFO] Sampling path {i+1} completed.")

    # Majority voting
    decision_agent = Agent(
        instruction="You are a final medical decision maker who reviews all opinions and makes the final decision.",
        role="decision maker",
        model_info=args.model,
        tracker=tracker
    )
    
    formatted_opinions = "\n".join([f"Path {i+1}: {ans}" for i, ans in enumerate(sampling_responses)])
    final_prompt = (
        f"The following are {sampling_num} different reasoning paths and answers generated for the same question.\n"
        f"Opinions:\n{formatted_opinions}\n"
        f"Perform majority voting to select the most consistent answer."
    )

    log(f"[INFO] Opinion paths:\n{formatted_opinions}")
    
    log(f"[INFO] Final decision by majority voting")
    
    final_decision = decision_agent.temp_responses(final_prompt)
    log(final_decision[0.0])

    return final_decision