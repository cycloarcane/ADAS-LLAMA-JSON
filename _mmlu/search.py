import argparse
import copy
import json
import os
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import logging
import backoff
import numpy as np
import openai
import pandas
from tqdm import tqdm
import httpx
import tqdm
from mmlu_prompt import get_init_archive, get_prompt, get_reflexion_prompt
from utils import format_multichoice_question, random_id, bootstrap_confidence_interval, validate_and_correct_json
import traceback
import re

OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')


# Increase the timeout significantly
timeout = httpx.Timeout(60.0, connect=60.0, read=300.0)  # 30 seconds for connection, 300 seconds for reading
client = openai.OpenAI(
    base_url=OPENAI_API_BASE,
    http_client=httpx.Client(timeout=timeout)
)



Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = """
You must respond with valid JSON only. Your entire response should be a single, well-formed JSON object.
Do not include any text before or after the JSON object. If you need to include explanations or comments,
put them inside the JSON object as a separate field.
"""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Call this function at the beginning of your script
setup_logging()


@backoff.on_exception(backoff.expo, (openai.RateLimitError, httpx.HTTPStatusError), max_tries=5)

def parse_or_extract_fields(content):
    try:
        # First, try to parse as JSON
        return json.loads(content)
    except json.JSONDecodeError:
        # If that fails, try to extract key-value pairs
        result = {}
        lines = content.split('\n')
        for line in lines:
            match = re.match(r'"?(\w+)"?\s*:\s*(.+)', line)
            if match:
                key, value = match.groups()
                result[key.strip('"')] = value.strip().strip('"').strip("'")
        return result

def extract_json_like_content(content):
    # Try to find content enclosed in curly braces
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        return match.group(0)
    return content

@backoff.on_exception(backoff.expo, 
                      (openai.RateLimitError, httpx.HTTPStatusError, httpx.ReadTimeout), 
                      max_tries=5)
def get_json_response_from_gpt_reflect(
        msg_list,
        model,
        temperature=0.8
):
    print(f"Sending request to model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Message list: {msg_list}")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=msg_list,
            temperature=temperature,
            max_tokens=4096,
            stop=None,
            timeout=300  # 5 minutes timeout
        )
        print(f"Received response: {response}")
        
        content = response.choices[0].message.content
        print(f"\nRaw GPT reflect response:\n{content}\n{'='*50}")
        
        json_like_content = extract_json_like_content(content)
        result = parse_or_extract_fields(json_like_content)
        
        if not result:
            print(f"Unable to parse reflect response. Raw content: {content}")
            raise ValueError("Unable to parse or extract fields from the response")
        
        return result
    except httpx.ReadTimeout:
        print("Request timed out. The server took too long to respond.")
        raise
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
        raise
    except Exception as e:
        print(f"Error in get_json_response_from_gpt_reflect: {e}")
        raise
@backoff.on_exception(backoff.expo, 
                      (openai.RateLimitError, httpx.HTTPStatusError, httpx.ReadTimeout), 
                      max_tries=5)
def get_json_response_from_gpt(
        msg,
        model,
        system_message,
        temperature=0.5
):
    print(f"Sending request to model: {model}")
    print(f"Temperature: {temperature}")
    print(f"System message: {system_message}")
    print(f"User message: {msg}")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": msg},
            ],
            temperature=temperature,
            max_tokens=4096,
            stop=None,
            timeout=300  # 5 minutes timeout
        )
        print(f"Received response: {response}")
        
        content = response.choices[0].message.content
        print(f"\nRaw GPT response:\n{content}\n{'='*50}")
        
        json_like_content = extract_json_like_content(content)
        result = parse_or_extract_fields(json_like_content)
        
        if not result:
            print(f"Unable to parse response. Raw content: {content}")
            raise ValueError("Unable to parse or extract fields from the response")
        
        return result
    except httpx.ReadTimeout:
        print("Request timed out. The server took too long to respond.")
        raise
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
        raise
    except Exception as e:
        print(f"Error in get_json_response_from_gpt: {e}")
        raise

class LLMAgentBase():
    """
    Attributes:
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model='gpt-3.5-turbo-0125', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        # construct system prompt
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY the alphabet choice, i.e. A or B or C or D." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description) + "\n\n" + SYSTEM_MSG

        # construct input infos text
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            print(f"Error in query: {e}")
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            # try to fill in the missing field
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                    del response_json[key]
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)

class AgentSystem():
    def __init__(self) -> None:
        pass


def search(args):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
            start = archive[-1]['generation']
        else:
            start = 0
    else:
        archive = get_init_archive()
        start = 0

    for n in range(start, args.n_generation):
        print(f"\n============Generation {n + 1}=================")
        system_prompt, prompt = get_prompt(archive)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
            print(f"Initial solution for generation {n+1}: {next_solution}")

            Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
            # Reflexion 1
            msg_list.append({"role": "assistant", "content": json.dumps(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_1})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
            print(f"Solution after Reflexion 1 for generation {n+1}: {next_solution}")
            # Reflexion 2
            msg_list.append({"role": "assistant", "content": json.dumps(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_2})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
            print(f"Final solution for generation {n+1}: {next_solution}")
        except httpx.ReadTimeout:
            print(f"Timeout occurred during generation {n+1}. Moving to next generation.")
            continue
        except Exception as e:
            print(f"Error during generation {n+1}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Fallback mechanism
            next_solution = {
                "thought": "Fallback due to server error",
                "name": f"Fallback Agent {n+1}",
                "code": "def forward(self, taskInfo):\n    return taskInfo"
            }
            print(f"Using fallback solution: {next_solution}")

        acc_list = []
        for _ in range(args.debug_max):
            try:
                acc_list = evaluate_forward_fn(args, next_solution["code"])
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                print("During evaluation:")
                print(e)
                msg_list.append({"role": "assistant", "content": json.dumps(next_solution)})
                msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                try:
                    next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
                except Exception as e:
                    print(f"During LLM generate new solution: {e}")
                    continue
                continue
        if not acc_list:
            continue

        fitness_str = bootstrap_confidence_interval(acc_list)
        next_solution['fitness'] = fitness_str
        next_solution['generation'] = n + 1

        if 'debug_thought' in next_solution:
            del next_solution['debug_thought']
        if 'reflection' in next_solution:
            del next_solution['reflection']
        archive.append(next_solution)

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

def evaluate(args):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    eval_file_path = str(os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")).strip(".json") + "_evaluate.json"
    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)

    current_idx = 0
    while (current_idx < len(archive)):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if current_idx < len(eval_archive):
            current_idx += 1
            continue
        sol = archive[current_idx]
        print(f"current_gen: {sol['generation']}, current_idx: {current_idx}")
        current_idx += 1
        try:
            acc_list = evaluate_forward_fn(args, sol["code"])
        except Exception as e:
            print(e)
            continue
        fitness_str = bootstrap_confidence_interval(acc_list)
        sol['test_fitness'] = fitness_str
        eval_archive.append(sol)

        # save results
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)

def evaluate_forward_fn(args, forward_str):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)

    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    # set seed 0 for valid set
    df = pandas.read_csv(args.data_filename)
    random.seed(args.shuffle_seed)
    examples = [row.to_dict() for _, row in df.iterrows()]
    random.shuffle(examples)

    if SEARCHING_MODE:
        examples = examples[:args.valid_size] * args.n_repreat
    else:
        examples = examples[args.valid_size:args.valid_size + args.test_size] * args.n_repreat

    questions = [format_multichoice_question(example) for example in examples]
    answers = [LETTER_TO_INDEX[example['Answer']] for example in examples]

    print(f"problem length: {len(examples)}")
    max_workers = min(len(examples), args.max_workers) if args.multiprocessing else 1

    task_queue = []
    for q in questions:
        taskInfo = Info('task', 'User', q, -1)
        task_queue.append(taskInfo)

    agentSystem = AgentSystem()

    acc_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue)))

    for q_idx, res in enumerate(results):
        try:
            if isinstance(res, str) and res in LETTER_TO_INDEX:
                predicted_idx = LETTER_TO_INDEX[res]
            elif 'A)' in res:
                predicted_idx = 0
            elif 'B)' in res:
                predicted_idx = 1
            elif 'C)' in res:
                predicted_idx = 2
            elif 'D)' in res:
                predicted_idx = 3
            elif isinstance(res, list):
                try_res = res[1]
                predicted_idx = LETTER_TO_INDEX[try_res.content]
            elif res.content in LETTER_TO_INDEX:
                predicted_idx = LETTER_TO_INDEX[res.content]
            elif 'A)' in res.content:
                predicted_idx = 0
            elif 'B)' in res.content:
                predicted_idx = 1
            elif 'C)' in res.content:
                predicted_idx = 2
            elif 'D)' in res.content:
                predicted_idx = 3
            else:
                print(f"error in q {q_idx}")
                acc_list.append(0)
                continue
        except Exception as e:
            print(f"Error processing result for question {q_idx}: {e}")
            acc_list.append(0)
            continue

        if predicted_idx == answers[q_idx]:
            acc_list.append(1)
        else:
            acc_list.append(0)
    print(f"acc: {bootstrap_confidence_interval(acc_list)}")
    return acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default="dataset/mmlu.csv")
    parser.add_argument('--valid_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--expr_name', type=str, default="mmlu_gpt3.5_results")
    parser.add_argument('--n_generation', type=int, default=30)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model',
                        type=str,
                        default='gpt-4o-2024-05-13',
                        choices=['gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 'gpt-4o-2024-05-13'])

    args = parser.parse_args()
    # search
    SEARCHING_MODE = True
    search(args)

    # evaluate
    SEARCHING_MODE = False
    evaluate(args)