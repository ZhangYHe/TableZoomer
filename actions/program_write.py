"""
Created Date: 2025-05-23.
Author: xiongsishi@chinatelecom.cn
"""
import ast
import json
import re
import sys

import fire
import subprocess
from metagpt.actions import Action
from metagpt.logs import logger

from actions.query_analyse import extract_from_content


def parse_code(rsp):
    pattern = r"```python(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text

def extract_first_curly_braces(text):
    match = re.search(r'\{[^{}]*\}', text)
    return match.group(0) if match else None


def strip_json_fence(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[len("```json"):].strip()
        text = text[:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    return text


def extract_balanced_json_object(text):
    for start in (idx for idx, char in enumerate(text) if char == "{"):
        candidate = _extract_balanced_json_object_from(text, start)
        if candidate is not None:
            return candidate
    return None


def extract_balanced_json_objects(text):
    candidates = []
    for start in (idx for idx, char in enumerate(text) if char == "{"):
        candidate = _extract_balanced_json_object_from(text, start)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _extract_balanced_json_object_from(text, start):
    depth = 0
    in_string = False
    escape_next = False

    for idx in range(start, len(text)):
        char = text[idx]
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def normalize_json_response(text):
    text = strip_json_fence(text)
    text = extract_from_content(text).strip()
    candidates = [text]
    for balanced in extract_balanced_json_objects(text):
        if balanced in candidates:
            continue
        candidates.append(balanced)

    last_error = None
    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate, None
        except Exception as error:
            last_error = error
    return text, last_error



class SimpleWriteCode(Action):
    """ Generate efficient and robust Python code based on the user's query, analysis, and data file path """
    name: str = "SimpleWriteCode"

    async def run(self, instruction: str):

        instruction = json.loads(instruction)
        # prompt = self.PROMPT_TEMPLATE.replace("{final_table_schema}", str(json.dumps(instruction['table_desc'], ensure_ascii=False, indent=4))).replace("{query}", instruction['query']).replace("{query_analysis}", instruction['query_analysis'])
        prompt = self.PROMPT_TEMPLATE.replace("{final_table_schema}",
                                              str(json.dumps(instruction['table_desc'], ensure_ascii=False,
                                                             indent=4))).replace("{query}", instruction['query'])

        if 'last_turn_error' in instruction:
            prompt = prompt + '\n' + instruction['last_turn_error']

        if 'table_zoom' in instruction and instruction['table_zoom'] is not None:
            table_zoom = instruction['table_zoom']
            prompt = prompt + '\n\n' + f'(Pay attention to Additional Information!!!:\n Table_zoom is streamlined information extracted by compressing rows and columns of the original table. Table_zoom:\n{table_zoom}\n\n In most cases, answers can be obtained by focusing only on the table_zoom, and more accurate code can be generated with reference to it.But it do not always contains all necessary information, please carefully check if its data is enough to solve current query, if not, please refer to origin table_schema for more details. )'

        rsp = await self._aask(prompt)
        rsp, parse_error = normalize_json_response(rsp)
        if parse_error is not None:
            logger.info(f'{self.name}: error! The generated response does not comply with JSON syntax: \n{parse_error}\nReflection and try again!')
            json_error_fix_request = f"""---- \
In the previous round, the response you output was:

{rsp}

Unfortunately, this response failed to load by `json.loads()`, indicating that your response did not follow the JSON format requirements. The error type is:
{parse_error}
----

Please fix the format and answer again with exactly one valid JSON object and nothing else.
Requirements:
1. The response must be loadable by Python `json.loads(response)`.
2. Use double quotes for JSON keys and string values.
3. Include exactly two keys: "code_thought" and "code".
4. Put the complete Python program inside the "code" JSON string.
5. Escape newlines as `\\n`, internal double quotes as `\\"`, and literal backslashes as `\\\\`.
6. Do not use Markdown fences, Python dict syntax, comments outside JSON, or any explanatory text outside JSON.
7. Do not output standalone brace fragments from Python code such as `{missing_cols}` outside the "code" string.

**User Query**: {instruction['query']}
**Response**: """
            prompt = prompt + '\n\n' + json_error_fix_request
            rsp = await self._aask(prompt)
            rsp, parse_error = normalize_json_response(rsp)
            if parse_error is not None:
                logger.warning(f'{self.name}: retry response still does not comply with JSON syntax: {parse_error}')

        # return json.dumps(rsp, ensure_ascii=False)
        return json.dumps({"prompt": prompt, "rsp": rsp}, ensure_ascii=False)


class SimpleRunCode(Action):
    name: str = "SimpleRunCode"

    async def run(self, inputs: str):
        try:
            code_rsp = json.loads(inputs)['rsp']
            code_gen_prompt = json.loads(inputs)['prompt']
            try:
                code_instructions = json.loads(code_rsp)
            except Exception as e:
                logger.warning(f'code_rsp load warning: {e}. Try to use ast.literal_eval() func.')
                
                try:
                    code_instructions = ast.literal_eval(code_rsp)
                except Exception as e:
                    logger.warning(f'code_rsp load warning: {e}.')
                    code_results = {
                        "prompt": code_gen_prompt,
                        "code_rsp": '',
                        "code": '',
                        "response": '', 
                        # "file": ci.get('file', '')
                        "execute_state": 'fail',
                        "error": f"Code loading failed!\n{e}"
                    }
                    return json.dumps(code_results, ensure_ascii=False)

            code_text = code_instructions['code']

        except Exception as e:
            logger.warning(f'code_rsp load warning: {e}.')
            code_results = {
                "prompt": code_gen_prompt,
                "code_rsp": '',
                "code": '',
                "response": '', 
                "execute_state": 'fail',
                "error": f"Code loading failed!\n{e}"
            }
            return json.dumps(code_results, ensure_ascii=False)

        try:
            result = subprocess.run([sys.executable, "-c", code_text], capture_output=True, text=True, check=True, timeout=60)
            code_result = result.stdout.strip()
            execute_state = result.returncode
            code_results = {
                "prompt": code_gen_prompt,
                "code_rsp": code_rsp,
                "code": code_text,
                "response": code_result,
                # "file": ci.get('file', '')
                "execute_state": execute_state
            }
        except subprocess.CalledProcessError as error:
            code_results = {
                "prompt": code_gen_prompt,
                "code_rsp": code_rsp,
                "code": code_text,
                "response": '',   
                "execute_state": 'fail',
                "error": error.stderr
            }
        except subprocess.TimeoutExpired as error:
            code_results = {
                "prompt": code_gen_prompt,
                "code_rsp": code_rsp,
                "code": code_text,
                "response": '',  
                "execute_state": 'fail',
                "error": "TimeoutExpired Exception"
            }
        except Exception as error:
            code_results = {
                "prompt": code_gen_prompt,
                "code_rsp": code_rsp,
                "code": code_text,
                "response": '', 
                "execute_state": 'fail',
                "error": error
            }

        return json.dumps(code_results, ensure_ascii=False)
