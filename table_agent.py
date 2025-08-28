"""
Created Date: 2025-05-22
Author: xiongsishi@chinatelecom.cn
"""
import ast
import asyncio
import copy
import json
import os
import time
from pathlib import Path
import yaml

from metagpt.logs import logger
from metagpt.config2 import Config
from roles import *
from actions.table_desc import get_refined_table_schema
import pandas as pd

CUR_ROOT = os.path.dirname(os.path.abspath(__file__))


class TableZoomer():
    def __init__(self,
            config_file,
            max_react_round=5,
            ):

        self._init_llm_prompt_config(config_file)
        self._init_roles()
        self.max_react_round = max_react_round
        
    def _init_llm_prompt_config(self, config_file):
        # Read YAML config file.
        with open(config_file, 'r') as file:
            yaml_config = yaml.safe_load(file)
        # prompt settings of different module.
        self.react_prompt = open(yaml_config['prompt_template']['react'], 'r', encoding='utf8').read()
        self.table_desc_prompt = open(yaml_config['prompt_template']['table_desc'], 'r', encoding='utf8').read()
        self.query_expansion_prompt = open(yaml_config['prompt_template']['query_expansion'], 'r', encoding='utf8').read()
        self.code_generate_prompt = open(yaml_config['prompt_template']['code_generation'], 'r', encoding='utf8').read()
        self.answer_summary_prompt = open(yaml_config['prompt_template']['answer_summary'], 'r', encoding='utf8').read()
        
        # llm settings  of different module.
        self.react_llm_config = Config.from_yaml_file(Path(os.path.join(CUR_ROOT, 'agent_config', yaml_config['llm_config']['react'])))
        self.table_llm_config = Config.from_yaml_file(Path(os.path.join(CUR_ROOT, 'agent_config', yaml_config['llm_config']['table_desc'])))
        self.query_llm_config = Config.from_yaml_file(Path(os.path.join(CUR_ROOT, 'agent_config', yaml_config['llm_config']['query_expansion'])))
        self.code_llm_config = Config.from_yaml_file(Path(os.path.join(CUR_ROOT, 'agent_config', yaml_config['llm_config']['code_generation'])))
        self.summary_llm_config = Config.from_yaml_file(Path(os.path.join(CUR_ROOT, 'agent_config', yaml_config['llm_config']['answer_summary'])))

    def _init_table_desc(self, table_file, table_schema_path=None):
        logger.info(f'0. Generate table description.')
        # Create or read table schema
        logger.info("Get table schema...")
        if  table_schema_path is not None and os.path.exists(table_schema_path):
            table_desc = json.load(open(table_schema_path, 'r', encoding='utf8'))
            logger.info(f'Read table schema from {table_schema_path}')
        else:
            table_desc = self.get_table_schema(table_file, table_schema_path, "str")
        #print(table_desc["cell_example"][0])
        return table_desc

    def get_table_schema(self, table_file, save_path=None, cell_example_format='raw'):
        """
        cell_example_format: 'raw' | 'str' | 'markdown' | 'json'
        """
        logger.info('** Step0: Table Schema Generation.')
        role = TableDescriber(llm_config=self.table_llm_config, prompt_template=self.table_desc_prompt)
        table_desc = asyncio.run(role.run(
            json.dumps({"table_file": table_file, "desc_save_path": save_path if save_path is not None else ''},
                       ensure_ascii=False)))
        logger.info(f"table_desc:\n {table_desc}")
        table_desc = json.loads(table_desc.content)

        return table_desc

    def _init_roles(self):
        self.query_planner_role = QueryPlanner(llm_config=self.query_llm_config, prompt_template=self.query_expansion_prompt)
        self.code_generator_role = CodeGenerator(llm_config=self.code_llm_config, prompt_template=self.code_generate_prompt)
        self.answer_formatter_role = AnswerFormatter(llm_config=self.summary_llm_config, prompt_template=self.answer_summary_prompt)
        self.llm = LLMChat(llm_config=self.react_llm_config, prompt_template=self.react_prompt)

    def act_pipeline(self, query, table_schema):
        """ Action pipeline. """

        # Step1: Schema Linking between table and query
        logger.info('** Step1: Schema Refining between table and query.')
        ## 1.1 Query Planning...
        logger.info('1.1 Query Planning...')
        try:
            msg = json.dumps({"query": query, "table_desc": table_schema}, ensure_ascii=False)
            query_rsp = json.loads(asyncio.run(self.query_planner_role.run(msg)).content)
        except Exception as e:
            logger.info(f'Exception: {e}')
            logger.info('Exceed max input lenght! Cutoff to 2000')
            if 'description' in table_schema and len(table_schema['description']) > 2000:
                table_schema['description'] = table_schema['description'][:2000]
            elif 'cell_example' in table_schema:
                del table_schema['cell_example']
            msg = json.dumps({"query": query, "table_desc": table_schema}, ensure_ascii=False)
            query_rsp = json.loads(asyncio.run(self.query_planner_role.run(msg)).content)

        query_analysis = query_rsp['rsp']

        try:
            query_expansions = json.loads(query_analysis)
        except:
            query_expansions = ast.literal_eval(query_analysis)

        ## 1.2 Table schema refinement...
        logger.info('1.2 Table schema refinement...')
        relevant_column_list = []
        type = query_expansions[0]['type']
        row_match_list = query_expansions[0]['row_match_list']

        relevant_column_list.extend(
            [r for q in query_expansions for r in q['relevant_column_list'] if r in table_schema['column_list']])
        relevant_column_list = list(set(relevant_column_list))
        if len(relevant_column_list) > 0:
            refined_table_schema = get_refined_table_schema(table_schema, relevant_column_list, type, row_match_list)

            #print(refined_table_schema)
            raw_cell_example = refined_table_schema.get("cell_example", [])
            if raw_cell_example != []:
                raw_cell_example = raw_cell_example[0]
            else:
                print("cell example not found")
            # Determine format: "string", "markdown" or "json"
            fmt = "struct"
            try:

                if isinstance(raw_cell_example, list) and len(raw_cell_example) == 1 and isinstance(raw_cell_example[0],                                                                              list):
                    flat_example = raw_cell_example[0]
                else:
                    flat_example = raw_cell_example

                df = pd.DataFrame(flat_example)

                if fmt == "markdown":
                    refined_table_schema["cell_example"] = df.to_markdown(index=False)
                elif fmt == "struct":
                    #print(df)
                    refined_table_schema["cell_example"] = {
                        "header": df.columns.tolist(),
                        "rows": df.values.tolist()
                    }
                elif fmt == "str":  # default to string
                    refined_table_schema["cell_example"] = df.to_string()

                #print(refined_table_schema["cell_example"])

            except Exception as e:
                logger.warning(f"Failed to format cell_example ({fmt}): {e}")
                # fallback: keep raw
                refined_table_schema["cell_example"] = raw_cell_example

            logger.info(f'Refined table schema:\n{refined_table_schema}')
        else:
            refined_table_schema = table_schema
            logger.info('Table schema refinement FAILED! Using the complete table schema without refinement.')

        table_zoom = refined_table_schema.get('table_zoom', None)

        # Step2: Programming-assisted Solution Generation
        logger.info('** Step2: Write Python program and execute it to get data.')

        if table_zoom is None:
            msg = json.dumps({"query": query, "table_desc": refined_table_schema}, ensure_ascii=False)
        else:
            msg = json.dumps({"query": query, "table_desc": refined_table_schema, "table_zoom": table_zoom}, ensure_ascii=False)


        # logger.info(f'Code Generation and Execution {code_try} times.')
        code_rsp = json.loads(asyncio.run(self.code_generator_role.run(msg)).content)

        cur, re_time = 1, 2
        while code_rsp['execute_state'] != 0 and cur <= re_time:
            logger.info(f"Warning: code generation error, regenerate {cur} time.")
            last_turn_error=f"""----
In the previous round, the code you wrote was:

{code_rsp['code']}

Unfortunately, this code failed to execute, indicating that there may be some bugs in the code. The error type is:
{code_rsp['error']}

----

Please check for errors in the code and answer again strictly following the above guidelines. Output the correct answer after reflection without additional explanation. 

**User Query**: {query}
**Response**: """
            
            if table_zoom is None:
                msg = json.dumps({"query": str(query), "table_desc": refined_table_schema, "last_turn_error": last_turn_error}, ensure_ascii=False)
            else:
                msg = json.dumps({"query": str(query), "table_desc": refined_table_schema, "last_turn_error": last_turn_error, "table_zoom": table_zoom}, ensure_ascii=False)
            
            code_rsp = json.loads(asyncio.run(self.code_generator_role.run(msg)).content)
            cur += 1

        return refined_table_schema, relevant_column_list, query_rsp,code_rsp

    def execute_qa(self, query, table_file, table_desc_file=None):
        # initial
        start_time = time.time()

        logger.info(f'Query: {query}')
        logger.info(f'Table: {table_file}')

        # Read table and generate a global table schema
        table_desc = self._init_table_desc(table_file, table_desc_file)

        log_item = {
            "question": query,
            "query_analysis_prompt": [],
            "query_analysis_response": [],
            "relevant_column_list": [],
            "code_prompt": [],
            "code_response": [],
            "code_result": [],
        }

        # Start iterative thinking
        logger.info('Start!')
        logger.info('First round of thinking')

        # Original cell_example from table_desc
        raw_cell_example = table_desc.get("cell_example", [])

        # Determine format: "string", "markdown" or "json"
        fmt = "struct"
        if fmt is not None:
            try:
                import pandas as pd

                if isinstance(raw_cell_example, list) and len(raw_cell_example) == 1 and isinstance(raw_cell_example[0],list):
                    flat_example = raw_cell_example[0]
                else:
                    flat_example = raw_cell_example

                df = pd.DataFrame(flat_example)

                if fmt == "markdown":
                    table_desc["cell_example"] = df.to_markdown(index=False)
                elif fmt == "struct":
                    table_desc["cell_example"] = {
                        "header": df.columns.tolist(),
                        "rows": df.values.tolist()
                    }
                elif fmt == "str":  # default to string
                    table_desc["cell_example"] = df.to_string()

                #print(table_desc["cell_example"])
            except Exception as e:
                logger.warning(f"Failed to format cell_example ({fmt}): {e}")
                # fallback: keep raw
                table_desc["cell_example"] = raw_cell_example

        # Continue pipeline with formatted example
        refined_table_schema, relevant_column_list, query_rsp, code_rsp = self.act_pipeline(query, table_desc)
        log_item['query_analysis_prompt'].append(query_rsp['prompt'])
        log_item['query_analysis_response'].append(query_rsp['rsp'])
        log_item['relevant_column_list'].append(relevant_column_list)
        log_item['code_prompt'].append(code_rsp['prompt'])
        log_item['code_response'].append(code_rsp['code_rsp'])
        log_item['code_result'].append(code_rsp['response'])
        log_item['code_exe_stat'] = [0]
        if len(code_rsp['response']) == 0:
            log_item['code_exe_stat'] = [1]

        thinking_round_memory = []
        round_query = query
        
        
        for r in range(1, self.max_react_round+1):
            code_action = copy.deepcopy(code_rsp['code_rsp'])
            try:
                code_action = json.loads(code_action)
            except:
                code_action = ast.literal_eval(code_action)
            if code_rsp['execute_state'] == 0:
                code_action['execute_state'] = 'success'
            else:
                if 'error' in code_rsp:
                    code_action['error_type'] = code_rsp['error']
                code_action['execute_state'] = 'fail'

            thinking_round_memory.append(f"=== Round {r} ===\n**Query**: {round_query}\n**Action**: {code_action}\n**Observation**: {code_rsp['response']}")

            his_thinking = '\n\n'.join(thinking_round_memory) + '\n\n' + f""" === Round {r+1} ===

(Remember! Make sure your brief output always adheres to one of the following two formats:\n\nA. If the answer to the question can be obtained or inferred from  thinking process records, indicating you have completed the task, please output:\n**Thought**: 'I have completed the task'\n**Response**: <str>\n\nB. Otherwise, please further rewrite and generate an **improved and clearer query** of the user's target question `{query}` based on previous thinking without explanation, and point out potential considerations and error prone points that neeed to be noted, making it easier for LLMs to uderstand and analyse, please output:\n**Query**: <str> \n\nSpecial reminder:\n1. If the answer to the question can be obtained or inferred from thinking process records (espcially when observation is valid), indicating you have completed the task.\n2. The rewritten question must not change the original meaning, and should not be freely expressed or add new constraints.\n3. You only need to response with a new Query, do not output Thought, Action and observation!)"""        

            prompt = self.react_prompt.format(table_schema=table_desc, query=query, his_observations=his_thinking)
            logger.info(prompt)
            # Think
            round_think = json.loads(asyncio.run(self.llm.run(prompt)).content)

            logger.info(f'Round {r+1} response')
            logger.info(round_think)
            # Observation
            if "I have completed the task" in round_think or r == self.max_react_round:
                thought_process = '\n\n'.join(thinking_round_memory) + '\n\n' + round_think
                msg = json.dumps({"query": query, "thought_process": thought_process, "table_schema": refined_table_schema}, ensure_ascii=False)
                # final_answer = asyncio.run(self.answer_formatter_role.run(msg)).content
                final_answer = json.loads(asyncio.run(self.answer_formatter_role.run(msg)).content)['response']
                break

            # New query
            round_query = round_think.split('**Query**:')[-1].strip().strip('"')
            refined_table_schema, relevant_column_list, query_rsp, code_rsp = self.act_pipeline(query, table_desc)
            log_item['query_analysis_prompt'].append(query_rsp['prompt'])
            log_item['query_analysis_response'].append(query_rsp['rsp'])
            log_item['relevant_column_list'].append(relevant_column_list)
            log_item['code_prompt'].append(code_rsp['prompt'])
            log_item['code_response'].append(code_rsp['code_rsp'])
            log_item['code_result'].append(code_rsp['response'])

            if len(code_rsp['response']) == 0:
                log_item['code_exe_stat'].append([1])
            else:
                log_item['code_exe_stat'].append([0])
        
        end_time = time.time()
        elapsed_time = round(end_time-start_time, 2)
        final_answer = final_answer.strip()
        log_item['elapsed_time/s'] = elapsed_time
        log_item['thinking_round_memory'] = thinking_round_memory
        logger.info(f"*** Final Answer (elapsed time: {elapsed_time} s) ***")
        logger.info(final_answer)
        
        log_item['response'] = final_answer
        logger.info("*"*10) 
        return final_answer, log_item

    
    def simple_voting(self, query, table_file, table_schema_path, k=5):
        responses = []
        log_items = []
        logger.info(f"Simple Voting! \nQuery: {query}")

        for i in range(k):
            logger.info(f'{i} infer...')
            try:
                response, log_item = self.execute_qa(query, table_file, table_schema_path)
            except Exception as e:
                logger(f'Simple voting error: {e}')
                response = "fail"
                log_item = None

            responses.append(response)
            log_items.append(log_item)

        filtered_responses = []
        filtered_log_items = []
        for response, log_item in zip(responses, log_items):
            if response != "fail":
                filtered_responses.append(response)
                filtered_log_items.append(log_item)

        if not filtered_responses:
            return "fail", None

        frequency = {}
        for response in filtered_responses:
            if response in frequency:
                frequency[response] += 1
            else:
                frequency[response] = 1

        vote_result = max(frequency, key=frequency.get)
        index = filtered_responses.index(vote_result)
        vote_log_item = filtered_log_items[index]
        vote_log_item['vote_answer_list'] = filtered_responses
        vote_log_item['vote_relevant_column_list'] = [i['relevant_column_list'] for i in filtered_log_items]
        
        logger.info(f'Simple Voting: \nQuery: {query}\nAnswers: {filtered_responses}\nFinal Answer: {vote_result}')

        return vote_result, vote_log_item


if __name__ == "__main__":
    config_file = '/agent_config/example.yaml'
    tablezoomer = TableZoomer(config_file, max_react_round=5)

    query = "Did Mr Harari write a book on history?"
    table_file = "data/databench_test/DataBench/test/080_Books/all.csv"
    table_schema_path = "data/databench_test/table_schema/test/080_Books.jsonl"  # Create and save if it does not exist

    answer, log_item = tablezoomer.execute_qa(query, table_file, table_schema_path)


