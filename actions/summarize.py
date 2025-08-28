"""
@Desc: query analyse.
@Author: xiongsishi
@Date: 2025-05-25.
"""

import json
from metagpt.actions import Action, UserRequirement
from actions.query_analyse import extract_from_content

class ThoughtSummary(Action):

    name: str = "ThoughtSummary"

    async def run(self, inputs: str):
        inputs = json.loads(inputs)
        query, thought_process, table_schema = inputs['query'], inputs['thought_process'], json.dumps(inputs['table_schema'], indent=4, ensure_ascii=False)
        prompt = self.PROMPT_TEMPLATE.format(query=query, thought_process=thought_process, table_schema=table_schema)

        rsp = await self._aask(prompt)
        # return rsp
        return json.dumps({"prompt": prompt, "response": extract_from_content(rsp)}, ensure_ascii=False)