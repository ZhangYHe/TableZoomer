"""
@Desc: General llm inference.
@Author: xiongsishi
@Date: 2025-05-22.
"""

import json
from metagpt.actions import Action, UserRequirement
from actions.query_analyse import extract_from_content

class LLMGenerate(Action):

    name: str = "ThoughtGenerator"

    async def run(self, prompt: str):

        rsp = await self._aask(prompt)
        rsp = rsp.strip()
        if rsp.startswith("```json") and rsp.endswith("```"):
            rsp = rsp.replace('```json', '').strip()
            rsp = rsp.replace('```', '')
        return json.dumps(extract_from_content(rsp), ensure_ascii=False)