from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message
from metagpt.logs import logger
from actions.query_analyse import QueryExpansion


class QueryPlanner(Role):
    """ Assistant to plan and analyze the query. """
    name: str = "TelePlanner"
    profile: str = "Query Planner"
    goal: str = "to deeply understand and carefully analyze user queries, and associate them with the information provided in the table"

    def __init__(self, llm_config, prompt_template, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([QueryExpansion(config=llm_config, PROMPT_TEMPLATE=prompt_template)])
        # self.set_actions([MockAnalyse])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        # By choosing the Action by order under the hood
        todo = self.rc.todo

        msg = self.get_memories(k=1)[0]  # find the most k recent messages
        result = await todo.run(msg.content)
        msg = Message(content=result, role=self.profile, cause_by=type(todo))
        self.rc.memory.add(msg)  
        return msg

