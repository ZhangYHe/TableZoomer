from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message
from metagpt.logs import logger
from actions.table_desc import TableDesc, get_refined_table_schema


class TableDescriber(Role):
    """ Assistant to describe the table. """
    name: str = "TeleDescriber"
    profile: str = "Table Describer"
    goal: str = "to analyze and describe the structure and content of tables, providing comprehensive schema information"

    def __init__(self, llm_config, prompt_template, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([TableDesc(config=llm_config, PROMPT_TEMPLATE=prompt_template)])
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

