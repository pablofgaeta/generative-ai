# Copyright 2025 Google. This software is provided as-is, without warranty or
# representation for any use or purpose. Your use of it is subject to your
# agreement with Google.

import logging
from typing import Literal

from concierge import schemas as concierge_schemas
from concierge.nodes.task_planning import schemas
from concierge.nodes.task_planning.ops import generate_plan
from langchain_core.runnables import config as lc_config
from langgraph import types as lg_types
from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)


def build_planner_node(
    node_name: str = "planner",
    plan_processor_node_name: str = "executor",
    response_processor_node_name: str = "save-turn",
):
    NextNodeT = Literal[plan_processor_node_name, response_processor_node_name]  # type: ignore

    async def ainvoke(
        state: schemas.PlannerState,
        config: lc_config.RunnableConfig,
    ) -> lg_types.Command[NextNodeT]:  # type: ignore
        """
        Asynchronously generates a plan or a direct response based on the current conversation state.

        This function takes the current conversation state, which includes the user's input and history,
        and uses the `generate_plan` function to determine whether to create a plan for further action
        or to provide a direct response. It then updates the conversation state and directs the flow
        to the appropriate next node (executor or post-processing).

        Runtime configuration should be passed in `config.configurable.planner_config`.

        Args:
            state: The current state of the conversation session, including user input and history.
            config: The LangChain RunnableConfig containing agent-specific configurations.

        Returns:
            A Command object that specifies the next node to transition to (executor or post-processing)
            and the updated conversation state. The state includes the generated plan or response.

        Raises:
            TypeError: If the plan reflection action is of an unsupported type.
        """

        planner_config = schemas.TaskPlannerConfig.model_validate(
            config["configurable"].get("planner_config", {})
        )

        stream_writer = get_stream_writer()

        current_turn = state.get("current_turn")
        assert current_turn is not None, "current turn must be set"

        user_input = current_turn.get("user_input")
        assert user_input is not None, "user input must be set"

        turns = state.get("turns", [])

        plan_reflection = await generate_plan.generate_plan(
            current_turn=current_turn,
            project=planner_config.project,
            region=planner_config.region,
            model_name=planner_config.planner_model_name,
            history=turns,
        )

        next_node = None
        if isinstance(plan_reflection.action, schemas.Plan):
            next_node = plan_processor_node_name

            # Ensure results aren't set
            for task in plan_reflection.action.tasks:
                task.result = None

            # Set initial plan
            current_turn["plan"] = plan_reflection.action
            stream_writer({"plan": plan_reflection.action.model_dump(mode="json")})

        elif isinstance(plan_reflection.action, schemas.Response):
            next_node = response_processor_node_name

            # Update turn response
            current_turn["response"] = plan_reflection.action.response
            stream_writer({"response": plan_reflection.action.response})

        else:
            raise TypeError(
                f"Unsupported plan reflection action: {type(plan_reflection.action)}"
            )

        return lg_types.Command(
            update=schemas.PlannerState(current_turn=current_turn),
            goto=next_node,
        )

    return concierge_schemas.Node(name=node_name, fn=ainvoke)
