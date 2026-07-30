from typing import TypedDict

from langgraph.graph import StateGraph, END


from agents.supervisor_agent import supervisor_agent
from agents.flood_agent import flood_agent
from agents.cyclone_agent import cyclone_agent
from agents.report_agent import report_agent



# State definition

class DisasterState(TypedDict):

    X: object

    disaster_type: str

    agent_result: dict

    report: str



# Supervisor Node

def supervisor_node(state):


    agent = supervisor_agent(
        state["disaster_type"]
    )


    return {
        "agent_result": agent
    }



# Flood Node

def flood_node(state):


    result = flood_agent(
        state["X"]
    )


    return {

        "agent_result": result

    }



# Cyclone Node

def cyclone_node(state):


    result = cyclone_agent(
        state["X"]
    )


    return {

        "agent_result": result

    }




# Report Node

def report_node(state):


    report = report_agent(

        state["disaster_type"],

        state["agent_result"]

    )


    return {

        "report": report

    }




# Router

def route_agent(state):


    if state["disaster_type"] == "Flood":

        return "flood"


    elif state["disaster_type"] == "Cyclone":

        return "cyclone"


    else:

        return END




# Create Graph

workflow = StateGraph(
    DisasterState
)



workflow.add_node(
    "supervisor",
    supervisor_node
)


workflow.add_node(
    "flood",
    flood_node
)


workflow.add_node(
    "cyclone",
    cyclone_node
)


workflow.add_node(
    "report",
    report_node
)



workflow.set_entry_point(
    "supervisor"
)



workflow.add_conditional_edges(

    "supervisor",

    route_agent,

    {

        "flood":"flood",

        "cyclone":"cyclone"

    }

)



workflow.add_edge(
    "flood",
    "report"
)


workflow.add_edge(
    "cyclone",
    "report"
)


workflow.add_edge(
    "report",
    END
)



disaster_graph = workflow.compile()