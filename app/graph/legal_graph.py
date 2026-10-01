from langgraph.graph import (StateGraph,START,END)
from app.graph.state import LegalRAGState
from app.graph.nodes import (
    retrieve_node,
    generate_node,
    verify_node,
    rewrite_node
)

from app.graph.router import verification_router

def build_graph():
    graph = StateGraph(
        LegalRAGState
    )
    graph.add_node( "retrieve", retrieve_node)
    graph.add_node( "generate", generate_node)
    graph.add_node( "verify", verify_node)
    graph.add_node( "rewrite", rewrite_node)
    graph.add_edge( START, "retrieve")
    graph.add_edge( "retrieve", "generate")
    graph.add_edge( "generate", "verify")
    graph.add_conditional_edges( "verify", verification_router,{
            "finish": END,
            "retry": "rewrite"
        }
    )
    graph.add_edge( "rewrite", "retrieve")

    return graph.compile()