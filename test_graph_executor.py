from models import Node, Graph
from graph_executor import run_graph, logger, GraphExecutor
import asyncio
import pytest
from copy import deepcopy
import subprocess

nodes = {
    "node_1": Node(
        id="node_1",
        input={
            "text": "In the quaint little village, the houses were painted in a charming array of colours. One cottage stood out with its vibrant red door, welcoming visitors with a warm embrace. Next to it, a serene blue house provided a sense of calm and tranquillity. Across the street, a lively yellow house brought cheer and brightness to the neighbourhood. Not far away, a lush green garden surrounded a modest black cottage, creating a harmonious blend of nature and simplicity. The colourful spectrum of the village painted a picturesque and inviting scene for all who passed through."
        },
        prompt="From the text below, identify the five colours mentioned.\nText: {text}",
        json_schema={
            "type": "object",
            "properties": {
                    "colours": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 5,
                        "maxItems": 5,
                    }
            },
            "required": ["colours"],
        },
        next_nodes=["node_2", "node_3"],
        dependencies=[],
    ),
    "node_2": Node(
        id="node_2",
        input={},
        prompt="From the list of colours: {node_1[colours]} provided, select the two that are considered 'warm' colours.",
        json_schema={
            "type": "object",
            "properties": {
                    "warm_colours": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    }
            },
            "required": ["warm_colours"],
        },
        next_nodes=["node_4"],
        dependencies=["node_1"],
    ),
    "node_3": Node(
        id="node_3",
        input={},
        prompt="From the list of colours: {node_1[colours]} provided, select the two that are considered 'cool' colours.",
        json_schema={
            "type": "object",
            "properties": {
                    "cool_colours": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    }
            },
            "required": ["cool_colours"],
        },
        next_nodes=["node_4"],
        dependencies=["node_1"],
    ),
    "node_4": Node(
        id="node_4",
        input={"question": "Which colour is NOT in either group A nor B?"},
        prompt="Colours given: {node_1[colours]}\nColours from group A: {node_2[warm_colours]}\nColours from group B: {node_3[cool_colours]}\n Question: {question}",
        json_schema={
            "type": "object",
            "properties": {"colour": {"type": "string"}},
            "required": ["colour"],
        },
        next_nodes=[],
        dependencies=["node_1", "node_2", "node_3"],
    ),
}

graph = Graph(id="test_graph", input={}, nodes=nodes, start_node="node_1")


@pytest.mark.asyncio
async def test_node_executor():
    node = deepcopy(nodes["node_1"])
    node.next_nodes = []  # Remove next nodes to test the node executor
    graph_executor = GraphExecutor(graph)
    await graph_executor.execute_node(node, 0)
    assert graph_executor.node_results["node_1"]["colours"] == [
        "red",
        "blue",
        "yellow",
        "green",
        "black",
    ]

@pytest.mark.asyncio
async def test_subgraph_executor():
    # Test the subgraph executor with node_2 and node_3 as the next nodes make a mock execute_node function so that we can test the subgraph executor

    async def mock_execute_node(node, retry=1):
        if node.id == "node_2":
            graph_executor.node_results["node_2"] = {"warm_colours": ["red", "yellow"]}
        elif node.id == "node_3":
            graph_executor.node_results["node_3"] = {"cool_colours": ["blue", "green"]}
        elif node.id == "node_4":
            graph_executor.node_results["node_4"] = {"colour": "black"}
        return
    
    node = deepcopy(nodes["node_1"])
    graph_executor = GraphExecutor(graph)
    graph_executor.execute_node = mock_execute_node

    await graph_executor._execute_subgraph(node)
    assert graph_executor.node_results["node_4"]["colour"] == "black"

@pytest.mark.asyncio
async def test_graph_executor():
    result = await asyncio.wait_for(run_graph(graph), timeout=30)
    assert result["node_4"]["colour"].lower() == "black"
    logger.info("Test passed!")

@pytest.mark.asyncio
async def test_execute_subgraph_no_dependencies():
    node = deepcopy(nodes["node_1"])
    node.next_nodes = []  # Remove next nodes to isolate the test
    executor = GraphExecutor(graph)
    
    await executor._execute_subgraph(node)
    
    assert node.id in executor.completed_nodes
    assert node.id not in executor.processing_nodes

# @pytest.mark.asyncio
# async def test_execute_subgraph_with_dependencies():
#     node = deepcopy(nodes["node_4"])
#     executor = GraphExecutor(graph)
    
#     await executor._execute_subgraph(node)
    
#     assert node.id in executor.completed_nodes
#     assert node.id not in executor.processing_nodes

@pytest.mark.asyncio
async def test_execute_node_with_retry():
    node = deepcopy(nodes["node_1"])
    node.next_nodes = []  # Remove next nodes to isolate the test
    executor = GraphExecutor(graph)

    async def mock_call_llm(prompt):
        raise Exception("Rate limit error")

    executor._call_llm = mock_call_llm

    with pytest.raises(Exception, match="Rate limit error"):
        await executor.execute_node(node, retry=0)

@pytest.mark.asyncio
async def test_wait_for_dependencies():
    node = deepcopy(nodes["node_4"])
    executor = GraphExecutor(graph)

    async def mock_wait_for_dependencies(node):
        await asyncio.sleep(0.1)
        executor.completed_nodes.update(node.dependencies)

    executor._wait_for_dependencies = mock_wait_for_dependencies

    await executor._execute_subgraph(node)
    
    assert node.id in executor.completed_nodes
    assert node.id not in executor.processing_nodes

@pytest.mark.asyncio
async def test_get_dependency_results():
    node = deepcopy(nodes["node_2"])
    executor = GraphExecutor(graph)
    executor.node_results = {
        "node_1": {"colours": ["red", "blue", "yellow", "green", "black"]}
    }

    results = await executor._get_dependency_results(node)
    
    assert results == {"node_1": {"colours": ["red", "blue", "yellow", "green", "black"]}}
