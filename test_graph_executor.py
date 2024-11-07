import pytest
from models import Node, Graph
from graph_executor import GraphExecutor
import asyncio
from copy import deepcopy

@pytest.fixture
def basic_nodes():
    return {
        "node_1": Node(
            id="node_1",
            input={"text": "Simple test text with red and blue colors"},
            prompt="Identify colors in text: {text}",
            json_schema={
                "type": "object",
                "properties": {
                    "colours": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    }
                },
                "required": ["colours"],
            },
            next_nodes=["node_2"],
            dependencies=[],
        ),
        "node_2": Node(
            id="node_2",
            input={},
            prompt="Are these warm colors: {node_1[colours]}?",
            json_schema={
                "type": "object",
                "properties": {
                    "warm_colours": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 1,
                    }
                },
                "required": ["warm_colours"],
            },
            next_nodes=[],
            dependencies=["node_1"],
        ),
    }

@pytest.fixture
def basic_graph(basic_nodes):
    return Graph(
        id="test_graph",
        input={},
        nodes=basic_nodes,
        start_node="node_1"
    )


def test_node_initialization(basic_nodes):
    node = basic_nodes["node_1"]
    assert node.id == "node_1"
    assert "text" in node.input
    assert len(node.next_nodes) == 1
    assert len(node.dependencies) == 0

def test_graph_initialization(basic_graph):
    assert basic_graph.id == "test_graph"
    assert len(basic_graph.nodes) == 2
    assert basic_graph.start_node == "node_1"

def test_graph_structure(basic_nodes):
    """Test graph creation and validation"""
    nodes = basic_nodes
    graph = Graph(id="test", input={}, nodes=nodes, start_node="node_1")
    assert graph.id == "test"
    assert len(graph.nodes) == 2
    assert graph.start_node == "node_1"


@pytest.mark.asyncio
async def test_complex_graph_execution(basic_nodes):
    """Test execution of a more complex graph structure"""
    nodes = deepcopy(basic_nodes)
    # Add more complex node relationships
    nodes["node_3"] = Node(
        id="node_3",
        input={},
        prompt="Test",
        json_schema={"type": "object", "properties": {
            "result": {"type": "string"}}},
        next_nodes=["node_4"],
        dependencies=["node_1"]
    )
    nodes["node_4"] = Node(
        id="node_4",
        input={},
        prompt="Test",
        json_schema={"type": "object", "properties": {
            "result": {"type": "string"}}},
        next_nodes=[],
        dependencies=["node_2", "node_3"]
    )
    nodes["node_1"].next_nodes.append("node_3")

    graph = Graph(id="complex_test", input={},
                  nodes=nodes, start_node="node_1")
    executor = GraphExecutor(graph)

    async def mock_execute(node, retry=1):
        executor.node_results[node.id] = {"result": "success"}
        executor.completed_nodes.add(node.id)

    executor.execute_node = mock_execute
    await executor.execute_graph()
    assert len(executor.completed_nodes) == 4

@pytest.mark.asyncio
async def test_node_execution(basic_nodes):
    executor = GraphExecutor(
        Graph(id="test", input={}, nodes=basic_nodes, start_node="node_1"))
    node = basic_nodes["node_1"]

    await executor.execute_node(node)

    assert "node_1" in executor.node_results
    result = executor.node_results["node_1"]
    assert "colours" in result
    assert isinstance(result["colours"], list)

@pytest.mark.asyncio
async def test_dependency_resolution(basic_graph):
    """Test dependency resolution"""
    executor = GraphExecutor(basic_graph)

    # node_1 completion
    executor.node_results["node_1"] = {"colours": ["red", "blue"]}
    executor.completed_nodes.add("node_1")

    # Testing dependency results retrieval
    results = await executor._get_dependency_results(basic_graph.nodes["node_2"])
    assert "node_1" in results
    assert results["node_1"]["colours"] == ["red", "blue"]

@pytest.mark.asyncio
async def test_error_handling():
    # Create a node that will trigger a validation error
    invalid_node = Node(
        id="invalid_node",
        input={"text": "test"},
        prompt="test prompt",
        json_schema={
            "type": "object",
            "properties": {
                "required_field": {"type": "string"}
            },
            "required": ["required_field"]
        },
        next_nodes=[],
        dependencies=[]
    )

    graph = Graph(
        id="error_test",
        input={},
        nodes={"invalid_node": invalid_node},
        start_node="invalid_node"
    )

    executor = GraphExecutor(graph)

    # Mock LLM call to return invalid response
    async def mock_call_llm(prompt):
        return {"message": {"content": '{"wrong_field": "value"}'}}

    executor._call_llm = mock_call_llm

    with pytest.raises(Exception):
        await executor.execute_node(invalid_node)

@pytest.mark.asyncio
async def test_parallel_execution(basic_nodes):
    # Add a parallel node
    nodes = deepcopy(basic_nodes)
    nodes["node_1"].next_nodes = ["node_2", "node_3"]
    nodes["node_3"] = Node(
        id="node_3",
        input={},
        prompt="Another parallel task",
        json_schema={
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"]
        },
        next_nodes=[],
        dependencies=["node_1"]
    )

    graph = Graph(id="parallel_test", input={},
                  nodes=nodes, start_node="node_1")
    executor = GraphExecutor(graph)

    # Mock successful execution for all nodes
    async def mock_execute_node(node, retry=1):
        if node.id == "node_1":
            executor.node_results["node_1"] = {"colours": ["red", "blue"]}
        elif node.id == "node_2":
            executor.node_results["node_2"] = {"warm_colours": ["red"]}
        elif node.id == "node_3":
            executor.node_results["node_3"] = {"result": "success"}

    executor.execute_node = mock_execute_node

    await executor._execute_subgraph(nodes["node_1"])

    assert "node_2" in executor.node_results
    assert "node_3" in executor.node_results


@pytest.mark.asyncio
async def test_concurrent_access(basic_nodes):
    """Test thread-safe operations with multiple concurrent accesses"""
    nodes = basic_nodes
    graph = Graph(id="test", input={}, nodes=nodes, start_node="node_1")
    executor = GraphExecutor(graph)

    async def access_results():
        async with executor.lock:
            executor.node_results["test"] = "value"
            await asyncio.sleep(0.1)
            return executor.node_results["test"]

    # Run multiple concurrent accesses
    results = await asyncio.gather(
        access_results(),
        access_results(),
        access_results()
    )
    assert all(r == "value" for r in results)


@pytest.mark.asyncio
async def test_retry_mechanism(basic_nodes):
    executor = GraphExecutor(
        Graph(id="test", input={}, nodes=basic_nodes, start_node="node_1"))
    node = basic_nodes["node_1"]

    # Mock LLM call to fail once then succeed
    call_count = 0

    async def mock_call_llm(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Rate limit error")
        return {"message": {"content": '{"colours": ["red", "blue"]}'}}

    executor._call_llm = mock_call_llm

    await executor.execute_node(node, retry=1)
    assert call_count == 2
    assert "node_1" in executor.node_results


def test_input_validation(basic_nodes):
    node = basic_nodes["node_1"]
    result = node.validate_output({"colours": ["red", "blue"]})
    assert result is True

    result = node.validate_output({"wrong_field": "value"})
    assert isinstance(result, dict)
    assert "error" in result

@pytest.mark.asyncio
async def test_validation_error(basic_nodes):
    """Test handling of validation errors"""
    nodes = basic_nodes
    graph = Graph(id="test", input={}, nodes=nodes, start_node="node_1")
    executor = GraphExecutor(graph)

    async def mock_invalid_response(prompt):
        return {"message": {"content": '{"invalid": "response"}'}}

    executor._call_llm = mock_invalid_response
    with pytest.raises(Exception):
        await executor.execute_node(nodes["node_1"])

def test_schema_validation(basic_nodes):
    """Test JSON schema validation"""
    node = basic_nodes["node_1"]
    valid_output = {"colours": ["red", "blue"]}
    invalid_output = {"colours": ["red"]}  # Less than minItems

    assert node.validate_output(valid_output) is True
    assert isinstance(node.validate_output(invalid_output), dict)

@pytest.mark.asyncio
async def test_performance():
    """Test performance with larger graphs"""
    # Create a larger graph with 10 nodes
    large_nodes = {}
    for i in range(100):
        large_nodes[f"node_{i}"] = Node(
            id=f"node_{i}",
            input={},
            prompt="Test",
            json_schema={"type": "object"},
            next_nodes=[f"node_{i+1}"] if i < 9 else [],
            dependencies=[]
        )

    graph = Graph(id="performance_test", input={},
                  nodes=large_nodes, start_node="node_0")
    executor = GraphExecutor(graph)

    async def mock_execute(node, retry=1):
        await asyncio.sleep(0.1)  # Simulate work
        executor.node_results[node.id] = {"result": "success"}

    executor.execute_node = mock_execute

    start_time = asyncio.get_event_loop().time()
    await executor.execute_graph()
    end_time = asyncio.get_event_loop().time()

    assert end_time - start_time < 2.0  # Should complete within 2 seconds
    assert len(executor.node_results) == 10
