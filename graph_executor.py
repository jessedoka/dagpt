import asyncio
import json
from typing import Dict, Any, Set
from loguru import logger
from ollama import AsyncClient
from models import Graph, Node

# Initialise the logger
logger.add("log/app.log")

class GraphExecutor:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.processing_nodes: Set[str] = set()
        self.completed_nodes: Set[str] = set()
        self.node_results: Dict[str, Any] = {}

        self.lock = asyncio.Lock()  # Thread lock to ensure thread safety

    async def execute_node(self, node: Node, retry: int = 1):
        logger.info(f"Executing node {node.id}")

        dependency_results = await self._get_dependency_results(node)

        logger.info(f"Dependency results for node {node.id}: {dependency_results}")

        prompt = node.build_prompt(dependency_results)
        try:
            response = await self._call_llm(prompt)
        except Exception as e:
            if retry > 0:
                logger.warning(f"Rate limit error for node {node.id}, retrying... ({retry} retries left)")
                await asyncio.sleep(1)  # Wait before retrying
                return await self.execute_node(node, retry - 1)
            else:
                logger.error(f"Rate limit error for node {node.id}, no retries left")
                raise e

        # TODO: 2. Parse the output into a dictionary Dict[str, Any]

        try:
            # Extract the output from the response
            output = json.loads(response["message"]["content"])
            logger.info(f"Response for node {node.id}: {output}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse response for node {node.id}: {e}")
            raise e

        # TODO: 3. Validate the output against the JSON schema

        validation = node.validate_output(output)
        if validation is True:
            logger.info(f"Output for node {node.id} validated successfully \n {output}")
        else:
            logger.error(f"Output for node {node.id} failed validation \n {validation}")
            return

        # TODO: 4. Store the output

        async with self.lock:
            self.node_results[node.id] = output
            logger.info(f"Node {node.id} completed with result: {output}")

    async def execute_graph(self) -> Dict[str, Any]:
        start_node = self.graph.nodes[self.graph.start_node]
        if not start_node:
            logger.error("Start node not found")
            return {}
        await self._execute_subgraph(start_node)
        return self.node_results

    async def _execute_subgraph(self, node: Node):
        logger.info(f"Starting execution of subgraph for node {node.id}")


        # Step 1: Check if the node is already processed or being processed
        async with self.lock:
            if node.id in self.completed_nodes:
                logger.info(f"Node {node.id} is already completed")
                return
            if node.id in self.processing_nodes:
                logger.info(f"Node {node.id} is already being processed, waiting for completion")
                while node.id in self.processing_nodes:
                    await asyncio.sleep(0.01)
                return

        # Step 2: Check if the node has dependencies and wait for them to complete
        if node.dependencies:
            logger.info(f"Waiting for dependencies for node {node.id}")
            await self._wait_for_dependencies(node)

        async with self.lock:
            self.processing_nodes.add(node.id)

        # Step 3: Execute the node
        try:
            await self.execute_node(node, retry=1)
        except Exception as e:
            logger.error(f"Failed to execute node {node.id}: {e}")
            raise e
        finally:
            async with self.lock:
                self.processing_nodes.discard(node.id)
                self.completed_nodes.add(node.id)

        # Step 4: Execute the next nodes it should run x amount of nodes in parallel

        tasks = []
        for next_node_id in node.next_nodes:
            next_node = self.graph.nodes[next_node_id]

            if next_node:
                tasks.append(self._execute_subgraph(next_node))
            else:
                logger.error(f"Next node {next_node_id} not found")


        # Run the tasks in parallel
        await asyncio.gather(*tasks)


    async def _wait_for_dependencies(self, node: Node):
        while True:
            async with self.lock:
                # are all dependencies completed? yes, then break
                if all(dep in self.completed_nodes for dep in node.dependencies):
                    break
            await asyncio.sleep(0.1)

    async def _get_dependency_results(self, node: Node) -> Dict[str, Any]:
        async with self.lock:
            # Return the results of the dependencies that are already completed
            return {
                dep: self.node_results[dep]
                for dep in node.dependencies
                if dep in self.node_results
            }

    async def _call_llm(self, prompt: str):
        return await AsyncClient().chat(
            model="mistral",  # using mistral as other models might be less accurate
            messages=[{"role": "user", "content": prompt}],
            format="json",  # Return the response in JSON format
        )


async def run_graph(graph: Graph) -> Dict[str, Any]:
    logger.info(f"Running graph: {graph.id}")
    executor = GraphExecutor(graph)
    return await executor.execute_graph()



if __name__ == "__main__":

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

    my_graph = Graph(
        id="apple_esg_report",
        input={},
        nodes=nodes,
        start_node="node_1",
    )

    result = asyncio.run(run_graph(my_graph))
    logger.info(f"Result: {result}")

