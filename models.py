from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from jsonschema import validate
from jsonschema.exceptions import ValidationError, SchemaError


class Node(BaseModel):
    id: str
    input: Optional[Dict[str, Any]] = {}  # Input data for the node
    prompt: str
    json_schema: Dict[str, Any]  # Use schema to define the json response format
    next_nodes: List[str]
    dependencies: List[str] = []

    def validate_output(self, output: Dict[str, Any]) -> bool:
        try:
            validate(instance=output, schema=self.json_schema)
            return True
        except ValidationError:
            return {"error": "Validation error", "message": f"Output does not match the schema {self.json_schema}", "output": output}
        except SchemaError as e:
            raise e

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        # Combine input data with node's prompt, and the schema to generate the prompt
        combined_input = {**input_data, **self.input}
        # print(combined_input)
        prompt = self.prompt.format(**combined_input)
        return f"{prompt}\n This is the JSONSchema:\n```JSON\n{self.json_schema}\n```\n Please provide the JSON response according to the schema. spell check your JSON response before submitting."


class Graph(BaseModel):
    id: str
    input: Dict[str, Any]
    nodes: Dict[str, Node]
    start_node: str

