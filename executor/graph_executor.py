import tensorflow as tf
from typing import Dict, List

class GraphExecutor:
    def __init__(self, graph, weight_map: Dict[str, tf.Tensor]):
        self.graph = graph
        self.weight_map = weight_map
        self.keep_intermediate_tensors = False
        self.intermediate_tensors = {}

    def execute(self, inputs: Dict[str, tf.Tensor], output_nodes: List[str]) -> List[tf.Tensor]:
        tensor_map = {**self.weight_map, **inputs}
        
        for node in self.graph.nodes:
            if node.name not in tensor_map:
                node_inputs = [tensor_map[input_name] for input_name in node.inputs]
                tensor_map[node.name] = self.executeOp(node, node_inputs)
                
                if self.keep_intermediate_tensors:
                    self.intermediate_tensors[node.name] = tensor_map[node.name]

        return [tensor_map[name] for name in output_nodes]

    async def executeAsync(self, inputs: Dict[str, tf.Tensor], output_nodes: List[str]) -> List[tf.Tensor]:
        # Similar to execute, but handle async operations if needed
        pass

    def executeOp(self, node, inputs: List[tf.Tensor]) -> tf.Tensor:
        # Implement operation execution logic here
        pass

    def getIntermediateTensors(self) -> Dict[str, tf.Tensor]:
        return self.intermediate_tensors

    def disposeIntermediateTensors(self):
        for tensor in self.intermediate_tensors.values():
            tensor.dispose()
        self.intermediate_tensors.clear()