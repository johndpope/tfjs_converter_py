from ..data.types import NamedTensorsMap
import api_pb2


class OperationMapper:
    def __init__(self):
        self.op_mappers = {}

    def transformGraph(self, graph_def, signature=None):
        transformed_nodes = []
        for node in graph_def.node:
            transformed_node = self.mapNode(node)
            transformed_nodes.append(transformed_node)
        
        return Graph(transformed_nodes, signature)

    def mapNode(self, node):
        op_mapper = self.op_mappers.get(node.op, {})
        
        mapped_node = Node(
            name=node.name,
            op=node.op,
            category=op_mapper.get('category'),
            inputs=[],
            inputParams={},
            attrParams={},
            rawAttrs=node.attr
        )

        self.mapInputParams(mapped_node, node, op_mapper)
        self.mapAttrParams(mapped_node, node, op_mapper)

        return mapped_node

    def mapInputParams(self, mapped_node, node, op_mapper):
        for param in op_mapper.get('inputs', []):
            mapped_node.inputParams[param['name']] = {
                'type': param['type'],
                'inputIndexStart': param['start'],
                'inputIndexEnd': param.get('end')
            }

    def mapAttrParams(self, mapped_node, node, op_mapper):
        for param in op_mapper.get('attrs', []):
            attr_value = self.getAttrValue(node, param)
            if attr_value is not None:
                mapped_node.attrParams[param['name']] = {
                    'value': attr_value,
                    'type': param['type']
                }

    def getAttrValue(self, node, param):
        # Implement logic to extract attribute value based on type
        pass