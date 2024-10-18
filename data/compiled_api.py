import api_pb2
from typing import Dict, List, Union


# You can define convenience functions or wrappers here if needed
def create_node_def(name: str, op: str) -> api_pb2.NodeDef:
    node = api_pb2.NodeDef()
    node.name = name
    node.op = op
    return node

def create_attr_value(value: Union[float, int, str, bool]) -> api_pb2.AttrValue:
    attr = api_pb2.AttrValue()
    if isinstance(value, float):
        attr.f = value
    elif isinstance(value, int):
        attr.i = value
    elif isinstance(value, str):
        attr.s = value.encode()
    elif isinstance(value, bool):
        attr.b = value
    return attr

# Add more helper functions as needed