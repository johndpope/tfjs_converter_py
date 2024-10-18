from typing import Dict, List, Union
import tensorflow as tf
import api_pb2

NamedTensorMap = Dict[str, tf.Tensor]
NamedTensorsMap = Dict[str, List[tf.Tensor]]
TensorArrayMap = Dict[int, 'TensorArray']
TensorListMap = Dict[int, 'TensorList']
HashTableMap = Dict[int, 'HashTable']

class TensorInfo:
    def __init__(self, proto: api_pb2.TensorInfo):
        self.proto = proto

    @property
    def name(self) -> str:
        return self.proto.name

    @property
    def dtype(self) -> tf.DType:
        return tf.dtypes.as_dtype(self.proto.dtype)

    @property
    def shape(self) -> tf.TensorShape:
        return tf.TensorShape([dim.size for dim in self.proto.tensor_shape.dim])
    


class GraphModel:

    def __init__(self):
        self.structured_outputs = None
        
    def predict(self, inputs):
        outputs = self.execute(inputs)
        return self.addStructuredOutputNames(outputs)
        
    async def predictAsync(self, inputs):
        outputs = await self.executeAsync(inputs)  
        return self.addStructuredOutputNames(outputs)

    def getIntermediateTensors(self):
        return self.executor.getIntermediateTensors()
        
    def disposeIntermediateTensors(self):
        self.executor.disposeIntermediateTensors()
        
    def addStructuredOutputNames(self, outputs):
        if self.structured_outputs:
            # Logic to add structured output names
            pass
        return outputs

class GraphExecutor:

    def __init__(self):
        self.keep_intermediate_tensors = False
        self.intermediate_tensors = {}
        
    def execute(self, inputs):
        # Updated logic to handle structured outputs
        # and keep intermediate tensors if needed
        pass
        
    async def executeAsync(self, inputs):
        # Similar updates as execute()
        pass

