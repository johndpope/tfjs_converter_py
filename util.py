

async def loadGraphModel(model_url: str, options: Dict = {}):
    model = GraphModel(model_url, options)
    await model.load()
    return model

def loadGraphModelSync(model_source):
    # Implement synchronous model loading
    pass



def getTensorShape(shape):
    return [dim.size for dim in shape.dim]

def parseDtypeParam(dtype):
    # Map TensorFlow dtype to appropriate Python dtype
    dtype_map = {
        tf.float32: 'float32',
        tf.int32: 'int32',
        tf.bool: 'bool',
        # Add more mappings as needed
    }
    return dtype_map.get(dtype, str(dtype))

def parseNodeName(name):
    parts = name.split(':')
    if len(parts) == 1:
        return parts[0], 0
    return parts[0], int(parts[1])