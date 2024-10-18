import tensorflow as tf
import api_pb2

class TensorArray:
    def __init__(self, dtype: tf.DType, size: int, dynamic_size: bool = False):
        self.handle = tf.raw_ops.TensorArrayV3(
            dtype=dtype, size=size, dynamic_size=dynamic_size
        )

    def write(self, index: int, value: tf.Tensor):
        tf.raw_ops.TensorArrayWriteV3(
            handle=self.handle, index=index, value=value, flow_in=self.handle.flow
        )

    def read(self, index: int) -> tf.Tensor:
        return tf.raw_ops.TensorArrayReadV3(
            handle=self.handle, index=index, flow_in=self.handle.flow
        )