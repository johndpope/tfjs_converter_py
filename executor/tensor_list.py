import tensorflow as tf
import api_pb2

class TensorList:
    def __init__(self, element_shape: tf.TensorShape, element_dtype: tf.DType):
        self.handle = tf.raw_ops.EmptyTensorList(
            element_shape=element_shape, element_dtype=element_dtype
        )

    def push_back(self, tensor: tf.Tensor):
        self.handle = tf.raw_ops.TensorListPushBack(
            input_handle=self.handle, tensor=tensor
        )

    def get_item(self, index: int) -> tf.Tensor:
        return tf.raw_ops.TensorListGetItem(
            input_handle=self.handle, index=index, element_shape=self.handle.element_shape
        )