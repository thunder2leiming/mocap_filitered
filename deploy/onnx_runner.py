"""
ONNX Runtime 推理封装。
"""

from pathlib import Path

import numpy as np


class OrtRunner:
    """
    ONNX Runtime 推理封装器。
    自动处理输入维度不匹配的情况（填充或截断），并打印警告。
    """
    def __init__(self, path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        self.expected_dim = self._read_expected_dim(self.input_shape)
        self._shape_warning_printed = False

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        # 处理输入维度不匹配
        if self.expected_dim is not None and x.size != self.expected_dim:
            if x.size < self.expected_dim:
                x = np.pad(x, (0, self.expected_dim - x.size))
                action = "padded"
            else:
                x = x[: self.expected_dim]
                action = "truncated"
            if not self._shape_warning_printed:
                print(f"Warning: {self.input_name} was {action} to ONNX input dim {self.expected_dim}.")
                self._shape_warning_printed = True
        y = self.session.run([self.output_name], {self.input_name: x.reshape(1, -1)})[0]
        return np.asarray(y, dtype=np.float32).reshape(-1)

    @staticmethod
    def _read_expected_dim(shape):
        if not shape:
            return None
        dim = shape[-1]
        return dim if isinstance(dim, int) and dim > 0 else None