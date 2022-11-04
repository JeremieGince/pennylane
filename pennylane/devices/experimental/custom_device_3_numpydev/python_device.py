from typing import List, Tuple, Union

from ..device_interface import *
from .python_simulator import *
from .preprocessor import simple_preprocessor
from pennylane.gradients import param_shift

class TestDevicePythonSim(AbstractDevice):
    "Device containing a Python simulator favouring composition-like interface"

    short_name = "test_py_dev"
    name = "TestDevicePythonSim PennyLane plugin"
    pennylane_requires = 0.1
    version = 0.1
    author = "Xanadu Inc."

    def __init__(self, dev_config: Union[DeviceConfig, None] = None, *args, **kwargs):
        super().__init__(dev_config, *args, **kwargs)
        self._private_sim = PlainNumpySimulator()

    def execute(self, qscript: Union[QuantumScript, List[QuantumScript]]):
        return self._private_sim.execute(qscript)

    def execute_and_gradients(self, qscript: Union[QuantumScript, List[QuantumScript]]):
        #print("EXEC_GRAD")
        #from IPython import embed; embed()
        tmp_tapes, fn = param_shift(qscript[0])
        res = []
        for t in tmp_tapes:
            res.append(self.execute(t))
        return [self.execute(qscript[0]), fn(res)]

    def capabilities(self) -> DeviceConfig:
        if hasattr(self, "dev_config"):
            return self.dev_config
        return {}

    def preprocess(
        self, qscript: Union[QuantumScript, List[QuantumScript]]
    ) -> Tuple[List[QuantumScript], Callable]:
        return simple_preprocessor(qscript)
