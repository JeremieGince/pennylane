from itertools import product
import pennylane as qml

def check_clifford_op(op, num_qubits):
    """ Check if an operator is Clifford or not Clifford. """

    pauli_terms = qml.pauli_decompose(qml.matrix(op))
    pauli_group = lambda x: [qml.Identity(x), qml.PauliX(x), qml.PauliY(x), qml.PauliZ(x)]

    pauli_coves = []
    try:
        pauli_qubit = [qml.ops.op_math.prod(*pauli) for pauli in product(*(
            pauli_group(idx) for idx in range(num_qubits)))]
    except: 
        pauli_qubit = [qml.Identity(0)]
    pauli_qubit = [qml.pauli.pauli_sentence(op).hamiltonian(wire_order=range(
        num_qubits)) for op in pauli_qubit]

    for idx, prod in enumerate(product([pauli_terms], pauli_qubit, [pauli_terms])):
        upu = qml.pauli.pauli_sentence(qml.ops.op_math.prod(*prod))
        upu.simplify()
        upu2 = upu.hamiltonian(wire_order=range(num_qubits))
        if len(upu2.ops) == 1:
            if not isinstance(upu2.ops[0], qml.Identity):
                pauli_coves.append(any([
                    qml.equal(upu2.ops[0], tm) for tm in pauli_qubit
                ]))
        else:
            pauli_coves.append(False)

    return all(pauli_coves)