from qiskit import *
from qiskit import Aer, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.opflow import StateFn, PauliSumOp, AerPauliExpectation, ListOp, Gradient
from qiskit.utils import QuantumInstance
from qiskit import IBMQ
provider=IBMQ.load_account()
from qiskit_machine_learning.neural_networks import OpflowQNN
from math import pi, sqrt
import pickle

def neuron(params1, ratio):
    qreg_q = QuantumRegister(1, 'q')
    creg_c = ClassicalRegister(1, 'c')
    qc1 = QuantumCircuit(qreg_q, creg_c)
    backend = provider.get_backend("ibmq_lima")
    qc1.h(0)
    qc1.ry(pi/params1[0], 0)
    qc1.rx(pi/-1 * params1[1], 0)
    qc1.rx(pi/0.99, 0)
    qc1.measure(0, 0)
    res = backend.run(transpile(qc1,backend), shots=100)
    result = res.result()
    out1 = result.get_counts(qc1)
    output = out1['0']/out1['1']
    if output >= ratio:
        circuit = QuantumCircuit(2, 1)
        circuit.ry(pi, 0)
        circuit.cx(0, 1)
        circuit.u(0.5*pi/3, pi/3, 0, 0)
        circuit.u(params1[0]*pi/3, pi/3, 0, 1)
        circuit.cx(0, 1)
        circuit.measure(0, 0)
        res = backend.run(transpile(circuit,backend), shots=1)
        result = res.result()
        out2 = result.get_counts(circuit)
        print(list(out2))
        if list(out2)[0] == "1":
            return [True, output]
        else:
            return [False]
    else:
        return [False]
def avg(input_):
    l = len(input_)
    v = 0
    for i in input_:
        v += i
    return v / l
def new_network(datadir, input_):
    weights =[[[0.5] for i in range(0, 100)] for i in range(0, 10)]
    in1 = [[0.005] for i in range(0, 10)]
    output = []
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            fires = neuron([avg(in1[i]), avg(weights[i][j])], 0.05)
            if fires[0]:
                if i <= 7:
                    in1[i + 1].append(input_)
                else:
                    output.append(input_)
                weights[i][j].append(fires[1]*5)
            else:
                if i > 8:
                    output.append(0)
    file = open(datadir + "/weights.pkl")
    pickle.dump(weights, file)
    return(avg(output))
  
  new_network("ai", 2)
