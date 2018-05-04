package nodes

import (
    "go_NN/gates"
    "fmt"
)

type SigmoidNode struct {
    // First Coefficient
    Beta0 *gates.Unit
    X0 *gates.Unit
    Mult0 *gates.MultiplyGate
    UOut_m0 *gates.Unit

    // Second Coefficient
    Beta1 *gates.Unit
    X1 *gates.Unit
    Mult1 *gates.MultiplyGate
    UOut_m1 *gates.Unit

    // Addition Combinati
    Plus0 *gates.AddGate
    UOut_p0 *gates.Unit

    // Bias and Sigmoid
    Bias *gates.Unit
    Plus1 *gates.AddGate
    UOut_p1 *gates.Unit
    Sig0 *gates.SigmoidGate
    UOut_s0 *gates.Unit
}

func NewSigmoidNode(b0 *gates.Unit, x0 *gates.Unit, b1 *gates.Unit, x1 *gates.Unit, bias *gates.Unit ) SigmoidNode {
    // a constructor function for a new SigmoidNode; Expects inputs and creates rest
    uOut_m0 := &gates.Unit{}
    uOut_m1 := &gates.Unit{}
    uOut_p0 := &gates.Unit{}
    uOut_p1 := &gates.Unit{}
    uOut_s0 := &gates.Unit{}

    m0 := &gates.MultiplyGate{ b0, x0, uOut_m0 }
    m1 := &gates.MultiplyGate{ b1, x1, uOut_m1 }
    p0 := &gates.AddGate{ uOut_m0, uOut_m1, uOut_p0 }
    p1 := &gates.AddGate{ uOut_p0, bias, uOut_p1 }
    s0 := &gates.SigmoidGate{ uOut_p1, uOut_s0 }

    return SigmoidNode {
        Beta0: b0,
        X0: x0,
        Mult0: m0,
        UOut_m0: uOut_m0,
        Beta1: b1,
        X1: x1,
        Mult1: m1,
        UOut_m1: uOut_m1,
        Plus0: p0,
        UOut_p0: uOut_p0,
        Bias: bias,
        Plus1: p1,
        UOut_p1: uOut_p1,
        Sig0: s0,
        UOut_s0: uOut_s0,
    }
}

func (n SigmoidNode) Forward() {
    n.Mult0.Forward()
    n.Mult1.Forward()
    n.Plus0.Forward()
    n.Plus1.Forward()
    n.Sig0.Forward()
}

func (n SigmoidNode) Backward() {
    n.Sig0.Backward()
    n.Plus1.Backward()
    n.Plus0.Backward()
    n.Mult1.Backward()
    n.Mult0.Backward()
}


// sum node; just needs to add all of the units together
type SumNode struct {
    Inputs []*gates.Unit
    Intermediates []*gates.Unit
    SumGates []*gates.AddGate
}

func NewSumNode(inputs []*gates.Unit) SumNode {
    N_inputs := len(inputs)
    N_gates := N_inputs - 1
    intermediates := make([]*gates.Unit, N_gates)
    sumgates := make([]*gates.AddGate, N_gates)
    for i:=0; i<N_gates; i++ {
        inter := &gates.Unit{}
        sgate := &gates.AddGate{}
        intermediates[i] = inter
        sumgates[i] = sgate
    }
    
    sumgates[0].U0 = inputs[0]
    sumgates[0].U1 = inputs[1]
    sumgates[0].UOut = intermediates[0]
    for i:=0; i<(N_gates - 1); i++ {
        sumgates[i + 1].U0 = intermediates[i]
        sumgates[i + 1].U1 = inputs[i + 1]
        sumgates[i + 1].UOut = intermediates[i + 1]
    }

    fmt.Println("here")
    return SumNode {
        Inputs: inputs,
        Intermediates: intermediates,
        SumGates: sumgates,
    }
}

func (n SumNode) Forward() {
    for _, gate := range n.SumGates {
        gate.Forward()
    }
}

func (n SumNode) Backward() {
    for i:=len(n.SumGates) - 1; i >= 0; i-- {
        n.SumGates[i].Backward()
    }
}

