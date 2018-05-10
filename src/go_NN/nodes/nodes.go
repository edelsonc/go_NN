package nodes

import (
    "go_NN/gates"
)

type SigmoidNode struct {
    // First Coefficient
    Beta0 *gates.Unit
    X0 *gates.Unit
    Mult0 *gates.MultiplyGate
    UOut_m0 *gates.Unit

    // Addition Combination
    Plus0 *gates.AddGate
    UOut_p0 *gates.Unit

    // Bias and Sigmoid
    Bias *gates.Unit
    Sig0 *gates.SigmoidGate
    UOut *gates.Unit
}

func NewSigmoidNode(b0 *gates.Unit, x0 *gates.Unit, bias *gates.Unit ) SigmoidNode {
    // a constructor function for a new SigmoidNode; Expects inputs and creates rest
    uOut_m0 := &gates.Unit{}
    uOut_p0 := &gates.Unit{}
    uOut_s0 := &gates.Unit{}

    m0 := &gates.MultiplyGate{ b0, x0, uOut_m0 }
    p0 := &gates.AddGate{ uOut_m0, bias, uOut_p0 }
    s0 := &gates.SigmoidGate{ uOut_p0, uOut_s0 }

    return SigmoidNode {
        Beta0: b0,
        X0: x0,
        Mult0: m0,
        UOut_m0: uOut_m0,
        Plus0: p0,
        UOut_p0: uOut_p0,
        Bias: bias,
        Sig0: s0,
        UOut: uOut_s0,
    }
}

func (n SigmoidNode) Forward() {
    n.Mult0.Forward()
    n.Plus0.Forward()
    n.Sig0.Forward()
}

func (n SigmoidNode) Backward() {
    n.Sig0.Backward()
    n.Plus0.Backward()
    n.Mult0.Backward()
}


// sum node; sum up a slice of units 
type SumNode struct {
    Inputs []*gates.Unit
    Intermediates []*gates.Unit
    SumGates []*gates.AddGate
    UOut *gates.Unit
}

func NewSumNode(inputs []*gates.Unit) SumNode {
    // constructor function for sumnode
    N_inputs := len(inputs)

    // we want to deal with the special case of a single input to the node
    if N_inputs == 1 {
        u1 := gates.Unit{0, 0}
        inputs = append(inputs, &u1)
        N_inputs = len(inputs)
    }

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
        sumgates[i + 1].U1 = inputs[i + 2]
        sumgates[i + 1].UOut = intermediates[i + 1]
    }

    return SumNode {
        Inputs: inputs,
        Intermediates: intermediates,
        SumGates: sumgates,
        UOut: intermediates[len(intermediates) - 1],
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


// product node; take the product of two input slices
type ProductNode struct {
    BetaVec []*gates.Unit
    XVec []*gates.Unit
    UOutVec []*gates.Unit
    MultGateVec []*gates.MultiplyGate
}

func NewProductNode( betavec []*gates.Unit, xvec []*gates.Unit ) ProductNode {
    N_inputs := len(betavec)
    uoutvec := make([]*gates.Unit, N_inputs)
    multgatevec := make([]*gates.MultiplyGate, N_inputs)
    for i, _ := range xvec {
        uout_i := &gates.Unit{}
        mult_i := &gates.MultiplyGate{ betavec[i], xvec[i], uout_i }
        uoutvec[i] = uout_i
        multgatevec[i] = mult_i
    }

    return ProductNode {
        BetaVec: betavec,
        XVec: xvec,
        UOutVec: uoutvec,
        MultGateVec: multgatevec,
    }
}

func (n ProductNode) Forward() {
    for _, m := range n.MultGateVec {
        m.Forward()
    }
}

func (n ProductNode) Backward() {
    for _, m := range n.MultGateVec {
        m.Backward()
    }
}

