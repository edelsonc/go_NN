package layers

import (
    "go_NN/gates"
    "go_NN/nodes"
)

type SumLayer struct {
    Inputs []*gates.Unit
    SumVec []*nodes.SumNode
    UOutVec []*gates.Unit
    N int
}

func NewSumLayer(inputs []*gates.Unit, n int) SumLayer {
    n_inputs := len(inputs)
    if n_inputs < n {
        sumnode := nodes.NewSumNode(inputs)
        sumnodes := []*nodes.SumNode{ &sumnode }
        uoutvec := []*gates.Unit{ sumnode.UOut }
        return SumLayer {
            Inputs: inputs,
            SumVec: sumnodes,
            UOutVec: uoutvec,
        }
    }

    var sumvec []*nodes.SumNode
    var uoutvec []*gates.Unit
    for i := 0; i < n_inputs; i += n {
        end := i + n
        if end > n_inputs {
            end = n_inputs
        }
        input_slice := inputs[i:end]
        snode := nodes.NewSumNode(input_slice)
        sumvec  = append(sumvec, &snode)
        uoutvec = append(uoutvec, snode.UOut)
    }
    return SumLayer {
        Inputs: inputs,
        SumVec: sumvec,
        UOutVec: uoutvec,
    }
}

func (l SumLayer) Forward() {
    for _, node := range l.SumVec {
        node.Forward()
    }
}

func (l SumLayer) Backward() {
    for _, node := range l.SumVec {
        node.Backward()
    }
}

type DenseLayer struct {
    Inputs []*gates.Unit
    DenseVec []*nodes.SumNode
    UOutVec []*gates.Unit
    N int
}

func NewDenseLayer( inputs []*gates.Unit, n int ) DenseLayer {
    // TODO Error if n < 1
    var densevec []*nodes.SumNode
    var uoutvec []*gates.Unit
    for i := 0; i < n; i++ {
        snode := nodes.NewSumNode(inputs)
        densevec = append(densevec, &snode)
        uoutvec = append(uoutvec, snode.UOut)
    }
    
    return DenseLayer {
        Inputs: inputs,
        DenseVec: densevec,
        UOutVec: uoutvec,
    }
}

func (l DenseLayer) Forward() {
    for _, node := range l.DenseVec {
        node.Forward()
    }
}

func (l DenseLayer) Backward() {
    for _, node := range l.DenseVec {
        node.Backward()
    }
}

