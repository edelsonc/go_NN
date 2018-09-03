package layers

import (
    "go_NN/gates"
    "go_NN/nodes"
)

type SumLayer struct {
    Inputs []*gates.Unit
    SumVec []*nodes.SumNode
    UOutVec []*gates.Unit
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
    Betas []*gates.Unit
    ProductVec []*nodes.ProductNode
    DenseVec []*nodes.SumNode
    UOutVec []*gates.Unit
}

func NewDenseLayer( inputs []*gates.Unit, n int, betagen func() float64 ) DenseLayer {
    // TODO Error if n < 1
    n_inputs := len(inputs)
    var betas []*gates.Unit
    var productvec []*nodes.ProductNode
    var densevec []*nodes.SumNode
    var uoutvec []*gates.Unit

    for i := 0; i < n_inputs * n; i++ {
        var beta_i gates.Unit
        beta_i.Value = betagen()
        betas = append(betas, &beta_i)
    }

    for i := 0; i < n; i++ {
        // we need to know what part of the product node to start the slice for
        start_slice := n_inputs * i
        end_slice := n_inputs * ( i + 1 )

        // actually create the nodes now
        pnode := nodes.NewProductNode(betas[start_slice:end_slice], inputs)
        snode := nodes.NewSumNode(pnode.UOutVec)
        productvec = append(productvec, &pnode)
        densevec = append(densevec, &snode)
        uoutvec = append(uoutvec, snode.UOut)
    }
    
    return DenseLayer {
        Inputs: inputs,
        Betas: betas,
        ProductVec: productvec,
        DenseVec: densevec,
        UOutVec: uoutvec,
    }
}

func (l DenseLayer) Forward() {

    for _, node := range l.ProductVec {
        node.Forward()
    }

    for _, node := range l.DenseVec {
        node.Forward()
    }
}

func (l DenseLayer) Backward() {
    for _, node := range l.DenseVec {
        node.Backward()
    }

    for _, node := range l.ProductVec {
        node.Backward()
    }
}

