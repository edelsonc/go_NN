package nodes

import (
    "testing"
    "go_NN/gates"
)

func TestSigmoidNode(t *testing.T) {
    b0 := gates.Unit{ 2.0, 0 }
    x0 := gates.Unit{ 0.5, 0 }
    bias := gates.Unit{ -1.0, 0}

    signode := NewSigmoidNode( &b0, &x0, &bias )
    signode.Forward()
    if signode.UOut.Value != 0.5 {
        t.Error("SigmoidNode output incorrect; expected 0.5, got:", signode.UOut)
    }

    signode.UOut.Gradient = 1
    signode.Backward()
    if bias.Gradient != 0.25 {
        t.Error("backprop has unexpected value: ", bias)
    }
}

func TestSumNode(t *testing.T) {
    x0 := gates.Unit{ 10, 0 }
    x1 := gates.Unit{ 2, 0 }
    x2 := gates.Unit{ 2, 0 }
    x3 := gates.Unit{ 4, 0 }
    inputs := []*gates.Unit{ &x0, &x1, &x2, &x3 }
    snode := NewSumNode(inputs)
    snode.Forward()
    if snode.UOut.Value != 18 {
        t.Error( "Summation incorrect 10 + 2 + 2 + 4 =", snode.UOut.Value )
    }
    
    snode.UOut.Gradient = 1.0
    snode.Backward()
    if x0.Gradient != 1 || x1.Gradient != 1 || x2.Gradient != 1 || x3.Gradient != 1 {
        t.Error("Backprop incorrect:", x0, x1, x2, x3)
    }
}

func TestProductNode(t *testing.T) {
    x0 := gates.Unit{ 2.0, 0.0 }
    b0 := gates.Unit{ 2.0, 0.0 }

    x1 := gates.Unit{ 3.0, 0.0 }
    b1 := gates.Unit{ 10.0, 0.0 }

    bvec := []*gates.Unit{ &b0, &b1 }
    xvec := []*gates.Unit{ &x0, &x1 }

    pnode := NewProductNode(bvec, xvec)
    pnode.Forward()
    if pnode.UOutVec[0].Value != 4 || pnode.UOutVec[1].Value != 30 {
        t.Error("Forward Prop Failed, multiplicatio incorrect", pnode.UOutVec[0])
    }

    pnode.UOutVec[0].Gradient, pnode.UOutVec[1].Gradient = 1.0, 1.0
    pnode.Backward()
    if x0.Gradient != 2.0 || b1.Gradient != 3.0 {
        t.Error("Backprop failed", x0, b0, x1, b1)
    }
}

