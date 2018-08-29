package layers

import (
    "testing"
    "go_NN/gates"
)

func All( aslice []bool ) bool {
    // collection function All; returns true if all bools are true
    for _, b := range aslice {
        if !b {
            return false
        }
    }
    return true
}

func TestSumLayerSingleOutput(t *testing.T) {
    inputs := make( []*gates.Unit, 10)
    for i, _ := range inputs {
        unit_i := gates.Unit{ float64(i), 0 }
        inputs[i] = &unit_i
    }
    
    // create a layer that sums more values than are in the input
    sumlayer := NewSumLayer(inputs, 11)
    sumlayer.Forward()
    if sumlayer.UOutVec[0].Value != 45 {
        t.Error("Did not perform sum of whole group: 45 != ", sumlayer.UOutVec[0])
    } 
    
    // set the grad to 1 and check that all the inputs have the correct gradient
    sumlayer.UOutVec[0].Gradient = 1
    sumlayer.Backward()
    
    grad_bool :=  make([]bool, len(inputs))
    for i, u := range inputs {
        check_grad := (u.Gradient == 1)
        grad_bool[i] = check_grad
    }

    if !All(grad_bool) {
        t.Error("Gradient did not backprop correctly:", grad_bool)
    }

}

func TestSumLayerMultipleOutputs(t *testing.T) {
    inputs := make( []*gates.Unit, 10)
    for i, _ := range inputs {
        unit_i := gates.Unit{ float64(i), 0 }
        inputs[i] = &unit_i
    }

    // check the number of output noded created is correct: ceil(10 / 3)
    sumlayer := NewSumLayer(inputs, 3)
    if len(sumlayer.UOutVec) != 4 {
        t.Error("Incorrect number of gates created: ", len(sumlayer.UOutVec))
    }
    
    // check that it performs the sums correctly for each group of three
    sumlayer.Forward()
    correct_vals := []float64{3, 12, 21, 9}
    value_bool := make([]bool, 4)
    for i, cv := range correct_vals {
        check_val := (cv == sumlayer.UOutVec[i].Value)
        value_bool[i] = check_val
    }

    if !All(value_bool) {
        t.Error("Output not as expected: ", value_bool)
    }

    // check backpropogation; should all be 1 since its a sum
    for _, u := range sumlayer.UOutVec {
        u.Gradient = 1
    }
    sumlayer.Backward()
    grad_bool :=  make([]bool, len(inputs))
    for i, u := range inputs {
        check_grad := (u.Gradient == 1)
        grad_bool[i] = check_grad
    }

    if !All(grad_bool) {
        t.Error("Gradient did not backprop correctly:", grad_bool)
    }
}


func TestDenseLayer(t *testing.T) {
    inputs := make( []*gates.Unit, 10)
    check_sum := 0.0
    for i, _ := range inputs {
        check_sum += float64(i)
        unit_i := gates.Unit{ float64(i), 0 }
        inputs[i] = &unit_i
    }

    // check to make sure that when we create a new dense layer with n nodes 
    // we actually get n nodes
    denselayer := NewDenseLayer(inputs, 7)
    if len(denselayer.DenseVec) != 7 || len(denselayer.Inputs) != 10 {
        t.Error("Dense layer did not create the correct number of nodes or input shape is not maintained")
    }

    denselayer.Forward()
    value_bool := make([]bool, 7)
    for i, units := range denselayer.UOutVec {
        check_val := ( check_sum == units.Value )
        value_bool[i] = check_val
    }

    if !All(value_bool) {
        t.Error("Dense did not sum correctly: ", value_bool)
    }
}

