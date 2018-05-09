package layers

import (
    "testing"
    "go_NN/gates"
)

func TestSumLayer(t *testing.T) {
    inputs := make( []*gates.Unit, 10)
    for i, _ := range inputs {
        unit_i := gates.Unit{ float64(i), 0 }
        inputs[i] = &unit_i
    }
    sumlayer := NewSumLayer(inputs, 11)
    sumlayer.Forward()
    if sumlayer.UOutVec[0].Value != 45 {
        t.Error("Did not perform sum of whole group: 45 != ", sumlayer.UOutVec[0])
    } 
}

