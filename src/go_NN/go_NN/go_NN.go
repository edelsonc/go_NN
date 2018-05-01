package main

import (
    "fmt"
    "go_NN/gates"
    "math/rand"
    "time"
)

func main() {
    // set a random seed and them initialize parameters between -5 and 5
    rand.Seed(time.Now().Unix())
    beta_0, beta_1 = rand.Float64() * 10 - 5, rand.Float64() * 10 - 5

    // create data arrays and populate them with noisy data; in this case we
    // have a line with N(0, 1) random errors
    var x_vals [100]float64
    var y_vals [100]float64
    for i := 0; i < 100; i++ {
        x_vals[i] = i
        y_vals[i] = beta_1 * x_vals[i] + beta_0 + rand.NormFloat64()  
    }

    // define units
    var b0 gates.Unit
    var b1 gates.Unit
    var x gates.Unit
    var y gates.Unit
    var b1x gates.Unit
    var y_hat gates.Unit
    var er gates.Unit
    var sqr_er gates.Unit

    // define gates
    multgate := gates.MultiplyGate{ &b1, &x, &b1x }
    addgate := gates.AddGate{ &b1x, &b0, &y_hat }
    subgate := gates.SubGate{ &y, &y_hat, &er }
    powergate := gates.PowerGate{ &er, &sqr_er, 2 }

    // set initial random value for the betas
    b0.Value, b1.Value = rand.Float64() * 10 - 5, rand.Float64() * 10 - 5
    
    // create an index for randomly selecting a variable and begin training
    var idx int
    alpha := 0.01
    for i := 0; i <= 50000; i++ {
        // pick random training point and assign value to x and y
        idx = rand.Intn(100)
        x.Value, y.Value = hours[idx], pass[idx]
        
        // forward propagation
        multgate.Forward()
        addgate.Forward()
        subgate.Forward()
        powergate.Forward()

        // backward propogation
        sqr_er.Gradient = 1.0
        powergate.Backward()
        subgate.Backward()
        addgate.Backward()
        multgate.Backward()

        // update beta parameters with learning rate alpha
        b0.Value = b0.Value - alpha * b0.Gradient
        b1.Value = b1.Value - alpha * b1.Gradient
    }
    
    // final model fit
    fmt.Println(b0.Value, b1.Value)
}

