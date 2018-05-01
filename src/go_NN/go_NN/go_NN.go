package main

import (
    "fmt"
    "go_NN/gates"
    "math/rand"
    "time"
)

func main() {
    // An example of the network using logistic regression; students study/pass
    hours := [20]float64{0.50, 0.75, 1.00, 1.25, 1.50,
                         1.75, 1.75, 2.00, 2.25, 2.50,
                         2.75, 3.00, 3.25, 3.50, 4.00,
                         4.25, 4.50, 4.75, 5.00, 5.50}

    pass := [20]float64{0, 0, 0, 0, 0,
                        0, 1, 0, 1, 0,
                        1, 0, 1, 0, 1,
                        1, 1, 1, 1, 1}

    // define all of the units
    var b0 gates.Unit
    var b1 gates.Unit
    var x gates.Unit
    var y gates.Unit
    var b1x gates.Unit
    var b1x_b0 gates.Unit
    var y_hat gates.Unit
    var er gates.Unit
    var sqr_er gates.Unit

    // define all of our gates
    multgate := gates.MultiplyGate{ &b1, &x, &b1x }
    addgate := gates.AddGate{ &b1x, &b0, &b1x_b0 }
    siggate := gates.SigmoidGate{ &b1x_b0, &y_hat }
    subgate := gates.SubGate{ &y, &y_hat, &er }
    powergate := gates.PowerGate{ &er, &sqr_er, 2 }

    // set initial random value for the betas
    rand.Seed( time.Now().Unix() )
    b0.Value, b1.Value = rand.Float64() * 5 - 2.5, rand.Float64() * 5 - 2.5
    
    // create an index for randomly selecting a variable and begin training
    var idx int
    alpha := 0.01
    for i := 0; i <= 50000; i++ {
        // pick random training point and assign value to x and y
        idx = rand.Intn(20)
        x.Value, y.Value = hours[idx], pass[idx]
        
        // forward propagation
        multgate.Forward()
        addgate.Forward()
        siggate.Forward()
        subgate.Forward()
        powergate.Forward()

        // backward propogation
        sqr_er.Gradient = 1.0
        powergate.Backward()
        subgate.Backward()
        siggate.Backward()
        addgate.Backward()
        multgate.Backward()

        // fmt.Println(x.Value * b1.Value + b0.Value, b1x_b0.Value )
        // fmt.Println(y.Value - y_hat.Value, er.Value)

        // update the beta parameters
        // fmt.Println(b0.Value, b1.Value)
        //fmt.Println("\t", b0.Gradient, b1.Gradient, er.Gradient)
        b0.Value = b0.Value - alpha * b0.Gradient
        b1.Value = b1.Value - alpha * b1.Gradient
    }
    
    fmt.Println(b0.Value, b1.Value)

    // // print the gradient and values
    // fmt.Println("\nSimple example of an add and multiply gates output: z * (x + y)")
    // fmt.Println("\tx:", x.Value, ", y:", y.Value, " z:", z.Value)
    // fmt.Println("\txy:", xy.Value, ", zxy:", zxy.Value)
    // fmt.Println("\tz.grad:", z.Gradient)
}

