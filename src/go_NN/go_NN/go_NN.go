package main

import (
    "fmt"
    "go_NN/gates"
)

// main function for neural network testing
func main() {
    // define all the wires
    x, y, z := gates.Unit{ 10, 0 }, gates.Unit{5, 0}, gates.Unit{26, 0}
    xy, zxy := gates.Unit{}, gates.Unit{}

    // create the gates
    addgate0 := gates.AddGate{ &x, &y, &xy }
    multgate0 := gates.MultiplyGate{ &z, &xy, &zxy }
    
    // perform one pass of the network
    addgate0.Forward()
    multgate0.Forward()
    zxy.Gradient = 1.0
    multgate0.Backward()
    addgate0.Backward()

    // print the gradient and values
    fmt.Println("\nSimple example of an add and multiply gates output: z * (x + y)")
    fmt.Println("\tx:", x.Value, ", y:", y.Value, " z:", z.Value)
    fmt.Println("\txy:", xy.Value, ", zxy:", zxy.Value)
    fmt.Println("\tz.grad:", z.Gradient)
}

