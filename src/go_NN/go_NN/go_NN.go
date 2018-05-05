package main

import (
    "fmt"
    "go_NN/gates"
    "go_NN/nodes"
    "math/rand"
    "time"
)

func linear_regression_example() {
    // create data arrays and populate them with noisy data; in this case we
    // have the line y = 5.0 * x + 22.4 + e, with e ~ N(0, 1) random errors
    N := 100
    var x_vals [100]float64
    var y_vals [100]float64
    x_i := 0.0
    for i := 0; i < N; i++ {
        x_vals[i] = x_i
        y_vals[i] = 5.0 * x_vals[i] + 22.4 + rand.NormFloat64()
        x_i++ 
    }

    // We now need to setup our neural network. For this we need to define each
    // gate and its corresponding inputs and outputs. Here we define all of
    // the input and output Units...
    var b0 gates.Unit
    var b1 gates.Unit
    var x gates.Unit
    var y gates.Unit
    var b1x gates.Unit
    var y_hat gates.Unit
    var er gates.Unit
    var sqr_er gates.Unit

    // ..and here we define all of the gates and how they connect Units
    multgate := gates.MultiplyGate{ &b1, &x, &b1x }
    addgate := gates.AddGate{ &b1x, &b0, &y_hat }
    subgate := gates.SubGate{ &y, &y_hat, &er }
    powergate := gates.PowerGate{ &er, &sqr_er, 2 }

    // We are now ready to begin training our neural net; we start by defining
    // random initial parameters between -5 and 5.
    b0.Value, b1.Value = rand.Float64() * 5 - 2.5, rand.Float64() * 5 - 2.5
    
    // Next we begin stochastic gradient descent; this is done by randomly
    // picking one of our data points and then forward and backpropogating the
    // network with its values. Following this, we update the beta parameters
    // using the comptued gradients and the learning rate alpha
    alpha := 0.0001
    iters := 100000
    var idx int
    for i := 0; i <= iters; i++ {
        // pick random training point and assign value to x and y
        idx = rand.Intn(100)
        x.Value, y.Value = x_vals[idx], y_vals[idx]
        
        // forward propagation
        multgate.Forward()
        addgate.Forward()
        subgate.Forward()
        powergate.Forward()

        // backward propogation; output units grad is 1 for proper backprop
        sqr_er.Gradient = 1.0
        powergate.Backward()
        subgate.Backward()
        addgate.Backward()
        multgate.Backward()

        // update beta parameters with learning rate alpha and the gradient
        b0.Value = b0.Value - alpha * b0.Gradient
        b1.Value = b1.Value - alpha * b1.Gradient
    }
    
    // final model fit
    fmt.Println("\nLinear Regression Example")
    fmt.Println("\tTraining data: y = 5x + 22.4 + e where e ~ N(0, 1)")
    fmt.Println("\tTraining Epochs:", iters / N, "; learning rate: ", alpha)
    fmt.Println("\tOutput model: y =", b1.Value, "x +", b0.Value)
}


func logistic_regression_example() {
    // An example of the network using logistic regression; here we'll look at
    // the data set of numbers of hours spent studying and if a student passed
    // a test. We initialize the arrays below
    hours := [20]float64{0.50, 0.75, 1.00, 1.25, 1.50,
                         1.75, 1.75, 2.00, 2.25, 2.50,
                         2.75, 3.00, 3.25, 3.50, 4.00,
                         4.25, 4.50, 4.75, 5.00, 5.50}

    pass := [20]float64{0, 0, 0, 0, 0,
                        0, 1, 0, 1, 0,
                        1, 0, 1, 0, 1,
                        1, 1, 1, 1, 1}

    // For this example we will be using a SigmoidNode in addition to our
    // normals gates. First we define all of the units for the inputs to the
    // SigmoidNode. This gives us the basic structure of logistic regression
    var b0 gates.Unit
    var x0 gates.Unit
    var bias gates.Unit

    // However, to optimize we need square error; these units will compute that
    var y gates.Unit
    var er gates.Unit
    var sqr_er gates.Unit

    // Next we create our networks gates; notice that the node is replacing
    // four individual gates. This means we no longer need to manually create
    // and populate these gates.
    signode := nodes.NewSigmoidNode(&b0, &x0, &bias)
    subgate := gates.SubGate{ &y, signode.UOut_s0, &er }
    powergate := gates.PowerGate{ &er, &sqr_er, 2 }

    // set initial random value for the betas; typical for new nets
    signode.Beta0.Value, signode.Bias.Value = rand.Float64() * 5 - 2.5, rand.Float64() * 5 - 2.5

    // Now we will use our net to perform stochastic gradient descent and find
    // parameters for our logistic regression
    var idx int
    alpha := 0.1
    iters := 100000
    for i := 0; i <= iters; i++ {
        // pick random training point and assign value to x and y
        idx = rand.Intn(20)
        signode.X0.Value, y.Value = hours[idx], pass[idx]
        
        // forward propagation
        signode.Forward()
        subgate.Forward()
        powergate.Forward()

        // backward propogation
        sqr_er.Gradient = 1.0
        powergate.Backward()
        subgate.Backward()
        signode.Backward()

        // update the beta parameters
        signode.Beta0.Value = signode.Beta0.Value - alpha * signode.Beta0.Gradient
        signode.Bias.Value = signode.Bias.Value - alpha * signode.Bias.Gradient
    }
    fmt.Println("\nLogistic Regression Example")
    fmt.Println("\tNeural network trained for model y = sigmoid(b0 + b1 * x)")
    fmt.Println("\tTrained on students passing a test with x hours of studying")
    fmt.Println("\tTraining Epochs:", 100000/20, "learning rate:", alpha)
    fmt.Println("\tOutput Model: y = sigmoid(", signode.Beta0.Value, "x +", signode.Bias.Value, ")")
}

func sum_example() {
    x0 := gates.Unit{ 10, 0 }
    x1 := gates.Unit{ 2, 0 }
    x2 := gates.Unit{ 2, 0 }
    inputs := []*gates.Unit{ &x0, &x1, &x2 }
    snode := nodes.NewSumNode(inputs)
    snode.Forward()
    snode.Intermediates[1].Gradient = 1.0
    snode.Backward()
    fmt.Println(snode.Intermediates[1], snode.Inputs[0])
}

func main() {
    // random seed for new results
    rand.Seed(time.Now().Unix())
    
    // example networks
    linear_regression_example()
    logistic_regression_example()
    sum_example()
}

