package gates

import "math"

// basic unit type to recreate the simpe neural network
type Unit struct {
    Value float64
    Gradient float64
}


// MultiplyGate structure and traits
type MultiplyGate struct {
    U0 *Unit
    U1 *Unit
    UOut *Unit
}

func (g MultiplyGate) Forward() {
    g.U0.Gradient, g.U1.Gradient = 0.0, 0.0
    g.UOut.Value, g.UOut.Gradient = g.U0.Value * g.U1.Value, 0.0
}

func (g *MultiplyGate) Backward() {
    g.U0.Gradient = g.U0.Gradient + g.UOut.Gradient * g.U1.Value
    g.U1.Gradient = g.U1.Gradient + g.UOut.Gradient * g.U0.Value
}


// AddGate structure and traits
type AddGate struct {
    U0 *Unit
    U1 *Unit
    UOut *Unit
}

func (g AddGate) Forward() {
    g.U0.Gradient, g.U1.Gradient = 0.0, 0.0
    g.UOut.Value, g.UOut.Gradient = g.U0.Value + g.U1.Value, 0.0
}

func (g AddGate) Backward() {
    g.U0.Gradient = g.U0.Gradient + g.UOut.Gradient * 1.0
    g.U1.Gradient = g.U1.Gradient + g.UOut.Gradient * 1.0
}


// SubGate structure and traits
type SubGate struct {
    U0 *Unit
    U1 *Unit
    UOut *Unit
}

func (g SubGate) Forward() {
    g.U0.Gradient, g.U1.Gradient = 0.0, 0.0
    g.UOut.Value, g.UOut.Gradient = g.U0.Value - g.U1.Value, 0.0
}

func (g SubGate) Backward() {
    g.U0.Gradient = g.U0.Gradient + g.UOut.Gradient * 1.0
    g.U1.Gradient = g.U1.Gradient + g.UOut.Gradient * -1.0
}


// PowerGate structure and triats
type PowerGate struct {
    U0 *Unit
    UOut *Unit
    Power float64
}

func (g PowerGate) Forward() {
    g.U0.Gradient = 0.0
    g.UOut.Value, g.UOut.Gradient = math.Pow(g.U0.Value, g.Power), 0.0 
}

func (g PowerGate) Backward() {
    g.U0.Gradient = g.U0.Gradient + g.UOut.Gradient * g.Power * math.Pow(g.U0.Value, g.Power - 1.0)
}


// Sigmoid gate for nonlinearity
func sigmoid(x float64) float64 {
    return 1 / ( 1 + math.Exp(-1.0 * x))
}

type SigmoidGate struct {
    U0 *Unit
    UOut *Unit
}

func (g SigmoidGate) Forward() {
    g.U0.Gradient = 0.0
    g.UOut.Value, g.UOut.Gradient = sigmoid(g.U0.Value), 0.0
}

func (g SigmoidGate) Backward() {
    g.U0.Gradient = g.U0.Gradient + g.UOut.Gradient * sigmoid(g.U0.Value) * (1 - sigmoid(g.U0.Value))
}

