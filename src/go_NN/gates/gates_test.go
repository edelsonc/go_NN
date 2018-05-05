package gates

import "testing"

func TestUnit(t *testing.T) {
    aunit := Unit{ 1.0, 0 }
    if aunit.Value != 1 ||  aunit.Gradient != 0 {
        t.Error("Unit Value and Gradient expectd to be 1 and 0, got ", aunit.Value, aunit.Gradient)
    }
}

func TestMultiplyGate(t *testing.T) {
    u0, u1, uout := Unit{ 2.0, 0 }, Unit{ 3.0, 0 }, Unit{}
    m0 := MultiplyGate{&u0, &u1, &uout}
    m0.Forward()

    if uout.Value != 6 || uout.Gradient != 0 {
        t.Error("Multgate did not multiply correctly and update uout: 2 x 3 =", uout.Value)
    }

    uout.Gradient = 1
    m0.Backward()
    if u0.Gradient != 3 || u1.Gradient != 2 {
        t.Error("Backpropogation failure - Did not get other value as gradient")
    }
}

func TestAddGate(t *testing.T) {
    u0, u1, uout := Unit{ 2.0, 0 }, Unit{ 3.0, 0 }, Unit{}
    p0 := AddGate{&u0, &u1, &uout}
    p0.Forward()

    if uout.Value != 5 || uout.Gradient != 0 {
        t.Error("Plusgate did not add correctly and update uout: 2 + 3 =", uout.Value)
    }

    uout.Gradient = 1
    p0.Backward()
    if u0.Gradient != 1 || u1.Gradient != 1 {
        t.Error("Backpropogation failure - Did not get 1 as value of gradient", u0.Gradient, u1.Gradient)
    }
}

func TestSubGate(t *testing.T) {
    u0, u1, uout := Unit{ 2.0, 0 }, Unit{ 3.0, 0 }, Unit{}
    s0 := SubGate{&u0, &u1, &uout}
    s0.Forward()

    if uout.Value != -1 || uout.Gradient != 0 {
        t.Error("Subgate did not subtract correctly and update uout: 2 - 3 =", uout.Value)
    }

    uout.Gradient = 1
    s0.Backward()
    if u0.Gradient != 1 || u1.Gradient != -1 {
        t.Error("Backpropogation failure - Did not get 1 and -1 as value of gradient", u0.Gradient, u1.Gradient)
    }
}

func TestPowerGate(t *testing.T) {
    u0, uout := Unit{ 2.0, 0 }, Unit{}
    pow0 := PowerGate{&u0, &uout, 2}
    pow0.Forward()

    if uout.Value != 4 || uout.Gradient != 0 {
        t.Error("Powergate did not power correctly and update uout: 2 ^ 2 =", uout.Value)
    }

    uout.Gradient = 1
    pow0.Backward()
    if u0.Gradient != 4 {
        t.Error("Backpropogation failure - Did not get 4 as value of gradient", u0.Gradient)
    }
}

func TestSigmoidGate(t *testing.T) {
    u0, uout := Unit{ 0, 0 }, Unit{}
    sig0 := SigmoidGate{&u0, &uout}
    sig0.Forward()

    if uout.Value != 0.5 || uout.Gradient != 0 {
        t.Error("Sigmoid did not evaluate forward correct:", uout.Value)
    }

    uout.Gradient = 1
    sig0.Backward()
    if u0.Gradient != 0.25 {
        t.Error("Backprop did not correctly compute gradient:", u0.Gradient)
    }
}

