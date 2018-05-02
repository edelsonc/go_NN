package graphs

import "go_NN/gates"

type LearningGraph struct {
    Gates []*gates.Gate
    Units []*gates.Unit
    alpha float64
}


