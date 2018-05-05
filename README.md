# go_NN
Simple Neural Network in Go

## Project Structure

```
.
├── LICENSE
├── README.md
└── src
    └── go_NN
        ├── gates
        │   └── gates.go
        ├── go_NN
        │   └── go_NN.go
        ├── graphs
        │   └── graphs.go
        └── nodes
            └── nodes.go

```

### `gates.go`
Functions and structures that facilitate the network's training.

### `go_NN.go`
Generates data required for linear fit. Makes calls to forward and backwards propagation and performs stochastic gradient descent.

### `graphs.go`
UNDER CONSTRUCTION

### `nodes.go`
DESCRIPTION NEEDED

## Example Usage 

```
$ go build src/go_NN/go_NN/go_NN.go
$ ./go_NN
```

## License
This project is licensed under the MIT License - see the `LICENSE` file for details
