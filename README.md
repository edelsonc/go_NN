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
        │   ├── gates.go
        │   └── gates_test.go
        ├── go_NN
        │   └── go_NN.go
        ├── graphs
        │   └── graphs.go
        └── nodes
            ├── nodes.go
            └── nodes_test.go
```

### `gates.go`
Functions and structures that facilitate the network's training.

### `go_NN.go`
Generates data required for linear fit. Makes calls to forward and backwards propagation and performs stochastic gradient descent.

### `graphs.go`
UNDER CONSTRUCTION

### `nodes.go`
Combinations of `gates` to create larger network structure.

## Install and Example Usage 
This package is built using the basic `golang` [package structure](https://golang.org/doc/code.html).
As such, it is important to configure your environment properly by setting your
`GOPATH` varible as the root of your clones package. For `bash` this is done as
follows
```
 $ export GOPATH="/path/to/my/clone/go_NN"
```

Now that you have configured your environmental variables, you need to add you
`bin` directory. This is not included in the project github since we are not
keep executables here. To make the `bin` directory run the following
```
$ mkdir "/path/to/my/clone/go_NN/bin"
```

To build the executable you now move to the directory of the main package
(`go_NN`) and call `go install`
```
$ cd /path/to/my/clone/go_NN/src/go_NN/go_NN
$ go install
```
This should create a new executable `go_NN` in the `bin` directory. You can
call this with `$ ./bin/go_NN`. Additionally, a `pkg` directory will be 
created and populated with the other packages in the project (`gates`, `nodes`, and `graph`).

Alternatively, if you do not wish to deal with this, simply build the executable
from the main package file
```
$ go build src/go_NN/go_NN/go_NN.go
$ ./go_NN
```

## License
This project is licensed under the MIT License - see the `LICENSE` file for details
