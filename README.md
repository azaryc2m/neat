![alt text](https://github.com/azaryc2s/neat/blob/master/banner.png "neat")
[![GoDoc](https://godoc.org/github.com/azaryc2s/neat?status.svg)](https://godoc.org/github.com/azaryc2s/neat)
[![Go Report Card](https://goreportcard.com/badge/github.com/azaryc2s/neat)](https://goreportcard.com/report/github.com/azaryc2s/neat)
[![cover.run go](https://cover.run/go/github.com/azaryc2s/neat.svg)](https://cover.run/go/github.com/azaryc2s/neat)

WORKING AGAIN!


NEAT (NeuroEvolution of Augmenting Topologies) is a neuroevolution algorithm by 
Dr. Kenneth O. Stanley which evolves not only neural networks' weights but also their 
topologies. This method starts the evolution process with genomes with minimal structure,
then complexifies the structure of each genome as it progresses. You can read the original
paper from [here](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf).

## Installation
To install `neat` run the following:

```bash
$ go get -u github.com/azaryc2s/neat
```

## Usage

This NEAT package is as simple as plug and play. All you have to do is to create
a new instance of NEAT, given the configuration from a JSON file, for which the
template is provided below, and an evaluation method of a neural network, and 
run.

```json
{
	"experimentName": "",
	"cppnActivations": ["sigmoid"],
	"outputActivation": "sigmoid",
	"verbose": true,
	"numInputs": 0,
	"numOutputs": 0,
	"fullyConnected": true,
	
	"numGenerations": 0,
	"populationSize": 0,
	"tournamentSize": 3,
	"initFitness": 0.0,
	"initConnWeight": 1,
	"survivalRate": 0.0,
	
	"rateCrossover": 0.0,
	"ratePerturb": 0.0,
	"rangeMutWeight": 0.0,
	"capWeight": 0.0,
	"rateAddNode": 0.0,
	"rateAddConn": 0.0,
	"rateEnableConn": 0.0,
	"rateDisableConn": 0.0,
	"rateMutateActFunc": 0.0,
	
	"targetSpecies": 0,
	"stagnationLimit": 0,
	"distanceThreshold": 0,
	"distanceMod": 0.0,
	"minDistanceTreshold": 0,
	"coeffUnmatching": 0,
	"coeffMatching": 0
}
```

Now that you have the configuration JSON file is ready as `config.json`, we can
start experiment with NEAT. Below is an example XOR experiment.

```go
package main

import (
	"log"
	"math"

	// Import NEAT package after installing the package through
	// the instruction provided above.
	"github.com/jinyeom/neat"
)

func main() {

	// First, create a new instance of Config from the JSON file created above.
	// If there's a file import error, the program will crash.
	config, err := neat.NewConfigJSON("config.json")
	if err != nil{
		log.Fatal(err)
	}

	// Then, we can define the evaluation function, which is a type of function
	// which takes a neural network, evaluates its performance, and returns some
	// score that indicates its performance. This score is essentially a genome's
	// fitness score. With the configuration and the evaluation function we
	// defined, we can create a new instance of NEAT and start the evolution 
	// process. The neural network will always maximize the fitness, so if you wish
	// to minimize some fitness value, you have to return 1/fitness in this function.
	// Note: watch out not to return 1/0 which is defined as 'Inf' in Go.
	neat.New(config, neat.XORTest()).Run()
}

```

## License
This package is under GNU General Public License.
