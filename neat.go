// neat.go implementation of NeuroEvolution of Augmenting Topologies (NEAT).
//
// Copyright (C) 2017  Jin Yeom
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

package neat

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

// NEAT is the implementation of NeuroEvolution of Augmenting Topology (NEAT).
type NEAT struct {
	Config      *Config           // configuration
	Population  []*Genome         // population of genome
	Species     []*Species        // species of subpopulation of genomes
	Activations []*ActivationFunc // set of activation functions
	Evaluation  EvaluationFunc    // evaluation function
	Comparison  ComparisonFunc    // comparison function
	Best        *Genome           // best genome
	Statistics  *Statistics       // statistics

	nextGenomeID  int // genome ID that is assigned to a newly created genome
	nextSpeciesID int // species ID that is assigned to a newly created species
}

// Wrapper to get the indexes of any sortable struct after sort
type SortSlice struct {
	sort.Interface
	idx []int
}

func (s SortSlice) Swap(i, j int) {
	s.Interface.Swap(i, j)
	s.idx[i], s.idx[j] = s.idx[j], s.idx[i]
}

func SortFloat(f []float64) []int {
	fs := &SortSlice{Interface: sort.Float64Slice(f), idx: make([]int, len(f))}
	for i := 0; i < len(f); i++ {
		fs.idx[i] = i
	}
	sort.Sort(fs)
	return fs.idx
}

func MinFloatSlice(fs ...float64) (m float64) {
	m = 0.0
	if len(fs) > 0 {
		m = fs[0]
	} else {
		panic(0)
	}
	for _, e := range fs {
		m = math.Min(m, e)
	}
	return
}

func MinIntSlice(fs ...int) (m int) {
	m = 0
	if len(fs) > 0 {
		m = fs[0]
	} else {
		panic(0)
	}
	for _, e := range fs {
		if e < m {
			m = e
		}
	}
	return
}

// New creates a new instance of NEAT with provided argument configuration and
// an evaluation function.
func New(config *Config, evaluation EvaluationFunc) *NEAT {
	NeatConfig = config
	nextGenomeID := 0
	nextSpeciesID := 0
	Innovation = 1
	rand.Seed(int64(time.Now().Nanosecond()))

	// in order to prevent containing multiple of the same activation function
	// in the set of activation functions, they will temporarily be added to a
	// map first, which contains Sigmoid function as a default, then be
	// transferred to a slice of ActivationFunc.
	temp := map[string]*ActivationFunc{
		"sigmoid": Sigmoid(),
	}

	// if more additional activation functions are needed,
	for _, name := range config.CPPNActivations {
		temp[name] = ActivationSet[name]
	}

	activations := make([]*ActivationFunc, 0, len(temp))
	for _, afunc := range temp {
		activations = append(activations, afunc)
	}

	population := make([]*Genome, config.PopulationSize)
	if config.FullyConnected {
		for i := 0; i < config.PopulationSize; i++ {
			population[i] = NewFCGenome(nextGenomeID, config.NumInputs,
				config.NumOutputs, config.InitFitness)
			nextGenomeID++
		}
		Innovation += len(population[0].ConnGenes)
	} else {
		for i := 0; i < config.PopulationSize; i++ {
			population[i] = NewGenome(nextGenomeID, config.NumInputs,
				config.NumOutputs, config.InitFitness)
			nextGenomeID++
		}
	}

	return &NEAT{
		Config:        config,
		Population:    population,
		Species:       []*Species{},
		Activations:   activations,
		Evaluation:    evaluation,
		Comparison:    NewComparisonFunc(),
		Best:          population[rand.Intn(config.PopulationSize)].Copy(),
		Statistics:    NewStatistics(config.NumGenerations),
		nextGenomeID:  nextGenomeID,
		nextSpeciesID: nextSpeciesID,
	}
}

// Summarize summarizes current state of evolution process.
func (n *NEAT) Summarize(gen int) {
	// summary template
	tmpl := "Gen. %4d | Num. Species: %4d | Best Fitness: %.4f | " +
		"Avg. Fitness: %.4f | Pop. Size: %4d"

	// compose each line of summary and the spacing of separating line
	str := fmt.Sprintf(tmpl, gen, len(n.Species),
		n.Best.Fitness, n.Statistics.AvgFitness[gen], len(n.Population))
	spacing := int(math.Max(float64(len(str)), 80.0))

	for i := 0; i < spacing; i++ {
		fmt.Printf("-")
	}
	fmt.Printf("\n%s\n", str)
	for i := 0; i < spacing; i++ {
		fmt.Printf("-")
	}
	fmt.Println()
}

// Evaluate evaluates fitness of every genome in the population. After the
// evaluation, their fitness scores are recored in each genome.
func (n *NEAT) Evaluate() {
	for _, genome := range n.Population {
		genome.Evaluate(n.Evaluation)
	}
}

// Speciate performs speciation of each genome. The speciation mechanism is as
// follows (from http://nn.cs.utexas.edu/downloads/papers/stanley.phd04.pdf):
//
//	The Genome Loop:
//		Take next genome g from P
//		The Species Loop:
//			If all species in S have been checked:
//				create new species snew and place g in it
//			Else:
//				get next species s from S
//				If g is compatible with s:
//					add g to s
//			If g has not been placed:
//				Species Loop
//		If not all genomes in G have been placed:
//			Genome Loop
//		Else STOP
//
func (n *NEAT) Speciate() {
	// Set the new representative for the species and flush its members
	if len(n.Species) < 1 {
		// initialize the first species with a randomly selected genome
		s := NewSpecies(n.nextSpeciesID, n.Population[rand.Intn(len(n.Population))])
		n.Species = append(n.Species, s)
		n.nextSpeciesID++
	} else {
		for _, s := range n.Species {
			s.Flush()
		}
	}

	// Divide into Species
	for _, genome := range n.Population {
		registered := false
		for i := 0; i < len(n.Species) && !registered; i++ {
			dist := Compatibility(n.Species[i].Representative, genome,
				n.Config.CoeffUnmatching, n.Config.CoeffMatching)

			if dist <= n.Config.DistanceThreshold {
				n.Species[i].Register(genome)
				registered = true
			}
		}

		if !registered {
			n.Species = append(n.Species, NewSpecies(n.nextSpeciesID, genome))
			n.nextSpeciesID++
		}
	}

	//Calculate Shared fitness
	normSum := 0.0
	for _, spec := range n.Species {
		if len(spec.Members) < 1 {
			//TODO: investigate why it's even possible for a species to have 0 members
			fmt.Println("Members empty")
		}
		fitSum := spec.Members[0].Fitness
		spec.BestFitness = spec.Members[0].Fitness
		for i := 1; i < len(spec.Members); i++ {
			fitSum += spec.Members[i].Fitness
			if spec.BestFitness < spec.Members[i].Fitness {
				spec.BestFitness = spec.Members[i].Fitness
			}
		}
		//Get rid of stagnant species by setting their shared fitness
		//to 0, so that they don't get to breed and get removed
		//in the last step
		if spec.BestFitness <= spec.BestEverFitness {
			spec.Stagnation += 1
			if spec.Stagnation >= n.Config.StagnationLimit {
				fitSum = 0
			}
		} else {
			//Reset the stagnation since the species is improving
			spec.Stagnation = 0
			spec.BestEverFitness = spec.BestFitness
		}

		//calculate species fitness
		fitSum /= float64(len(spec.Members))
		spec.SharedFitness = fitSum
		normSum += fitSum
	}

	//Normalize the shared fitness and calculate offspring
	earnedKids := make([]float64, len(n.Species))
	remainder := n.Config.PopulationSize
	for i := 0; i < len(n.Species); i++ {
		n.Species[i].SharedFitness /= normSum
		earnedKids[i] = n.Species[i].SharedFitness * float64(n.Config.PopulationSize)
		n.Species[i].Offspring = int(math.Floor(earnedKids[i]))
		earnedKids[i] -= float64(n.Species[i].Offspring)
		remainder -= n.Species[i].Offspring
	}
	//Sort the array to get the most cheated species by rounding
	//And award them with the remainder rounding error
	idx := SortFloat(earnedKids)
	for r := 0; r < remainder; r++ {
		s := (len(idx) - 1) - r
		//TODO: investigate how it can happen, that there are more remainders than species
		//for now we award the most cheated species even more offspring
		if s >= 0 {
			n.Species[idx[s]].Offspring += 1
		} else {
			n.Species[idx[len(idx)-1]].Offspring += 1
		}
	}

	//remove species that didn't get to make any children
	//means they are stagnant
	for s := len(n.Species) - 1; s >= 0; s-- {
		if n.Species[s].Offspring == 0 {
			n.Species = append(n.Species[:s],
				n.Species[s+1:]...)
		}
	}

}

// Reproduce performs reproduction of genomes in each species. Reproduction is
// performed under the assumption of speciation being already executed. The
// number of eliminated genomes in each species is determined by rate of
// elimination specified in n.Config; after some number of genomes are
// eliminated, the empty space is filled with resulting genomes of crossover
// among surviving genomes. If the number of eliminated genomes is 0 or less
// then 2 genomes survive, every member survives and mutates.
func (n *NEAT) Reproduce() {
	nextGeneration := make([]*Genome, 0, n.Config.PopulationSize)
	for _, s := range n.Species {
		// genomes in this species can inherit to the next generation, if two or
		// more genomes survive in this species, and there is room for more
		// children, i.e., at least one genome must be eliminated.
		numSurvived := int(math.Ceil(float64(len(s.Members)) *
			n.Config.SurvivalRate))

		//Sort the members by their fitness (better first)
		sort.Slice(s.Members, func(i, j int) bool {
			return n.Comparison(s.Members[i], s.Members[j])
		})
		//and kill the weakest
		s.Members = s.Members[:numSurvived]

		// Elitism
		nextGeneration = append(nextGeneration, s.Members[0])

		//start at 1 because we already added the best individual
		for i := 1; i < s.Offspring; i++ {
			//tournament
			perm := make([]int, n.Config.TournamentSize)
			for t := 0; t < n.Config.TournamentSize; t++ {
				perm[t] = rand.Intn(numSurvived)
			}
			sort.Ints(perm)
			//get the minimum index from the random generated slice (best parents)
			p0 := s.Members[perm[0]] // parent 0
			p1 := s.Members[perm[1]] // parent 1

			// create a child from two chosen parents as a result of crossover
			// but only with a given chance, and if the chosen parents are not the same
			child := &Genome{}
			if p0.ID == p1.ID || rand.Float64() > n.Config.RateCrossover {
				child = p0
			} else {
				child = Crossover(n.nextGenomeID, p0, p1, n.Config.InitFitness)
				n.nextGenomeID++
			}

			// this mutations are per item. So for example the chance to mutate
			// a connection is checked for every connection, not just once
			child.MutateDisEnConn(n.Config.RateEnableConn, n.Config.RateDisableConn)
			child.MutatePerturb(n.Config.RatePerturb, n.Config.RangeMutWeight, n.Config.CapWeight)

			//this mutations are checked once, so we check it here
			if rand.Float64() < n.Config.RateAddNode && len(child.ConnGenes) != 0 {
				child.MutateAddNode(n.nextGenomeID, n.randActivationFunc())
				n.nextGenomeID++
			}

			if rand.Float64() < n.Config.RateAddConn {
				child.MutateAddConn()
			}

			if len(child.HiddenNodes) > 0 && rand.Float64() < n.Config.RateMutateActFunc {
				child.MutateActFunc(n.nextGenomeID, n.Activations)
				n.nextGenomeID++
			}
			nextGeneration = append(nextGeneration, child)
		}
	}

	// update the population with the new generation
	n.Population = nextGeneration
}

// randActivationFunc is a helper function that returns a random activation
// function.
func (n *NEAT) randActivationFunc() *ActivationFunc {
	return n.Activations[rand.Intn(len(n.Activations))]
}

// Run executes evolution and return the best genome.
func (n *NEAT) Run() *Genome {
	if n.Config.Verbose {
		n.Config.Summarize()
	}
	n.Evaluate()
	n.Speciate()
	// for each generation
	for i := 0; i < n.Config.NumGenerations; i++ {
		// reproduce children genomes, evaluate and speciate them
		n.Reproduce()
		n.Evaluate()
		n.Speciate()

		//adjust species threshold to reach species number target
		//		if i > 1 {
		//			if len(n.Species
		//		}

		// update the best genome
		for _, genome := range n.Population {
			if n.Comparison(genome, n.Best) {
				n.Best = genome.Copy()
			}
		}

		n.Statistics.Update(i, n)
		if n.Config.Verbose {
			n.Summarize(i)
		}

		//		// eliminate stagnant species
		//		if len(n.Species) > 1 {
		//			var survived []*Species
		//			var remainedPopulation []*Genome
		//			for j := range n.Species {
		//				if n.Species[j].Stagnation <= n.Config.StagnationLimit {
		//					n.Species[j].Stagnation++
		//					survived = append(survived, n.Species[j])
		//					remainedPopulation = append(remainedPopulation, n.Species[j].Members...)
		//				}
		//			}
		//			n.Species = survived
		//			n.Population = remainedPopulation
		//			if len(n.Population) == 0 {
		//				fmt.Println("Everyone died :(")
		//				return n.Best
		//			}
		//		}
	}

	return n.Best
}
