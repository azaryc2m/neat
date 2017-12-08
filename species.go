// species.go implementation of the species of genomes.
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

// Species is an implementation of species, or niche for speciation of genomes
// that are differentiated by their toplogical differences, measured with
// compatibility distance. Each species is created with a new genome that is not
// compatible with other genomes in the population, i.e., when a genome is not
// compatible with any other species.
type Species struct {
	ID              int       // species ID
	Stagnation      int       // number of generations of stagnation
	Representative  *Genome   // genome that represents this species (permanent)
	BestFitness     float64   // best fitness score in this species at the moment
	BestEverFitness float64   // best fitness this species has ever scored
	SharedFitness   float64   // Shared species fitness
	Offspring       int       // Value representing how many children the species "deserves" to get when reproducing
	Members         []*Genome // member genomes
}

// NewSpecies creates and returns a new instance of Species, given an initial
// genome that will also become the new species' representative.
func NewSpecies(id int, g *Genome) *Species {
	g.SpeciesID = id
	return &Species{
		ID:             id,
		Stagnation:     0,
		Representative: g.Copy(),
		BestFitness:    g.Fitness,
		Members:        []*Genome{g},
	}
}

// Register adds an argument genome as a new member of this species; in
// addition, if the new member genome outperforms this species' best genome, it
// replaces the best genome in this species.
func (s *Species) Register(g *Genome) {
	if g.ID == s.Representative.ID {
		// we ignore it, cause the representative already is added as member
		return
	}
	s.Members = append(s.Members, g)
	g.SpeciesID = s.ID
}

// Flush empties the species membership, except for its representative.
func (s *Species) Flush() {
	//reassign the representative of this species
	s.Representative = s.Members[0]
	s.Members = []*Genome{s.Representative}
	s.Offspring = 0
	s.BestFitness = s.Representative.Fitness
	s.SharedFitness = 0
}
