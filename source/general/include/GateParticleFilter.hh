/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateParticleFilter
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GATEPARTICLEFILTER_HH
#define GATEPARTICLEFILTER_HH

#include "GateVFilter.hh"

#include "GateActorManager.hh"

#include "GateParticleFilterMessenger.hh"

#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"

class  GateParticleFilter : 
  public GateVFilter
{
public:
  GateParticleFilter(G4String name);
  virtual ~GateParticleFilter();

  FCT_FOR_AUTO_CREATOR_FILTER(GateParticleFilter)

  virtual G4bool Accept(const G4Track*);

  void Add(const G4String& particleName);
  void AddParent(const G4String& particleName);
 // add the particle into acceptable particle list.
  //
  virtual void show();

private:
 std::vector<G4String> thePdef;
 std::vector<G4String> theParentPdef;
 GateParticleFilterMessenger * pPartMessenger;
 
  int nFilteredParticles;
};

MAKE_AUTO_CREATOR_FILTER(particleFilter,GateParticleFilter)

#endif /* end #define GATEPARTICLEFILTER_HH */
