/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateEnergyFilter
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GATEENERGYFILTER_HH
#define GATEENERGYFILTER_HH

#include "GateVFilter.hh"

#include "GateActorManager.hh"

#include "GateEnergyFilterMessenger.hh"

#include "G4UnitsTable.hh"


class GateEnergyFilter : 
  public GateVFilter
{
public:
  GateEnergyFilter(G4String name);
  virtual ~GateEnergyFilter(){}

  FCT_FOR_AUTO_CREATOR_FILTER(GateEnergyFilter)

  virtual G4bool Accept(const G4Step*) ;
  virtual G4bool Accept(const G4Track*);

  void add(const G4String& particleName);
  // add the particle into accepatable particle list.
  //
  void SetEmin(G4double e);
  void SetEmax(G4double e);
  // Set acceptable kinetic energy range.
  //
 
  virtual void show();

private:
  G4double fLowEnergy;
  G4double fHighEnergy;
  GateEnergyFilterMessenger * pEneMessenger;


};

MAKE_AUTO_CREATOR_FILTER(energyFilter,GateEnergyFilter)


#endif /* end #define GATEENERGYFILTER_HH */
