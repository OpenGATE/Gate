/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*!
  \class  GateMaterialFilter
*/

#ifndef GATEMATERIALFILTER_HH
#define GATEMATERIALFILTER_HH

#include "GateVFilter.hh"
#include "GateActorManager.hh"
#include "GateMaterialFilterMessenger.hh"

class  GateMaterialFilter : 
  public GateVFilter
{
public:
  GateMaterialFilter(G4String name);
  virtual ~GateMaterialFilter();

  FCT_FOR_AUTO_CREATOR_FILTER(GateMaterialFilter)

  virtual G4bool Accept(const G4Step*);
  virtual G4bool Accept(const G4Track*);
  void Add(const G4String& materialName);
  virtual void show();

private:
 std::vector<G4String> theMdef;
 GateMaterialFilterMessenger * pMatMessenger;
 
 int nFilteredParticles;
};

MAKE_AUTO_CREATOR_FILTER(materialFilter,GateMaterialFilter)

#endif /* end #define GATEMATERIALFILTER_HH */
