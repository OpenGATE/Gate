/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateVolumeFilter
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEVOLUMEFILTER_HH
#define GATEVOLUMEFILTER_HH

#include "GateVFilter.hh"

#include "GateActorManager.hh"

#include "GateVolumeFilterMessenger.hh"

#include "GateVVolume.hh"
#include "G4LogicalVolume.hh"

class  GateVolumeFilter : 
  public GateVFilter
{
public:
  GateVolumeFilter(G4String name);
  virtual ~GateVolumeFilter(){delete pVolumeFilterMessenger;}

  FCT_FOR_AUTO_CREATOR_FILTER(GateVolumeFilter)

  virtual G4bool Accept(const G4Step*);
  virtual G4bool Accept(const G4Track*);

  void addVolume(G4String volName);

  virtual void show();

  void Initialize();

private:

  std::vector<G4String> theTempoListOfVolumeName;
  std::vector<GateVVolume *> theListOfVolume;
  std::vector<G4LogicalVolume*> theListOfLogicalVolume;

  bool IsInitialized;

  GateVolumeFilterMessenger * pVolumeFilterMessenger;
};

MAKE_AUTO_CREATOR_FILTER(volumeFilter,GateVolumeFilter)

#endif /* end #define GATEVOLUMEFILTER_HH */
