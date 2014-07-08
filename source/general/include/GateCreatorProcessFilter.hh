/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*
   \class  GateCreatorProcessFilter
   \author pierre.gueth@creatis.insa-lyon.fr
   */

#ifndef GATECREATORPROCESSFILTER_HH
#define GATECREATORPROCESSFILTER_HH

#include "GateVFilter.hh"
#include "GateActorManager.hh"
#include "GateCreatorProcessFilterMessenger.hh"

#include <list>

class  GateCreatorProcessFilter : 
  public GateVFilter
{
  public:
    GateCreatorProcessFilter(G4String name);
    virtual ~GateCreatorProcessFilter();

    FCT_FOR_AUTO_CREATOR_FILTER(GateCreatorProcessFilter)

    virtual G4bool Accept(const G4Track*);

    void AddCreatorProcess(const G4String& processName);

    virtual void show();

  private:

    typedef std::list<G4String> CreatorProcesses;
    CreatorProcesses creatorProcesses;

    GateCreatorProcessFilterMessenger * pMessenger;

};

MAKE_AUTO_CREATOR_FILTER(creatorProcessFilter,GateCreatorProcessFilter)

#endif
