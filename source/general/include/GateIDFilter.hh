/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateIDFilter
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEIDFILTER_HH
#define GATEIDFILTER_HH

#include "GateVFilter.hh"

#include "GateActorManager.hh"

#include "GateIDFilterMessenger.hh"


class  GateIDFilter : 
  public GateVFilter
{
public:
  GateIDFilter(G4String name);
  virtual ~GateIDFilter(){delete pIDMessenger;}

  FCT_FOR_AUTO_CREATOR_FILTER(GateIDFilter)

  virtual G4bool Accept(const G4Track*);

  void addID(G4int id);
  void addParentID(G4int id);
  // add the particle into accepatable particle list.
  //
  virtual void show();

private:
  G4int mID;
  G4int mParentID;
  GateIDFilterMessenger * pIDMessenger;
};

MAKE_AUTO_CREATOR_FILTER(IDFilter,GateIDFilter)

#endif /* end #define GATEIDFILTER_HH */
