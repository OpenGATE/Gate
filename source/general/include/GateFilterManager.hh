/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateFilterManager
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GATEFILTERMANAGER_HH
#define GATEFILTERMANAGER_HH

#include "G4VSDFilter.hh"

#include "GateVFilter.hh"

#include "globals.hh"
#include "G4String.hh"
#include <vector>

#include "G4Step.hh"
#include "G4Track.hh"


class GateFilterManager : 
  public G4VSDFilter
{
public:
  GateFilterManager(G4String name);
  virtual ~GateFilterManager();

  virtual G4bool Accept(const G4Step*) const;
  virtual G4bool Accept(const G4Track*) const;

  void AddFilter(GateVFilter* filter){theFilters.push_back(filter);}
  G4int GetNumberOfFilters(){return theFilters.size();}
  void show();

protected:
  G4String mFilterName;
  std::vector<GateVFilter*> theFilters;

private:
  
};


#endif /* end #define GATEFILTERMANAGER_HH */
