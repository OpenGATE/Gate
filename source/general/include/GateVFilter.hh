/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \class  GateVFilter
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/



#ifndef GATEVFILTER_HH
#define GATEVFILTER_HH

#include "GateNamedObject.hh"

#include "globals.hh"
#include "G4String.hh"
#include <vector>

#include "G4Step.hh"
#include "G4Track.hh"

#define _FILTER_RETURN_WITH_INVERSION return IsInverted !=

class GateVFilter : 
  public GateNamedObject
{
public:
  GateVFilter(G4String name);
  virtual ~GateVFilter(){}

  virtual G4bool Accept(const G4Step*);
  virtual G4bool Accept(const G4Track*);

 
  virtual void show();

  void setInvert(){IsInverted = true;}
 

private:

protected:
  bool IsInverted = false;

};


#define MAKE_AUTO_CREATOR_FILTER(NAME,CLASS)		\
  class NAME##Creator {					\
  public:						\
    NAME##Creator() {					\
      GateActorManager::GetInstance()->theListOfFilterPrototypes[#NAME]= CLASS::make_filter; } }; \
  static NAME##Creator ActorCreator##NAME;

#define FCT_FOR_AUTO_CREATOR_FILTER(CLASS) \
  static GateVFilter *make_filter(G4String name){ return new CLASS(name); }; \
  using GateVFilter::Accept;

#endif /* end #define GATEVFILTER_HH */
