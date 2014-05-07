/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateSecondaryProductionActor
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
 */

#ifndef GATESECONDARYPRODACTOR_HH
#define GATESECONDARYPRODACTOR_HH

#include "GateVImageActor.hh"
#include "GateActorManager.hh"

#include "G4UnitsTable.hh"

#include "GateActorMessenger.hh"


#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"

class GateSecondaryProductionActor : public GateVActor
{
 public:

  //-----------------------------------------------------------------------------
  // Actor name

  virtual ~GateSecondaryProductionActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateSecondaryProductionActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void BeginOfEventAction(const G4Event * event);

  virtual void PreUserTrackingAction(const GateVVolume *, const G4Track*);

 /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();



protected:
  GateSecondaryProductionActor(G4String name, G4int depth=0);
  GateActorMessenger * pMessenger;

  int mCurrentEvent;



  G4String mProdFilename;
  G4String mSecondariesFilename;

  TFile * pTfile;

  TH1F * pFrag;



};

MAKE_AUTO_CREATOR_ACTOR(SecondaryProductionActor,GateSecondaryProductionActor)

#endif
