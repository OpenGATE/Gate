/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEPROMPTGAMMAPRODUCTIONANALOGACTOR_HH
#define GATEPROMPTGAMMAPRODUCTIONANALOGACTOR_HH

#include "GateConfiguration.h"
#include "GateVImageActor.hh"
#include "GateActorMessenger.hh"
#include "GatePromptGammaAnalogActorMessenger.hh"
#include "GateImageOfHistograms.hh"
#include "GatePromptGammaData.hh"


//-----------------------------------------------------------------------------
class GatePromptGammaAnalogActor: public GateVImageActor
{
public:
  virtual ~GatePromptGammaAnalogActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GatePromptGammaAnalogActor)

  virtual void Construct();
  virtual void UserPreTrackActionInVoxel(const int index, const G4Track* t);
  virtual void UserPostTrackActionInVoxel(const int index, const G4Track* t);
  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step);

  void SetInputDataFilename(std::string filename);
  virtual void SaveData();
  virtual void ResetData();

  void SetOutputCount(bool b) { mSetOutputCount = b; }  //output counts instead of yield

protected:
  GatePromptGammaAnalogActor(G4String name, G4int depth=0);
  GatePromptGammaAnalogActorMessenger * pMessenger;

  //we'll not use it, but extract the bins and binsizes for the PG output.
  std::string mInputDataFilename;
  GatePromptGammaData data;

  bool mSetOutputCount;
  bool alreadyHere;

  GateImageOfHistograms * mImageGamma;  //main output (yield)

};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(PromptGammaAnalogActor,GatePromptGammaAnalogActor)

#endif // end GATEPROMPTGAMMAPRODUCTIONANALOGACTOR
