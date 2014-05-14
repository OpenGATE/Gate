/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEPROMPTGAMMAPRODUCTIONTLEACTOR_HH
#define GATEPROMPTGAMMAPRODUCTIONTLEACTOR_HH

#include "GateConfiguration.h"
#include "GateVImageActor.hh"
#include "GateActorMessenger.hh"
#include "GatePromptGammaTLEActorMessenger.hh"
#include "GateImageOfHistograms.hh"
#include "GatePromptGammaEnergySpectrumData.hh"

#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

//-----------------------------------------------------------------------------
class GatePromptGammaTLEActor: public GateVImageActor
{
public:
  virtual ~GatePromptGammaTLEActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GatePromptGammaTLEActor)

  virtual void Construct();
  virtual void UserPreTrackActionInVoxel(const int index, const G4Track* t);
  virtual void UserPostTrackActionInVoxel(const int index, const G4Track* t);
  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step);

  void SetInputDataFilename(std::string filename);
  virtual void SaveData();
  virtual void ResetData();

protected:
  GatePromptGammaTLEActor(G4String name, G4int depth=0);
  GatePromptGammaTLEActorMessenger * pMessenger;

  std::string mInputDataFilename;
  GateImageOfHistograms mImageGamma;
  GatePromptGammaEnergySpectrumData data;

};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(PromptGammaTLEActor,GatePromptGammaTLEActor)

#endif // end GATEPROMPTGAMMAPRODUCTIONTLEACTOR
