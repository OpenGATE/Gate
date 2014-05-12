/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*!
  \class  GateWashOutActor
  \author I. Martinez-Rovira (immamartinez@gmail.com)
          S. Jan (sebastien.jan@cea.fr)
 */

#ifndef GATEWASHOUTACTOR_HH
#define GATEWASHOUTACTOR_HH

#include "GateVActor.hh"
#include "GateActorManager.hh"
#include "GateActorMessenger.hh"
#include "GateWashOutActorMessenger.hh"
#include "GateVVolume.hh"
#include "GateSourceMgr.hh"
#include "GateVSource.hh"
#include "GateSourceVoxellized.hh"
#include "GateVSourceVoxelReader.hh"


class GateWashOutActor : public GateVActor
{
 public:

  virtual ~GateWashOutActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateWashOutActor)

  virtual void Construct();

  virtual void BeginOfRunAction(const G4Run * r);
  virtual void BeginOfEventAction(const G4Event * event);
  virtual void EndOfEventAction(const G4Event * event);
  virtual void UserSteppingAction(const GateVVolume * /*v*/, const G4Step* /*s*/){};
  virtual void PreUserTrackingAction(const GateVVolume * /*v*/, const G4Track* /*t*/) {};
  virtual void PostUserTrackingAction(const GateVVolume * /*v*/, const G4Track* /*t*/) {};

  G4double GetWashOutModelValue();
  virtual void ReadWashOutTable(G4String fileName);
  G4double ScaleValue(G4double value,G4String unit);

  virtual void SaveData() {};
  virtual void ResetData() {};

  protected:

  GateWashOutActor(G4String name, G4int depth=0);
  GateWashOutActorMessenger * pWashOutActor;
  GateActorMessenger * pActor;

  GateVVolume * v;
  GateVSource * mSourceNow;
  GateVSourceVoxelReader * mSVReader;

  G4int mSourceID;
  G4int mSourceWashOutID;
  G4double mTimeNow;
  G4double mModel;

  std::vector<G4double> mGateWashOutActivityIni;
  std::vector< std::vector<G4double> > mGateWashOutParameters;
  std::vector<G4String> mGateWashOutSources;

};

MAKE_AUTO_CREATOR_ACTOR(WashOutActor,GateWashOutActor)


#endif /* end #define GATEWASHOUTACTOR_HH */
