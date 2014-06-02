/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateDetectionProfileActorMessenger
  \author pierre.gueth@creatis.insa-lyon.fr
*/

#ifndef GATEDETECTIONPROFILEACTORMESSENGER_HH
#define GATEDETECTIONPROFILEACTORMESSENGER_HH

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

#include "GateActorMessenger.hh"
#include "GateImageActorMessenger.hh"

class GateDetectionProfileActor;
class GateDetectionProfilePrimaryTimerActor;

class GateDetectionProfileActorMessenger : public GateImageActorMessenger
{
  public:
    GateDetectionProfileActorMessenger(GateDetectionProfileActor * v);
    virtual ~GateDetectionProfileActorMessenger();
    virtual void SetNewValue(G4UIcommand*, G4String);
  protected:
    GateDetectionProfileActor *actor;
    G4UIcmdWithAString *cmdSetTimer;
    G4UIcmdWithADoubleAndUnit *cmdSetDistanceThreshold;
    G4UIcmdWithADoubleAndUnit *cmdSetDeltaEnergyThreshold;
    G4UIcmdWithAString *cmdSetDetectionPosition;
    G4UIcmdWithABool *cmdSetUseCristalNormal;
    G4UIcmdWithABool *cmdSetUseCristalPosition;
};

class GateDetectionProfilePrimaryTimerActorMessenger : public GateActorMessenger
{
  public:
    GateDetectionProfilePrimaryTimerActorMessenger(GateDetectionProfilePrimaryTimerActor * v);
    virtual ~GateDetectionProfilePrimaryTimerActorMessenger();
    virtual void SetNewValue(G4UIcommand*, G4String);
  protected:
    GateDetectionProfilePrimaryTimerActor *actor;
    G4UIcmdWithAString *cmdAddReportForDetector;
    G4UIcmdWithADoubleAndUnit *cmdSetDetectionSize;
};

#endif
