/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATERECORDER_BASE_H_
#define GATERECORDER_BASE_H_

#include "G4Run.hh"
#include "G4Event.hh"
#include "G4Track.hh"
#include "G4Step.hh"

class GateVVolume;

//---------------------------------------------------------------------------
class GateRecorderBase {

public:

  virtual ~GateRecorderBase() {};
  virtual void RecordBeginOfRun(const G4Run*) = 0;
  virtual void RecordEndOfRun(const G4Run*) = 0;
  virtual void RecordBeginOfEvent(const G4Event*) {};
  virtual void RecordEndOfEvent(const G4Event*) {};
  virtual void RecordTrack(const G4Track*) {};
  virtual void RecordStepWithVolume(const GateVVolume * , const G4Step *) {};

};
//---------------------------------------------------------------------------

#endif
