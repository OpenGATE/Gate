/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

/*! \class  GateReadoutMessenger
    \brief  Messenger for the GateReadout

    \sa GateReadout, GateReadoutMessenger
*/


#ifndef GateReadoutMessenger_h
#define GateReadoutMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateReadout;

class GateReadoutMessenger : public GateClockDependentMessenger
{
public:
  
  GateReadoutMessenger(GateReadout*);
  ~GateReadoutMessenger();
  
  inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

     inline GateReadout* GetReadout()
       { return m_Readout;}//(GateReadout*) GetDigitizerModule(); }
  
private:
  GateReadout* m_Readout;
  G4UIcmdWithAnInteger*      SetDepthCmd;
  G4UIcmdWithAString*        SetPolicyCmd;
  G4UIcmdWithAString*        SetVolNameCmd;
  G4UIcmdWithABool*          ForceDepthCentroidCmd;
  //G4UIcmdWithAString*		   SetResultingXYCmd;
  //G4UIcmdWithAString*        SetResultingZCmd;


};

#endif








