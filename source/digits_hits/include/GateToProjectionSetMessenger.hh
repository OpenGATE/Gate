/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateToProjectionSetMessenger_h
#define GateToProjectionSetMessenger_h 1

#include "GateOutputModuleMessenger.hh"

class GateToProjectionSet;

class GateToProjectionSetMessenger: public GateOutputModuleMessenger
{
  public:
    GateToProjectionSetMessenger(GateToProjectionSet* gateToProjectionSet);
   ~GateToProjectionSetMessenger();

    void SetNewValue(G4UIcommand*, G4String);

  protected:
    GateToProjectionSet*     m_gateToProjectionSet;

    G4UIcmdWithAString*     	SetFileNameCmd;
    G4UIcmdWithADoubleAndUnit*  PixelSizeXCmd;
    G4UIcmdWithADoubleAndUnit*  PixelSizeYCmd;
    G4UIcmdWithAnInteger*     	PixelNumberXCmd;
    G4UIcmdWithAnInteger*     	PixelNumberYCmd;
    G4UIcmdWithAString*     	projectionPlaneCmd;
    G4UIcmdWithAString*         SetInputDataCmd; //!< The UI command "set input data name"
    G4UIcmdWithAString*         AddInputDataCmd; //!< The UI command "add input data name"
};

#endif
