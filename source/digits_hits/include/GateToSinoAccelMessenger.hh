/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateToSinoAccelMessenger_h
#define GateToSinoAccelMessenger_h 1

#include "GateOutputModuleMessenger.hh"

class GateToSinoAccel;

class GateToSinoAccelMessenger: public GateOutputModuleMessenger
{
  public:
    GateToSinoAccelMessenger(GateToSinoAccel* gateToSinoAccel);
   ~GateToSinoAccelMessenger();

    void SetNewValue(G4UIcommand*, G4String);

  protected:
    GateToSinoAccel*     m_gateToSinoAccel;

    G4UIcmdWithAString*         SetFileNameCmd;          //!< The UI command "set file name"
    G4UIcmdWithABool*           TruesOnlyCmd; 		 //!< The UI command "true coincidences only"
    G4UIcmdWithAnInteger*       SetRadialElemNbCmd;  	 //!< The UI command "Number of radial sinogram bins"
    G4UIcmdWithABool*		RawOutputCmd;		 //!< The UI command "enable sinograms raw output"
    G4UIcmdWithADoubleAndUnit*  SetTangCrystalResolCmd;  //!< The UI command "set crystal location blurring FWHM in the tangential direction"
    G4UIcmdWithADoubleAndUnit*  SetAxialCrystalResolCmd; //!< The UI command "set crystal location blurring FWHM in the axial direction"
    G4UIcmdWithAString*         SetInputDataCmd;         //!< The UI command "set input data name"
};

#endif
