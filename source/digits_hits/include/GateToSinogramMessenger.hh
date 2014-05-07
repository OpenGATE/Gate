/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*----------------------
   Modifications history

     Gate 6.2

	C. Comtat, CEA/SHFJ, 10/02/2011	   Allows for virtual crystals, needed to simulate ecat like sinogram output for Biograph scanners

----------------------*/

#ifndef GateToSinogramMessenger_h
#define GateToSinogramMessenger_h 1

#include "GateOutputModuleMessenger.hh"

class GateToSinogram;

class GateToSinogramMessenger: public GateOutputModuleMessenger
{
  public:
    GateToSinogramMessenger(GateToSinogram* gateToSinogram);
   ~GateToSinogramMessenger();

    void SetNewValue(G4UIcommand*, G4String);

  protected:
    GateToSinogram*             m_gateToSinogram;

    G4UIcmdWithAString*         SetFileNameCmd;          //!< The UI command "set file name"
    G4UIcmdWithABool*           TruesOnlyCmd; 		 //!< The UI command "true coincidences only in data=0"
    G4UIcmdWithAnInteger*       SetRadialElemNbCmd;  	 //!< The UI command "Number of radial sinogram bins"
    G4UIcmdWithABool*		RawOutputCmd;		 //!< The UI command "enable sinograms raw output"
    G4UIcmdWithADoubleAndUnit*  SetTangCrystalResolCmd;  //!< The UI command "set crystal location blurring FWHM in the tangential direction"
    G4UIcmdWithADoubleAndUnit*  SetAxialCrystalResolCmd; //!< The UI command "set crystal location blurring FWHM in the axial direction"
    G4UIcmdWithAString*         SetInputDataCmd;         //!< The UI command "set input data name"

    // 07.02.2006, C. Comtat, Store randoms and scatters sino
    G4UIcmdWithABool*           StoreDelayedsCmd;        //!< The UI command "store dealayed coincidences in data=1 and prompt coincidences in data=0"
    G4UIcmdWithABool*           StoreScattersCmd;        //!< The UI command "store true scattered coincidences in data=4"
    G4UIcmdWithAString*         SetDataChannelCmd;       //!< The UI command "set input data channel name"

    // C. Comtat, February 2011: Required to simulate Biograph output sinograms with virtual crystals
    G4UIcmdWithAnInteger*       SetVirtualRingCmd;       //!< The UI command "set the number of virtual rings between blocks (Biograph, for example)
    G4UIcmdWithAnInteger*       SetVirtualCrystalCmd;    //!< The UI command "set the number of virtual crystals between radial blocks (Biograph, for example)

};

#endif
