/*----------------------
   OpenGATE Collaboration

   Didier Benoit <benoit@cppm.in2p3.fr>
   Franca Cassol Brunner <cassol@cppm.in2p3.fr>

   Copyright (C) 2009 imXgam/CNRS, CPPM Marseille

   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See GATE/LICENSE.txt for further details
----------------------*/

/*!
  \file GateToImageCTMessenger.hh

  \brief Class GateToImageCTMessenger
  \author Didier Benoit <benoit@cppm.in2p3.fr>
  \author Franca Cassol Brunner <cassol@cppm.in2p3.fr>
*/

#ifndef GATETOIMAGECTMESSENGER_HH
#define GATETOIMAGECTMESSENGER_HH

#include "GateOutputModuleMessenger.hh"

class GateToImageCT;

class GateToImageCTMessenger : public GateOutputModuleMessenger
{
	public:
		GateToImageCTMessenger( GateToImageCT* );
		~GateToImageCTMessenger();

		void SetNewValue( G4UIcommand*, G4String );

	private:
		GateToImageCT* m_gateToImageCT;

		G4UIcmdWithAString* setFileNameCmd;
//		G4UIcmdWithABool* rawOutputCmd;
		G4UIcmdWithAnInteger* setStartSeedCmd;
		G4UIcmdWithAnInteger* vrtFactorCmd;

		//fast simulation
		G4UIcmdWithAnInteger* numFastPixelXCmd;
		G4UIcmdWithAnInteger* numFastPixelYCmd;
		G4UIcmdWithAnInteger* numFastPixelZCmd;

		//fast source simulation
		G4UIcmdWithADoubleAndUnit* detectorInXCmd;
		G4UIcmdWithADoubleAndUnit* detectorInYCmd;
		G4UIcmdWithADoubleAndUnit* sourceDetectorCmd;
};

#endif
