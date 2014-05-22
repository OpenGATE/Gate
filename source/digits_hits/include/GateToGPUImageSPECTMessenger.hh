/*----------------------
   OpenGATE Collaboration

   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See GATE/LICENSE.txt for further details
----------------------*/

/*!
  \file GateToGPUImageSPECTMessenger.hh

  \brief Class GateToGPUImageSPECTMessenger
*/

#ifndef GATETOGPUIMAGESPECTMESSENGER_HH
#define GATETOGPUIMAGESPECTMESSENGER_HH

#include "GateOutputModuleMessenger.hh"

class GateToGPUImageSPECT;

class GateToGPUImageSPECTMessenger : public GateOutputModuleMessenger
{
	public:
		GateToGPUImageSPECTMessenger( GateToGPUImageSPECT* );
		~GateToGPUImageSPECTMessenger();

		void SetNewValue( G4UIcommand*, G4String );

	private:
		GateToGPUImageSPECT* m_gateToGPUImageSPECT;

		G4UIcmdWithAString        *setFileNameCmd;
        G4UIcmdWithAString        *attachToCmd;
        G4UIcmdWithAnInteger      *bufferParticleEntryCmd;
        G4UIcmdWithAnInteger      *cudaDeviceCmd;
				G4UIcmdWithAnInteger      *cpuNumberCmd;
				G4UIcmdWithABool          *cpuFlagCmd;
        G4UIcmdWithABool          *rootHitCmd;
        G4UIcmdWithABool          *rootSingleCmd;
        G4UIcmdWithABool          *rootSourceCmd;
				G4UIcmdWithABool          *rootExitCollimatorSourceCmd;
				G4UIcmdWithABool          *timeCmd;
        G4UIcmdWithAnInteger      *nzPixelCmd;
        G4UIcmdWithAnInteger      *nyPixelCmd;
				G4UIcmdWithADoubleAndUnit *zPixelSizeCmd;
				G4UIcmdWithADoubleAndUnit *yPixelSizeCmd;
        G4UIcmdWithADoubleAndUnit *septaCmd;
        G4UIcmdWithADoubleAndUnit *fyCmd;
        G4UIcmdWithADoubleAndUnit *fzCmd;
        G4UIcmdWithADoubleAndUnit *collimatorHeightCmd;
        G4UIcmdWithADoubleAndUnit *spaceBetweenCollimatorDetectorCmd;
				G4UIcmdWithADoubleAndUnit *rorCmd;
};
#endif
