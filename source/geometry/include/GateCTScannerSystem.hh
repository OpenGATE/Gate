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
  \file GateCTScannerSystem.hh

  \brief Class GateCTScannerSystem
  \author Didier Benoit <benoit@cppm.in2p3.fr>
  \author Franca Cassol Brunner <cassol@cppm.in2p3.fr>
*/

#ifndef GATECTSCANNERSYSTEM_HH
#define GATECTSCANNERSYSTEM_HH

#include "GateVSystem.hh"

class GateToImageCT;
class GateClockDependentMessenger;

class GateCTScannerSystem : public GateVSystem
{
	public:
		//! Constructor
		GateCTScannerSystem( const G4String& );
		//! Destructor
		virtual ~GateCTScannerSystem();
		
	private:
		GateToImageCT* m_gateToImageCT;
		GateClockDependentMessenger* m_messenger;
		
};

#endif
