/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

// OK GND 2022
/*!
  \class  GateDiscretizer (by marc.granado@universite-paris-saclay.fr)
  \brief   for discretizing the position within a monolithic crystal

    - For each volume the local position of the interactions within the crystal are  discretized.
	the X,Y,Z vector is translated into the IDs of virtual divisions within the crystal
	ocuppying the levels of SubmoduleID, CrystalID and LayerID.

*/


#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"


GateDiscretizer::GateDiscretizer(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer, digitizer->GetSD()),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
{

}


GateDiscretizer::GateDiscretizer()
{
}
