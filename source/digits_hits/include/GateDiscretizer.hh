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

#ifndef GateDiscretizer_h
#define GateDiscretizer_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateSinglesDigitizer.hh"


class GateDiscretizer : public GateVDigitizerModule
{
public:

  GateDiscretizer(GateSinglesDigitizer *digitizer, G4String name);
  ~GateDiscretizer();
  adder_policy_t   m_positionPolicy;

private:
  GateAdderMessenger *m_Messenger;

  GateDigi* m_outputDigi;
  GateDigiCollection*  m_OutputDigiCollection;
  GateSinglesDigitizer *m_digitizer;




};

#endif
