/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSourceVoxelLinearTranslator.hh"
#include "GateSourceVoxelLinearTranslatorMessenger.hh"

GateSourceVoxelLinearTranslator::GateSourceVoxelLinearTranslator(GateVSourceVoxelReader* voxelReader) 
  : GateVSourceVoxelTranslator(voxelReader)
{
  m_name = G4String("linearTranslator");
  m_valueToActivityScale = 0.;
  m_messenger = new GateSourceVoxelLinearTranslatorMessenger(this);
}

GateSourceVoxelLinearTranslator::~GateSourceVoxelLinearTranslator() 
{
  delete m_messenger;
}

G4double GateSourceVoxelLinearTranslator::TranslateToActivity(G4double voxelValue)
{
  G4double activity = 0.;

  if (voxelValue > 0) {
    activity = voxelValue * m_valueToActivityScale;
  }

  return activity;
}
void GateSourceVoxelLinearTranslator::UpdateActivity(G4double, G4double, G4double){ ; }
