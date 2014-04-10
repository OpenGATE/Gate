/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*!
  \class  GateSourcePromptGammaEmission

  Gate Source: manage a source of Prompt Gamma.  The source is a 3D
  spatial discrete distribution. Primary particles are photon created
  with position from the 3D spatial distribution. Energy is given by
  1D spectrum distribution in each voxel. Directions are isotropic.
*/

#ifndef GATESOURCEPROMPTGAMMAEMISSION_HH
#define GATESOURCEPROMPTGAMMAEMISSION_HH

#include "G4UnitsTable.hh"
#include "GatePromptGammaSpatialEmissionDistribution.hh"
#include "GateVSource.hh"
#include "GateSourcePromptGammaEmissionMessenger.hh"

//------------------------------------------------------------------------
class GateSourcePromptGammaEmission : public GateVSource
{
public:
  GateSourcePromptGammaEmission(G4String name);
  ~GateSourcePromptGammaEmission();

  G4int GeneratePrimaries(G4Event* event);
  void GenerateVertex(G4Event* );
  void SetFilename(G4String filename);

protected:
  GateSourcePromptGammaEmissionMessenger * pMessenger;
  bool mIsInitializedFlag;
  GatePromptGammaSpatialEmissionDistribution * mDistrib;
  G4String mFilename;

  void Initialize();
}; // end class
//------------------------------------------------------------------------

#endif /* end #define GATESOURCEPROMPTGAMMAEMISSION */
