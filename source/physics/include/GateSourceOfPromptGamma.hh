/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateSourceOfPromptGamma

  Gate Source: manage a source of Prompt Gamma.  The source is a 3D
  spatial discrete distribution. Primary particles are photon created
  with position from the 3D spatial distribution. Energy is given by
  1D spectrum distribution in each voxel. Directions are isotropic.
*/

#ifndef GATESOURCEPROMPTGAMMAEMISSION_HH
#define GATESOURCEPROMPTGAMMAEMISSION_HH

#include "G4UnitsTable.hh"
#include "GateSourceOfPromptGammaData.hh"
#include "GateVSource.hh"
#include "GateSourceOfPromptGammaMessenger.hh"
#include <iostream>
#include <fstream>
#include "GateImageOfHistograms.hh"

class GateSourceOfPromptGammaMessenger;

//------------------------------------------------------------------------
class GateSourceOfPromptGamma : public GateVSource
{
public:
  GateSourceOfPromptGamma(G4String name);
  ~GateSourceOfPromptGamma();

  G4int GeneratePrimaries(G4Event* event);
  void GenerateVertex(G4Event* );
  void SetFilename(G4String filename);
  void SetTof(G4bool newflag);
  
protected:
  GateSourceOfPromptGammaMessenger * pMessenger;
  bool mIsInitializedFlag;
  bool mIsInitializedNumberOfPrimariesFlag;
  GateSourceOfPromptGammaData * mData;
  G4String mFilename;
  double mEnergy; // because particle_energy is private (FIXME will be changed)
  double mTime; /** Modif Oreste **/
  
  void Initialize();
  double ng;
  void InitializeNumberOfPrimaries();
}; // end class
//------------------------------------------------------------------------

#endif /* end #define GATESOURCEPROMPTGAMMAEMISSION */
