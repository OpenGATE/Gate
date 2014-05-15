/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! 
  \class  GateHounsfieldToMaterialsBuilder.hh
  \brief  
  \author david.sarrut@creatis.insa-lyon.fr
*/
 
#ifndef __GateHounsfieldToMaterialsBuilder__hh__
#define __GateHounsfieldToMaterialsBuilder__hh__

#include "GateHounsfieldToMaterialsBuilderMessenger.hh"
#include "GateMessageManager.hh"

class GateHounsfieldToMaterialsBuilder
{
public:
  GateHounsfieldToMaterialsBuilder();
  ~GateHounsfieldToMaterialsBuilder();
  
  void BuildAndWriteMaterials();
  void SetMaterialTable(G4String filename) { mMaterialTableFilename = filename; }
  void SetDensityTable(G4String filename) { mDensityTableFilename = filename; }
  void SetOutputMaterialDatabaseFilename(G4String filename) { mOutputMaterialDatabaseFilename = filename; }
  void SetOutputHUMaterialFilename(G4String filename) { mOutputHUMaterialFilename = filename; }
  void SetDensityTolerance(double tol) { mDensityTol = tol; }
  
protected:
  GateHounsfieldToMaterialsBuilderMessenger * pMessenger;
  G4String mMaterialTableFilename;
  G4String mDensityTableFilename;
  G4String mOutputMaterialDatabaseFilename;
  G4String mOutputHUMaterialFilename;
  double mDensityTol;

};

#endif
