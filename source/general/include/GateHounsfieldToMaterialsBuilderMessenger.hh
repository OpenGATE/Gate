/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! 
  \class  GateHounsfieldToMaterialsBuilderMessenger.hh
  \brief  
  \author david.sarrut@creatis.insa-lyon.fr
*/
 
#ifndef __GateHounsfieldToMaterialsBuilderMessenger__hh__
#define __GateHounsfieldToMaterialsBuilderMessenger__hh__

#include "GateMessenger.hh"

class GateHounsfieldToMaterialsBuilder;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithADoubleAndUnit;

class GateHounsfieldToMaterialsBuilderMessenger: public G4UImessenger
{
public:
  GateHounsfieldToMaterialsBuilderMessenger(GateHounsfieldToMaterialsBuilder * m);
  virtual ~GateHounsfieldToMaterialsBuilderMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateHounsfieldToMaterialsBuilder * mBuilder;
  G4UIcmdWithoutParameter* pGenerateCmd;
  G4UIcmdWithAString * pSetMaterialTable;
  G4UIcmdWithAString * pSetDensityTable;
  G4UIcmdWithAString * pSetOutputMaterialDatabaseFilename;
  G4UIcmdWithAString * pSetOutputHUMaterialFilename;
  G4UIcmdWithADoubleAndUnit * pSetDensityTolerance;
};

#endif
