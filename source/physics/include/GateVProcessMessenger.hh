/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateVProcessMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GATEVPROCESSMESSENGER_HH
#define GATEVPROCESSMESSENGER_HH

#include "globals.hh"

#include "G4UImessenger.hh"
#include "GateUIcmdWith2String.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "GateUIcmdWithADoubleAnd3String.hh"
#include "GateUIcmdWithADoubleAnd2String.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "GateUIcmdWithAStringAndAnInteger.hh"
#include "GateUIcmdWithAStringAndADouble.hh"
#include "G4UIcmdWithABool.hh"

class GateVProcess;
class G4UIdirectory;

class GateVProcessMessenger:public G4UImessenger
{
public:
  GateVProcessMessenger(GateVProcess *pb);
  virtual ~GateVProcessMessenger();

  void BuildERangeCommands(G4String base);// Not yet used!
  void SetERangeNewValue(G4UIcommand*, G4String);// Not yet used!

  void BuildModelsCommands(G4String base);
  void BuildEnergyRangeModelsCommands(G4String base);
  void SetModelsNewValue(G4UIcommand*, G4String);
  void SetEnergyRangeModelsNewValue(G4UIcommand*, G4String);

  void BuildDataSetCommands(G4String base);
  void SetDataSetNewValue(G4UIcommand*, G4String);

  void BuildWrapperCommands(G4String base);
  void SetWrapperNewValue(G4UIcommand*, G4String);

  virtual void BuildCommands(G4String base)=0;
  virtual void SetNewValue(G4UIcommand*, G4String)=0;

  G4double ScaleValue(G4double value,G4String unit);

protected:
  GateVProcess *pProcess;

  GateUIcmdWithAStringAndAnInteger * pSetSplit;
  GateUIcmdWithAStringAndAnInteger * pSetRussianR;
  GateUIcmdWith2String    * pAddFilter;
  GateUIcmdWithAStringAndADouble * pSetCSE;
  G4UIcmdWithABool * pFilteredParticleState;

  GateUIcmdWithADoubleAnd3String * pSetEmin;
  GateUIcmdWithADoubleAnd3String * pSetEmax;
  G4UIcmdWithoutParameter * pClearERange;

  GateUIcmdWith2String *pAddDataSet;
  GateUIcmdWith2String *pRemoveDataSet;
  G4UIcmdWithAString * pListDataSet;

  GateUIcmdWith2String * pAddModel;
  GateUIcmdWith2String * pRemoveModel;
  G4UIcmdWithAString * pListModel;

  std::vector<GateUIcmdWithADoubleAnd3String *> plModelSetEmin;
  std::vector<GateUIcmdWithADoubleAnd3String *> plModelSetEmax;
  std::vector<G4UIcmdWithAString *> plModelClearERange;

  G4String mPrefix;

  G4UIdirectory*  pProcessDir;
};

#endif /* end #define GATEVPROCESSMESSENGER_HH */
