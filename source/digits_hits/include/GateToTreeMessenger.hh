/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


//
// Created by mdupont on 17/05/19.
//

#ifndef GATE_GATETOTREEMESSENGER_HH
#define GATE_GATETOTREEMESSENGER_HH

#include <unordered_map>
#include "GateOutputModuleMessenger.hh"

class GateToTree;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithoutParameter;


class GateToTreeMessenger : public GateOutputModuleMessenger
{
public:
  GateToTreeMessenger(GateToTree *m);

  virtual ~GateToTreeMessenger();

  void SetNewValue(G4UIcommand *icommand, G4String string) override;

private:

  G4UIcmdWithAString* m_addFileNameCmd;
  G4UIcmdWithoutParameter* m_enableCCoutputCmd;
  G4UIcmdWithoutParameter* m_disableCCoutputCmd;

  G4UIcmdWithoutParameter *m_enableHitsOutput;
  G4UIcmdWithoutParameter *m_disableHitsOutput;

  G4UIcmdWithoutParameter *m_enableOpticalDataOutput;
  G4UIcmdWithoutParameter *m_disableOpticalDataOutput;

  G4UIcmdWithAString* m_addHitsCollectionCmd;
  G4UIcmdWithAString* m_addOpticalCollectionCmd;
  G4UIcmdWithAString* m_addCollectionCmd;
  GateToTree *m_gateToTree;

  std::unordered_map<G4UIcmdWithoutParameter*, G4String> m_maphits_cmdParameter_toTreeParameter;
  std::unordered_map<G4UIcmdWithoutParameter*, G4String> m_mapoptical_cmdParameter_toTreeParameter;
  std::unordered_map<G4UIcmdWithoutParameter*, G4String> m_mapsingles_cmdParameter_toTreeParameter;
  std::unordered_map<G4UIcmdWithoutParameter*, G4String> m_mapscoincidences_cmdParameter_toTreeParameter;


};


#endif //GATE_GATETOTREEMESSENGER_HH
