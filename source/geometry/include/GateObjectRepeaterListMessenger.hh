/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateObjectRepeaterListMessenger_h
#define GateObjectRepeaterListMessenger_h 1

#include "globals.hh"
#include "GateListMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;
class G4UIcmdWithABool;
class G4UIcmdWithADouble;

#include "GateObjectRepeaterList.hh"

class GateVVolume;


class GateObjectRepeaterListMessenger: public GateListMessenger
{
  public:
    GateObjectRepeaterListMessenger(GateObjectRepeaterList* itsRepeaterList);
   ~GateObjectRepeaterListMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);

  private:
    virtual const G4String& DumpMap();
    virtual void ListChoices();
    virtual void DoInsertion(const G4String& typeName);
    virtual inline GateObjectRepeaterList* GetRepeaterList()
      { return (GateObjectRepeaterList*) GetListManager();}
    virtual GateVVolume* GetCreator()
      { return GetRepeaterList()->GetCreator(); }


  private:
  
};

#endif

