/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/



#ifndef GateCoincidenceDigitizerMessenger_h
#define GateCoincidenceDigitizerMessenger_h 1

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

#include "GateCoincidenceDigitizer.hh"


class G4VDigitizerModule;

/*! \class  GateCoincidenceDigitizerMessenger
    \brief  This messenger manages a chain of pulse-processor modules

    - GateCoincidenceDigitizerMessenger - by Daniel.Strul@iphe.unil.ch
*/
//    Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr, for the multi-system approach.

class GateCoincidenceDigitizerMessenger: public GateListMessenger
{
  public:
    //! Constructor: the argument is the chain of pulse-processor modules
    GateCoincidenceDigitizerMessenger(GateCoincidenceDigitizer* itsDigitizer);

    ~GateCoincidenceDigitizerMessenger();

    //! Standard messenger command interpreter
    void SetNewValue(G4UIcommand*, G4String);

    G4bool CheckNameConflict(const G4String& name);

  private:
    //! Dumps the list of modules that the user can insert into the chain
    virtual const G4String& DumpMap();

    //! Inserts a new module into the digitizer
    virtual void DoInsertion(const G4String& typeName);



    //! Returns the pulse-processor chain managed by this messenger
   // virtual GateCoincidenceDigitizer* GetDigitizer()
   //   {   	return (GateCoincidenceDigitizer*) GetListManager(); }



  private:

    G4UIcmdWithAString*         AddInputNameCmd;        //!< The UI command "add input name"
    G4UIcmdWithABool*           usePriorityCmd;    //!< The UI command "usePriority"

    GateCoincidenceDigitizer* m_CoinDigitizer;
};

#endif
