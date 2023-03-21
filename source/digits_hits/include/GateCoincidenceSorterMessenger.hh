/*----------------------
   Copyright (C): OpenGATE Collaboration
   
This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateCoincidenceSorterMessenger_h
#define GateCoincidenceSorterMessenger_h 1

#include "GateClockDependentMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateCoincidenceSorter;

/*! \class  GateCoincidenceSorterMessenger
    \brief  Messenger used for commanding a  GateCoincidenceSorter
    
    - GateCoincidenceSorterMessenger - by Daniel.Strul@iphe.unil.ch
    
    - This messenger inherits from the abilities and responsabilities of the
      GateClockDependentMessenger base class: creation of a command directory,
      with various commands ('describe', 'verbose', 'enable', 'disable', 'setWindow')
      
    - In addition, it provides commands for settings the parameters of a cylindrical
      scanner's coincidence sorter: 'setWindow', 'minSectorDifference'
      
    \sa GateCoincidenceSorter
*/
//    Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr, for the multi-system approach.
//    Modified 01/2016 by Jared.STRYDHORST@cea.fr to add control of the presort buffer size
// 	  10/2022 Adapted to new digitizer by olga.kochebina@cea.fr

class GateCoincidenceSorterMessenger: public GateClockDependentMessenger
{
public:
    //! Constructor
    GateCoincidenceSorterMessenger(GateCoincidenceSorter* itsCoincidenceSorter);
    //! Destructor
    ~GateCoincidenceSorterMessenger();
    
    //! UI command interpreter
    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);
    
    
private:
    G4UIcmdWithADoubleAndUnit   *windowCmd;          //!< the UI command 'setWindow'
    G4UIcmdWithADoubleAndUnit   *offsetCmd;          //!< the UI command 'setOffset'
    G4UIcmdWithADoubleAndUnit   *windowJitterCmd;    //!< the UI command 'setWindowJitter'
    G4UIcmdWithADoubleAndUnit   *offsetJitterCmd;    //!< the UI command 'setOffsetJitter'
    G4UIcmdWithAnInteger        *minSectorDiffCmd;   //!< the UI command 'minSectorDifference'
    G4UIcmdWithAnInteger        *setDepthCmd;        //!< the UI command 'setDepth'
    G4UIcmdWithAnInteger        *setPresortBufferSizeCmd;  //!< the UI command 'setPresortBufferSize'
    G4UIcmdWithAString          *SetInputNameCmd;    //!< The UI command "set input name"
    G4UIcmdWithAString          *MultiplePolicyCmd;  //!< The UI command "MultiplesPolicy"
    G4UIcmdWithAString          *SetAcceptancePolicy4CCCmd;  //!< The UI command "MultiplesPolicy"
    G4UIcmdWithABool            *AllDigiOpenCoincGateCmd;  //!< The UI command "allowMultiples"
    G4UIcmdWithABool            *SetTriggerOnlyByAbsorberCmd;
    G4UIcmdWithABool            *SetEventIDCoincCmd;
    GateCoincidenceSorter* m_CoincidenceSorter;

};

#endif
