/*----------------------
  03/2012
  ----------------------*/


#ifndef GateTriCoincidenceSorterMessenger_h
#define GateTriCoincidenceSorterMessenger_h 1

#include "GatePulseProcessorMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithADoubleAndUnit;
//class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class GateTriCoincidenceSorter;

class GateTriCoincidenceSorterMessenger: public GateClockDependentMessenger
{
public:
  GateTriCoincidenceSorterMessenger(GateTriCoincidenceSorter* itsProcessor);
  virtual ~GateTriCoincidenceSorterMessenger();

  inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

private:
   GateTriCoincidenceSorter*   m_itsProcessor;
   G4UIcmdWithAString*         m_SetInputSPLNameCmd;
   G4UIcmdWithADoubleAndUnit*  m_triCoincWindowCmd;
   G4UIcmdWithAnInteger*       m_SetWSPulseListSizeCmd;
};

#endif
