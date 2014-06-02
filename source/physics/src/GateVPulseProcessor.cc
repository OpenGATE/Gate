/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVPulseProcessor.hh"

#include "GateTools.hh"
#include "GatePulseProcessorChain.hh"
#include "GateSingleDigiMaker.hh"
#include "GateDigitizer.hh"

// Constructs a new pulse-processor attached to a GateDigitizer
GateVPulseProcessor::GateVPulseProcessor(GatePulseProcessorChain* itsChain,
      	      	      	   const G4String& itsName) 
    : GateClockDependent(itsName),
      m_chain(itsChain)
{
  GateDigitizer* digitizer = GateDigitizer::GetInstance();

  digitizer->InsertDigiMakerModule( new GateSingleDigiMaker(digitizer, itsName,false) );
}  


GatePulseList* GateVPulseProcessor::ProcessPulseList(const GatePulseList* inputPulseList)
{
  if (!inputPulseList)
    return 0;

  size_t n_pulses = inputPulseList->size();
  if (nVerboseLevel==1)
      	G4cout << "[" << GetObjectName() << "::ProcessPulseList]: processing input list with " << n_pulses << " entries\n";
  if (!n_pulses)
    return 0;

  GatePulseList* outputPulseList = new GatePulseList(GetObjectName());

  GatePulseConstIterator iter;
  for (iter = inputPulseList->begin() ; iter != inputPulseList->end() ; ++iter)
      	ProcessOnePulse( *iter, *outputPulseList);
  
  if (nVerboseLevel==1) {
      G4cout << "[" << GetObjectName() << "::ProcessPulseList]: returning output pulse-list with " << outputPulseList->size() << " entries\n";
      for (iter = outputPulseList->begin() ; iter != outputPulseList->end() ; ++iter)
      	G4cout << **iter << G4endl;
      G4cout << G4endl;
  }

  return outputPulseList;
}



// Method overloading GateClockDependent::Describe()
// Print-out a description of the component
// Calls the pure virtual method DecribeMyself()
void GateVPulseProcessor::Describe(size_t indent) 
{
  GateClockDependent::Describe(indent);
  G4cout << GateTools::Indent(indent) << "Attached to:        '" << GetChain()->GetObjectName() << "'" << G4endl;
  G4cout << GateTools::Indent(indent) << "Output:             '" << GetObjectName() << "'" << G4endl;
  DescribeMyself(indent);
}
     
