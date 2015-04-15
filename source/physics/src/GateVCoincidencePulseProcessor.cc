/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVCoincidencePulseProcessor.hh"

#include "GateTools.hh"
#include "GateCoincidencePulseProcessorChain.hh"
#include "GateCoincidenceDigiMaker.hh"
#include "GateDigitizer.hh"
//------------------------------------------------------------------------------------------------------
// Constructs a new pulse-processor attached to a GateDigitizer
GateVCoincidencePulseProcessor::GateVCoincidencePulseProcessor(GateCoincidencePulseProcessorChain* itsChain,
      	      	      	   const G4String& itsName) 
    : GateClockDependent(itsName),
/*      m_isTriCoincProc(0), //mhadi_add*/
      m_chain(itsChain)
{
  GateDigitizer* digitizer = GateDigitizer::GetInstance();

  G4cout << " in GateVCoincidencePulseProcessor call new GateCoincidenceDigiMaker" << Gateendl;
  digitizer->InsertDigiMakerModule( new GateCoincidenceDigiMaker(digitizer, itsName,false) );
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Method overloading GateClockDependent::Describe()
// Print-out a description of the component
// Calls the pure virtual method DecribeMyself()
void GateVCoincidencePulseProcessor::Describe(size_t indent) 
{
  GateClockDependent::Describe(indent);
  G4cout << GateTools::Indent(indent) << "Attached to:        '" << GetChain()->GetObjectName() << "'" << Gateendl;
  G4cout << GateTools::Indent(indent) << "Output:             '" << GetObjectName() << "'" << Gateendl;
  DescribeMyself(indent);
}
//------------------------------------------------------------------------------------------------------    
