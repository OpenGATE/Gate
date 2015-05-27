/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GatePulseProcessorChain.hh"
#include "GatePulseProcessorChainMessenger.hh"

#include "G4UnitsTable.hh"

#include "GateDigitizer.hh"
#include "GateVPulseProcessor.hh"
#include "GateTools.hh"
#include "GateHitConvertor.hh"
#include "GateSingleDigiMaker.hh"



GatePulseProcessorChain::GatePulseProcessorChain( GateDigitizer* itsDigitizer,
    			                          const G4String& itsOutputName)
  : GateModuleListManager(itsDigitizer,itsDigitizer->GetObjectName() + "/" + itsOutputName,"pulse-processor"),
    m_system( itsDigitizer->GetSystem() ),
    m_outputName(itsOutputName),
    m_inputName(GateHitConvertor::GetOutputAlias())
{
//  G4cout << " DEBUT Constructor GatePulseProcessorChain \n";
  m_messenger = new GatePulseProcessorChainMessenger(this);

//  G4cout << " in GatePulseProcessorChain call GateSingleDigiMaker\n";
  itsDigitizer->InsertDigiMakerModule( new GateSingleDigiMaker(itsDigitizer, itsOutputName,true) );
  
//  G4cout << " FIN Constructor GatePulseProcessorChain \n";
}

GatePulseProcessorChain::~GatePulseProcessorChain()
{  
    delete m_messenger;
}

void GatePulseProcessorChain::InsertProcessor(GateVPulseProcessor* newChildProcessor)
{
	theListOfNamedObject.push_back(newChildProcessor);
}


void GatePulseProcessorChain::Describe(size_t indent)
{
  GateModuleListManager::Describe();
  G4cout << GateTools::Indent(indent) << "Input:              '" << m_inputName << "'\n";
  G4cout << GateTools::Indent(indent) << "Output:             '" << m_outputName << "'\n";
}

void GatePulseProcessorChain::DescribeProcessors(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Nb of modules:       " << size() << Gateendl;
  for (iterator it = begin(); it != end(); ++it)
      ((GateVPulseProcessor*)(*it))->Describe(indent+1);
}

void GatePulseProcessorChain::ListElements()
{
  DescribeProcessors(0);
}

GatePulseList* GatePulseProcessorChain::ProcessPulseList()
{
  GatePulseList* pulseList = GateDigitizer::GetInstance()->FindPulseList( m_inputName );

  if (!pulseList)
    return 0;

  // Empty pulse list: no need to process
  if (pulseList->empty())
    return 0;

  // Sequentially launch all pulse processors
  for (iterator it = begin(); it != end(); ++it)
    if (((GateVPulseProcessor*)(*it))->IsEnabled()) {
      pulseList = ((GateVPulseProcessor*)(*it))->ProcessPulseList(pulseList);
      if (pulseList) GateDigitizer::GetInstance()->StorePulseList(pulseList);
      else break;
    }

  if (pulseList)  GateDigitizer::GetInstance()->StorePulseListAlias(m_outputName,pulseList);
  return pulseList;
}


