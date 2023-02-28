/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/
 //GND 2022 Class to Remove

#include "GatePulseProcessorChain.hh"
#include "GatePulseProcessorChainMessenger.hh"

#include "G4UnitsTable.hh"

#include "GateDigitizer.hh"
#include "GateVPulseProcessor.hh"
#include "GateTools.hh"
#include "GateHitConvertor.hh"
//#include "GateDigiMaker.hh"



GatePulseProcessorChain::GatePulseProcessorChain( GateDigitizer* itsDigitizer,
    			                          const G4String& itsOutputName)
  : GateModuleListManager(itsDigitizer,itsDigitizer->GetObjectName() + "/" + itsOutputName,"pulse-processor"),
    m_system( itsDigitizer->GetSystem() ),
    m_outputName(itsOutputName),
    m_inputName(GateHitConvertor::GetOutputAlias())
{
//  G4cout << " DEBUT Constructor GatePulseProcessorChain \n";
  m_messenger = new GatePulseProcessorChainMessenger(this);

//  G4cout << " in GatePulseProcessorChain call GateDigiMaker\n";
  //itsDigitizer->InsertDigiMakerModule( new GateDigiMaker(itsDigitizer, itsOutputName,true) );
  
//  G4cout << " FIN Constructor GatePulseProcessorChain \n";
}




GatePulseProcessorChain::~GatePulseProcessorChain()
{
  for (auto processor = theListOfNamedObject.begin(); processor != theListOfNamedObject.end(); ++processor)
  {
    GateMessage("Core", 5, "~GatePulseProcessorChain -- delete module: " << (*processor)->GetObjectName() << Gateendl );
    delete (*processor);
  }
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
  G4cout << GateTools::Indent(indent) << "Nb of modules:       " << theListOfNamedObject.size() << Gateendl;
  for (size_t i=0; i<theListOfNamedObject.size(); i++)
      GetProcessor(i)->Describe(indent+1);
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
  for (size_t processorID = 0 ; processorID < GetProcessorNumber(); processorID++) 
    if (GetProcessor(processorID)->IsEnabled()) {
      pulseList = GetProcessor(processorID)->ProcessPulseList(pulseList);
      if (pulseList) GateDigitizer::GetInstance()->StorePulseList(pulseList);
      else break;
    }

  if (pulseList)  GateDigitizer::GetInstance()->StorePulseListAlias(m_outputName,pulseList);
  return pulseList;
}


