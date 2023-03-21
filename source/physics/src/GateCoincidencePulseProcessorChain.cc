/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/
// GND ClassToRemove

#include "GateCoincidencePulseProcessorChain.hh"
#include "GateCoincidencePulseProcessorChainMessenger.hh"

#include "Randomize.hh"
#include "G4UnitsTable.hh"

#include "GateDigitizer.hh"
#include "GateVCoincidencePulseProcessor.hh"
#include "GateTools.hh"
//#include "GateHitConvertor.hh"
//#include "GateCoincidenceDigiMaker.hh"


//------------------------------------------------------------------------------------------------------
GateCoincidencePulseProcessorChain::GateCoincidencePulseProcessorChain( GateDigitizer* itsDigitizer,
    			                          const G4String& itsOutputName)
  : GateModuleListManager(itsDigitizer,itsDigitizer->GetObjectName() + "/" + itsOutputName,"pulse-processor"),
    m_system(0 /*itsDigitizer->GetSystem() */),//mhadi_modif
    m_outputName(itsOutputName),
    m_inputNames(),
    m_noPriority(true)
{
  
  m_messenger = new GateCoincidencePulseProcessorChainMessenger(this);
  //name of your chain
   //G4cout<<"GateCoincidencePulseProcessor m_outputName="<<m_outputName<<G4endl;

  G4cout << " in GateCoincidencePulseProcessorChain call new GateCoincidenceDigiMaker "  << Gateendl;
  //itsDigitizer->InsertDigiMakerModule( new GateCoincidenceDigiMaker(itsDigitizer, itsOutputName,true) );
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
GateCoincidencePulseProcessorChain::~GateCoincidencePulseProcessorChain()
{  
    delete m_messenger;
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
void GateCoincidencePulseProcessorChain::InsertProcessor(GateVCoincidencePulseProcessor* newChildProcessor)
{
  theListOfNamedObject.push_back(newChildProcessor);
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
void GateCoincidencePulseProcessorChain::Describe(size_t indent)
{
  GateModuleListManager::Describe();
  //G4cout << GateTools::Indent(indent) << "Input:              '" << m_inputNames << "'\n";
  G4cout << GateTools::Indent(indent) << "Output:             '" << m_outputName << "'\n";
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
void GateCoincidencePulseProcessorChain::DescribeProcessors(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Nb of modules:       " << theListOfNamedObject.size() << Gateendl;
  for (size_t i=0; i<theListOfNamedObject.size(); i++)
      GetProcessor(i)->Describe(indent+1);
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
void GateCoincidencePulseProcessorChain::ListElements()
{
  DescribeProcessors(0);
}
const std::vector<GateCoincidencePulse*> GateCoincidencePulseProcessorChain::MakeInputList() const
{
   std::vector<GateCoincidencePulse*> ans;
   for (std::vector<G4String>::const_iterator itName = m_inputNames.begin() ; itName != m_inputNames.end() ; ++itName){
     std::vector<GateCoincidencePulse*> pulseList 
        = GateDigitizer::GetInstance()->FindCoincidencePulse( *itName );
     for (std::vector<GateCoincidencePulse*>::const_iterator it = pulseList.begin() ; it != pulseList.end() ; ++it){
	GateCoincidencePulse* pulse = *it;
	if (pulse->empty()) continue;
	G4double time = pulse->GetTime();
	bool last=true;
	// should use a map to imrove sorted insertion, but typically lists are short,
	// so this may not be a problem
	for (std::vector<GateCoincidencePulse*>::iterator it2 = ans.begin() ; it2 != ans.end() ; ++it2){
	  if ( (*it2)->GetTime()>time) {ans.insert(it2,pulse) ; last=false; break;}
	  if ( m_noPriority && ((*it2)->GetTime()==time)) {

//S.Jan 15/02/2006
	    //G4double p = RandFlat::shoot();
	    G4double p = G4UniformRand();
	    if (p<0.5) {ans.insert(it2,pulse) ; last=false; break;}
	  }
        }
        if (last) ans.push_back(pulse);
     }
   }
   return ans;
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
void GateCoincidencePulseProcessorChain::ProcessCoincidencePulses()
{
  if (m_inputNames.empty()) m_inputNames.push_back("Coincidences");
  std::vector<GateCoincidencePulse*> pulseList = MakeInputList();

  //mhadi_add[
  for (size_t processorID = 0 ; processorID < GetProcessorNumber(); processorID++) 
  {
    GateVCoincidencePulseProcessor* processor =  GetProcessor(processorID);
    if (processor->IsEnabled() && processor->IsTriCoincProcessor())
        processor->CollectSingles();
    
  }
  //mhadi_add]
        
  if (pulseList.empty())
    return;


  // Sequentially launch all pulse processors
  G4int i=0;
  for (std::vector<GateCoincidencePulse*>::iterator it = pulseList.begin() ; it != pulseList.end() ; ++it,++i){
     GateCoincidencePulse* pulse = *it;
     if (pulse->empty()) continue;
     for (size_t processorID = 0 ; processorID < GetProcessorNumber(); processorID++) {
       GateVCoincidencePulseProcessor* processor =  GetProcessor(processorID);
       if (processor->IsEnabled()) {
	 pulse = processor->ProcessPulse(pulse,i);
	 if (pulse){
	   //G4cout<<"processorName="<<processor->GetObjectName()<<G4endl;
      	   pulse->SetName(processor->GetObjectName());
      	   GateDigitizer::GetInstance()->StoreCoincidencePulse(pulse);
	 } else break;
       }
     }
      //G4cout<<"CoincChain m_outputName="<<m_outputName<<G4endl;
     if (pulse) GateDigitizer::GetInstance()->StoreCoincidencePulseAlias(m_outputName,pulse);
   }

  return;
}
//------------------------------------------------------------------------------------------------------

//mhadi_add[
//------------------------------------------------------------------------------------------------------
GateVSystem* GateCoincidencePulseProcessorChain::FindSystem(G4String& inputName)
{
   GateDigitizer* digitizer = GateDigitizer::GetInstance();
   std::vector<GateCoincidenceSorterOld*> CoincidenceSorterList = digitizer->GetCoinSorterList();

   /*G4int index = -1;

   for(size_t i=0; i<CoincidenceSorterList.size(); i++)
   {
      G4String coincSorterChainName = CoincidenceSorterList[i]->GetOutputName();
      if(inputName.compare(coincSorterChainName) == 0)
      {
         index = i;
         break;
      }
   }

   GateVSystem* system = 0;
   if(index != -1)
      system = CoincidenceSorterList[index]->GetSystem();
   
   return system;*/
   
   //faster!!
   for (std::vector<GateCoincidenceSorterOld*>::iterator itr=CoincidenceSorterList.begin(); itr!=CoincidenceSorterList.end(); itr++)
     if (inputName.compare((*itr)->GetOutputName()) == 0)
       return (*itr)->GetSystem();
   
   return 0;
}
//mhadi_add]
