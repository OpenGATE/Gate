/*----------------------
  03/2012
  ----------------------*/


#include "GateTriCoincidenceSorter.hh"
#include "GateTriCoincidenceSorterMessenger.hh"
#include "GateDigi.hh"

#include "TFile.h"

GateTriCoincidenceSorter::GateTriCoincidenceSorter(GateCoincidencePulseProcessorChain* itsChain,
			   const G4String& itsName)
  : GateVCoincidencePulseProcessor(itsChain,itsName),
  m_triCoincWindow(0),
  m_waitingSingles(0),
  m_sTree(0),
  m_waitingSinglesSize(50)
{
   m_digitizer = GateDigitizer::GetInstance();
   m_messenger = new GateTriCoincidenceSorterMessenger(this);
   m_sBuffer.Clear();
   m_sTreeName = SetSinglesTreeName(itsName);
   m_triCoincID = 0;
}


GateTriCoincidenceSorter::~GateTriCoincidenceSorter()
{
  delete m_messenger;
  if(m_waitingSingles) delete m_waitingSingles; //mhadi_note We have to empty this list.
  delete m_sTree;
}

//=============================================================================
//=============================================================================

GateCoincidencePulse* GateTriCoincidenceSorter::ProcessPulse(GateCoincidencePulse* inputPulse,G4int )
{
   GateCoincidencePulse* ans = 0;/*new GateCoincidencePulse(*inputPulse)*/;

   // 1) If there is no singles or no time coincidence we leave with no outputs.
   if(m_waitingSingles->size() == 0)
      return ans;

   // 2) Getting the coincident singles from the singles pulse list
   G4double cTime = (inputPulse->at(0)->GetTime() + inputPulse->at(1)->GetTime())/2;

   GatePulseList cSPulseList("");
   std::vector<GatePulseIterator> toDel;
   for (GatePulseIterator itr = m_waitingSingles->begin() ; itr != m_waitingSingles->end() ; ++itr)
   {

      if((cTime - (*itr)->GetTime()) > m_triCoincWindow)
         toDel.push_back(itr);

      if(std::abs((*itr)->GetTime() - cTime) <= m_triCoincWindow)
      {
         GatePulse* cPulse = new GatePulse(*(*itr));
         cSPulseList.push_back(cPulse);
         toDel.push_back(itr);
      }
   }

   for (int i= (int)toDel.size()-1;i>=0;i--){
      delete (*toDel[i]);
      m_waitingSingles->erase( toDel[i] );
   }

   if(cSPulseList.size() == 0)
      return ans;

   // 3) Registring the coincident singles in m_sTree.
   RegisterTCSingles(cSPulseList);
   ans = new GateCoincidencePulse(*inputPulse);

   return ans;
}

//=============================================================================
//=============================================================================
void GateTriCoincidenceSorter::CollectSingles()
{
   // This method is for stocking singles in m_waitingSingles buffer, it is called each event.
   GatePulseList* sPulseList(FindSinglesPulseList(m_sPulseListName));

   if(sPulseList)
   {
   if(!m_waitingSingles)
   {
      // 1) Create m_waitingSingles and fill it by the sPulseList elements.
      m_waitingSingles =new GatePulseList(*sPulseList);
      return ;
   }
   else
   {
      // 2) Fill the m_waitingSingles buffer by sPulseList elements.
      if(m_waitingSingles->size() <= (size_t)m_waitingSinglesSize)
         for(GatePulseIterator itr=sPulseList->begin(); itr!=sPulseList->end(); itr++)
            m_waitingSingles->push_back(new GatePulse(*itr));
      else
      {
         // 3) If the size of the buffer exceeds 'm_waitingSinglesSize', we empty it,from the begin, until that its size becomes 'm_waitingSinglesSize'
         while(m_waitingSingles->size() > (size_t)m_waitingSinglesSize)
         {
            delete m_waitingSingles->at(0);
            m_waitingSingles->erase(m_waitingSingles->begin());
         }

         // 4) Refill the m_waitingSingles buffer by sPulseList elements.
         for(GatePulseIterator itr=sPulseList->begin(); itr!=sPulseList->end(); itr++)
            m_waitingSingles->push_back(new GatePulse(*itr));
      }
   }
   }

}

//=============================================================================
//=============================================================================
void GateTriCoincidenceSorter::RegisterTCSingles(GatePulseList& sPulseList)
{
   if(m_sTree == 0)
   {
      m_sTree  = new GateSingleTree(m_sTreeName);
      m_sTree->Init(m_sBuffer);
      // We add here a new branch to the singles tree to hold the the ID of the coincidence pulse,
      // all coincident singles with the same coincidence pulse have the same triCID.
      m_triCID = m_sTree->Branch("triCID",&m_triCoincID,"m_triCoincID/Int_t");
   }

   GatePulseConstIterator itr;
   for (itr = sPulseList.begin() ; itr != sPulseList.end() ; ++itr) {
      GateDigi* Digi = new GateDigi((*itr));
      m_sBuffer.Fill(Digi);
      m_sTree->Fill();
      delete Digi;

   }

   m_triCoincID++;

}

//=============================================================================
//=============================================================================
G4String GateTriCoincidenceSorter::SetSinglesTreeName(const G4String& name)
{
   G4String treeName = name;
   treeName = treeName.substr(10);

   G4String::size_type pos;
      pos = treeName.find_first_of('/');
      treeName = treeName.substr(0,pos);
      treeName += "CSingles";
   return treeName;
}

//=============================================================================
//=============================================================================
void GateTriCoincidenceSorter::DescribeMyself(size_t /*indent*/)
{
//   G4cout << GateTools::Indent(indent) << "Buffer: " << G4BestUnit(m_bufferSize,"Memory size")
//          << "Read @ "<< G4BestUnit(m_readFrequency,"Frequency")<< Gateendl;
}
