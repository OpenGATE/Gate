
#include "GateLocalMultipleRejection.hh"

#include "GateLocalMultipleRejectionMessenger.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"

#include "Randomize.hh"

#include "G4UnitsTable.hh"

#include "GateObjectStore.hh"
#include "GateConstants.hh"


GateLocalMultipleRejection::GateLocalMultipleRejection(GatePulseProcessorChain* itsChain,
				     const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName)
{
  m_messenger = new GateLocalMultipleRejectionMessenger(this);
}

GateLocalMultipleRejection::~GateLocalMultipleRejection()
{
  delete m_messenger;
}

G4int GateLocalMultipleRejection::ChooseVolume(G4String val)
{
  GateObjectStore* m_store = GateObjectStore::GetInstance();
  if (m_store->FindCreator(val)!=nullptr) {
    m_param.multipleDef=kvolumeName;
    m_param.rejectionAllPolicy=true;

    m_table[val] = m_param;
    return 1;
  }
  else {
    G4cout << "Wrong Volume Name\n";
    return 0;
  }
}

void GateLocalMultipleRejection::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{

    im=m_table.find(((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName());
    GatePulse* outputPulse = new GatePulse(*inputPulse);
    if(im != m_table.end()){

       currentVolumeName=(outputPulse->GetVolumeID().GetBottomCreator())->GetObjectName();
       if((*im).second.multipleDef==kvolumeID){
         currentNumber=outputPulse->GetVolumeID().GetBottomVolume()->GetCopyNo();
         currentVolumeName=currentVolumeName+std::to_string(currentNumber);
       }

       if( multiplesRejPol.find(currentVolumeName)== multiplesRejPol.end()){
            multiplesRejPol.insert(std::pair<G4String,G4bool> (currentVolumeName,(*im).second.rejectionAllPolicy));
       }

         //multiplesIndex.
       if(multiplesIndex.find(currentVolumeName)!=multiplesIndex.end()){;
            multiplesIndex[currentVolumeName].push_back(outputPulseList.size());
       }
       else{
           multiplesIndex.insert(std::pair<G4String,std::vector<int>> (currentVolumeName,{(int)(outputPulseList.size())}));
       }


    }
    outputPulseList.push_back(outputPulse);
}



GatePulseList* GateLocalMultipleRejection::ProcessPulseList(const GatePulseList* inputPulseList)
{
  if (!inputPulseList)
    return nullptr;

  size_t n_pulses = inputPulseList->size();
  if (nVerboseLevel==1)
        G4cout << "[" << GetObjectName() << "::ProcessPulseList]: processing input list with " << n_pulses << " entries\n";
  if (!n_pulses)
    return nullptr;

  GatePulseList* outputPulseList = new GatePulseList(GetObjectName());

  multiplesIndex.erase(multiplesIndex.begin(), multiplesIndex.end());
  GatePulseConstIterator iter;
  for (iter = inputPulseList->begin() ; iter != inputPulseList->end() ; ++iter)
      ProcessOnePulse( *iter, *outputPulseList);

  if( outputPulseList->size()>1){
  //ApplyRejecyion(*outputPulseList);
      G4bool flagDeleteAll=false;
      std::vector<int> posErase;
      for (auto const& x : multiplesIndex){
                 if(x.second.size()>1){
                     //multiples
                     //If one have Reject all policy  I need to delete the whole event otherwise just the pulses with the index of that volume
                     //Here indexes are saved to delete pulses unless reject all policy is  found
                      if(multiplesRejPol.find(x.first)!=multiplesRejPol.end()){
                         if(multiplesRejPol[x.first]==1){
                             flagDeleteAll=true;
                             break;
                         }
                      }
                      //Only delete those singles in the volume
                      for(unsigned int in=0; in<x.second.size(); in++){
                            posErase.push_back(x.second.at(in));
                      }


                 }
      }
      if(flagDeleteAll==true){
          while (outputPulseList->size())
         {
             delete outputPulseList->back();
             outputPulseList->erase(outputPulseList->end()-1);
           }
      }
      else if (flagDeleteAll==false && posErase.size()>1){
           std::sort (posErase.begin(), posErase.end());
          for(unsigned int i=0; i<posErase.size(); i++){
              // G4cout<<" position to delte="<<posErase.at(i)<<G4endl;
              delete outputPulseList->at(posErase.at(i)-i);
              outputPulseList->erase(outputPulseList->begin()+posErase.at(i)-i);
          }
      }
  }



  if (nVerboseLevel==1) {
      G4cout << "[" << GetObjectName() << "::ProcessPulseList]: returning output pulse-list with " << outputPulseList->size() << " entries\n";
      for (iter = outputPulseList->begin() ; iter != outputPulseList->end() ; ++iter)
        G4cout << **iter << Gateendl;
      G4cout << Gateendl;
  }

  return outputPulseList;
}


void GateLocalMultipleRejection::DescribeMyself(size_t indent)
{
  for (im=m_table.begin(); im!=m_table.end(); im++)
    G4cout << GateTools::Indent(indent) << "Multiple rejection " << (*im).first << ":\n"
       << GateTools::Indent(indent+1) << (*im).second.multipleDef <<
         GateTools::Indent(indent+1) << (*im).second.rejectionAllPolicy<<Gateendl;



}
