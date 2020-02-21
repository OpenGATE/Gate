
#include "GateLocalTimeDelay.hh"

#include "GateLocalTimeDelayMessenger.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"

#include "Randomize.hh"

#include "G4UnitsTable.hh"

#include "GateObjectStore.hh"
#include "GateConstants.hh"


GateLocalTimeDelay::GateLocalTimeDelay(GatePulseProcessorChain* itsChain,
				     const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName)
{
  m_messenger = new GateLocalTimeDelayMessenger(this);
}

GateLocalTimeDelay::~GateLocalTimeDelay()
{
  delete m_messenger;
}

G4int GateLocalTimeDelay::ChooseVolume(G4String val)
{
  GateObjectStore* m_store = GateObjectStore::GetInstance();
  //AE
  DescribeMyself(2);
  if (m_store->FindCreator(val)!=0) {
    m_param.delay=-1;
    //m_delay = -1;
    m_table[val] = m_param;
    return 1;
  }
  else {
    G4cout << "Wrong Volume Name\n";
    return 0;
  }
}

void GateLocalTimeDelay::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{

    im=m_table.find(((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName());
    GatePulse* outputPulse = new GatePulse(*inputPulse);
    if(im != m_table.end()){
        outputPulse->SetTime(inputPulse->GetTime()+(*im).second.delay);
        G4cout<<"Time delay applied "<<(*im).second.delay<<G4endl;
    }
    outputPulseList.push_back(outputPulse);
}

void GateLocalTimeDelay::DescribeMyself(size_t indent)
{
  for (im=m_table.begin(); im!=m_table.end(); im++)
    G4cout << GateTools::Indent(indent) << "Time delay of " << (*im).first << ":\n"
       << GateTools::Indent(indent+1) << (*im).second.delay <<Gateendl;
}
