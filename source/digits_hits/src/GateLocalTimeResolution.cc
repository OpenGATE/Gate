
#include "GateLocalTimeResolution.hh"

#include "GateLocalTimeResolutionMessenger.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"

#include "Randomize.hh"

#include "G4UnitsTable.hh"

#include "GateObjectStore.hh"
#include "GateConstants.hh"


GateLocalTimeResolution::GateLocalTimeResolution(GatePulseProcessorChain* itsChain,
				     const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName)
{
  m_messenger = new GateLocalTimeResolutionMessenger(this);
}

GateLocalTimeResolution::~GateLocalTimeResolution()
{
  delete m_messenger;
}

G4int GateLocalTimeResolution::ChooseVolume(G4String val)
{
  GateObjectStore* m_store = GateObjectStore::GetInstance();
  //AE
  DescribeMyself(2);
  if (m_store->FindCreator(val)!=0) {
    m_param.resol=-1;
    //m_delay = -1;
    m_table[val] = m_param;
    return 1;
  }
  else {
    G4cout << "Wrong Volume Name\n";
    return 0;
  }
}

void GateLocalTimeResolution::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{

    im=m_table.find(((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName());
    GatePulse* outputPulse = new GatePulse(*inputPulse);
    if(im != m_table.end()){
        // set the new time by a Gaussian shot of mean: old time, and of sigma: m_timeResolution/2.35
        G4double sigma =  (*im).second.resol/ GateConstants::fwhm_to_sigma;
        outputPulse->SetTime(G4RandGauss::shoot(inputPulse->GetTime(), sigma));
    }
    outputPulseList.push_back(outputPulse);

    if (nVerboseLevel>1)
          {
            G4cout << "Pulse real time: \n"
               << G4BestUnit(inputPulse->GetTime(),"Time") << Gateendl
               << "Pulse new time: \n"
               << G4BestUnit(outputPulse->GetTime(),"Time") << Gateendl
               << "Difference (real - new time): \n"
               << "volume name: \n"<<(*im).first<<Gateendl
               << G4BestUnit(inputPulse->GetTime() - outputPulse->GetTime(),"Time")
               << Gateendl << Gateendl ;

          }
}

void GateLocalTimeResolution::DescribeMyself(size_t indent)
{
  for (im=m_table.begin(); im!=m_table.end(); im++)
    G4cout << GateTools::Indent(indent) << "Time resolution of " << (*im).first << ":\n"
       << GateTools::Indent(indent+1) << (*im).second.resol <<Gateendl;
}
