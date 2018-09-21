
#include "GateCC3DlocalSpblurring.hh"

#include "GateCC3DlocalSpblurringMessenger.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"

#include "Randomize.hh"

#include "G4UnitsTable.hh"

#include "GateObjectStore.hh"
#include "GateConstants.hh"


GateCC3DlocalSpblurring::GateCC3DlocalSpblurring(GatePulseProcessorChain* itsChain,
				     const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName)
{
  m_messenger = new GateCC3DlocalSpblurringMessenger(this);
}

GateCC3DlocalSpblurring::~GateCC3DlocalSpblurring()
{
  delete m_messenger;
}

G4int GateCC3DlocalSpblurring::ChooseVolume(G4String val)
{
  GateObjectStore* m_store = GateObjectStore::GetInstance();
  //AE
  DescribeMyself(2);
  if (m_store->FindCreator(val)!=0) {
      m_param.sigmaSpblurr.setX(-1);
      m_param.sigmaSpblurr.setY(-1);
      m_param.sigmaSpblurr.setZ(-1);

      m_table[val] = m_param;
      return 1;
  }
  else {
      G4cout << "Wrong Volume Name\n";
      return 0;
  }
}

void GateCC3DlocalSpblurring::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{

    im=m_table.find(((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName());
    GatePulse* outputPulse = new GatePulse(*inputPulse);
    if(im != m_table.end()){
            G4ThreeVector P = inputPulse->GetLocalPos();

         // G4cout<<((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName()<<" "<<(*im).second.sigmaSpblurr.x()<<"  "<<(*im).second.sigmaSpblurr.y()<<"  "<<(*im).second.sigmaSpblurr.z()<<G4endl;

            G4double Px = P.x();
            G4double Py = P.y();
            G4double Pz = P.z();
            G4double PxNew = G4RandGauss::shoot(Px,(*im).second.sigmaSpblurr.x()/GateConstants::fwhm_to_sigma);
            G4double PyNew = G4RandGauss::shoot(Py,(*im).second.sigmaSpblurr.y()/GateConstants::fwhm_to_sigma);
            G4double PzNew = G4RandGauss::shoot(Pz,(*im).second.sigmaSpblurr.z()/GateConstants::fwhm_to_sigma); //
            //Limits calculated for the bottom volume of the pulse) Maybe could  be done one for ecah volume and try to associate
               inputPulse->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kXAxis, limits, at, Xmin, Xmax);
               inputPulse->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kYAxis, limits, at, Ymin, Ymax);
               inputPulse->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kZAxis, limits, at, Zmin, Zmax);
               if(PxNew<Xmin) PxNew=Xmin;
               if(PxNew<Ymin) PyNew=Ymin;
               if(PzNew<Zmin) PzNew=Zmin;
               if(PxNew>Xmax) PxNew=Xmax;
               if(PyNew>Ymax) PyNew=Ymax;
               if(PzNew>Zmax) PzNew=Zmax;
               outputPulse->SetLocalPos(G4ThreeVector(PxNew,PyNew,PzNew)); //TC
               outputPulse->SetGlobalPos(outputPulse->GetVolumeID().MoveToAncestorVolumeFrame(outputPulse->GetLocalPos())); //TC

    }
    outputPulseList.push_back(outputPulse);
}

void GateCC3DlocalSpblurring::DescribeMyself(size_t indent)
{
  for (im=m_table.begin(); im!=m_table.end(); im++)
    G4cout << GateTools::Indent(indent) << "3D sigma blurring   " << (*im).first << ":\n"
       << GateTools::Indent(indent+1) << (*im).second.sigmaSpblurr.x()<<" "<<(*im).second.sigmaSpblurr.y()<<"  "<<(*im).second.sigmaSpblurr.z()<<Gateendl;
}
