
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
  DescribeMyself(2);
  if (m_store->FindCreator(val)!=0) {
      m_param.sigmaSpblurr.setX(-1);
      m_param.sigmaSpblurr.setY(-1);
      m_param.sigmaSpblurr.setZ(-1);


      /*Before it was working. Now The solidVolume is not created yet (
       * initialization change?) so I can not load here the size of the detector
       * G4VoxelLimits limits;
      G4double min, max;
      G4AffineTransform at;
      G4cout <<"logical "<< m_store->FindCreator(val)->GetLogicalVolume() << G4endl;
      G4cout <<"solid "<< m_store->FindCreator(val)->GetLogicalVolume()->GetSolid() << G4endl;
      m_store->FindCreator(val)->GetLogicalVolume()->GetSolid()->CalculateExtent(kXAxis, limits, at, min, max);*/

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
        //G4cout<<((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName()<<" "<<(*im).second.sigmaSpblurr.x()<<"  "<<(*im).second.sigmaSpblurr.y()<<"  "<<(*im).second.sigmaSpblurr.z()<<G4endl;
        G4double Px = P.x();
        G4double Py = P.y();
        G4double Pz = P.z();
        // G4cout<<"eventID "<<inputPulse->GetEventID()<<"vX"<<(*im).second.sigmaSpblurr.x()/GateConstants::fwhm_to_sigma<<G4endl;
        G4double PxNew = G4RandGauss::shoot(Px,(*im).second.sigmaSpblurr.x());
        G4double PyNew = G4RandGauss::shoot(Py,(*im).second.sigmaSpblurr.y());
        G4double PzNew = G4RandGauss::shoot(Pz,(*im).second.sigmaSpblurr.z()); //
        //Limits calculated for the bottom volume of the pulse. Maybe try to do it once for each volume and then find each time the corresponding value  in a map?
        inputPulse->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kXAxis, limits, at, Xmin, Xmax);
        inputPulse->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kYAxis, limits, at, Ymin, Ymax);
        inputPulse->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kZAxis, limits, at, Zmin, Zmax);
        if(PxNew<Xmin) PxNew=Xmin;
        if(PyNew<Ymin) PyNew=Ymin;
        if(PzNew<Zmin) PzNew=Zmin;
        if(PxNew>Xmax) PxNew=Xmax;
        if(PyNew>Ymax) PyNew=Ymax;
        if(PzNew>Zmax) PzNew=Zmax;
        outputPulse->SetLocalPos(G4ThreeVector(PxNew,PyNew,PzNew)); //TC
        outputPulse->SetGlobalPos(outputPulse->GetVolumeID().MoveToAncestorVolumeFrame(outputPulse->GetLocalPos())); //TC
       //Errors need to be set for better calculation of CCSR outputPulse->SetGlobalPosError((*im).second.sigmaSpblurr)
    }
    outputPulseList.push_back(outputPulse);
}

void GateCC3DlocalSpblurring::DescribeMyself(size_t indent)
{
    for (im=m_table.begin(); im!=m_table.end(); im++)
        G4cout << GateTools::Indent(indent) << "3D sigma blurring   " << (*im).first << ":\n"
               << GateTools::Indent(indent+1) << (*im).second.sigmaSpblurr.x()<<" "<<(*im).second.sigmaSpblurr.y()<<"  "<<(*im).second.sigmaSpblurr.z()<<Gateendl;
}
