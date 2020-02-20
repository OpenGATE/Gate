#include "G4VoxelLimits.hh"
#include "G4Transform3D.hh"
#include "G4VSolid.hh"

#include "GateDualLayerLaw.hh"

#include "GateCC3DlocalSpblurring.hh"

#include "GateCC3DlocalSpblurringMessenger.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"

#include "Randomize.hh"

#include "G4UnitsTable.hh"

#include "GateObjectStore.hh"
#include "GateConstants.hh"



GateDualLayerLaw::GateDualLayerLaw(const G4String& itsName) :
    GateVDoILaw(itsName)
{
    m_messenger = new GateDualLayerLawMessenger(this);
    xAxis.setX(1);
    yAxis.setY(1);
    zAxis.setZ(1);



}



void GateDualLayerLaw::ComputeDoI(GatePulse* pulse, G4ThreeVector axis)  {

    //It is not efficient. Maybe create a map with the names of the volumes and  the limits in each direction ?? For local easy.
    //for global  how do I get the volume names. From senssitive detecotr. That information in the actor shoulf be provided here ?

    G4ThreeVector newLocalPos;
    newLocalPos=pulse->GetLocalPos();

    if(axis.isParallel(xAxis)){
         // G4cout<<"[GateDualLayerLaw]:  DoI model applied in X direction."<<G4endl;
        pulse->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kXAxis, limits, at, DoImin, DoImax);

        if(axis.getX()>0){
            if(pulse->GetLocalPos().getX()<=0){
                newLocalPos.setX(DoImin);
            }
            else{
                newLocalPos.setX(0.0);
            }

        }
        else{
            if(pulse->GetLocalPos().getX()<=0){
                newLocalPos.setX(0.0);
            }
            else{
                newLocalPos.setX(DoImax);
            }
        }


    }
    else if(axis.isParallel(yAxis)){
        // G4cout<<"[GateDualLayerLaw]:  DoI model applied in Y direction."<<G4endl;
        pulse->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kYAxis, limits, at, DoImin, DoImax);
        if(axis.getY()>0){
            if(pulse->GetLocalPos().getY()<=0){
                newLocalPos.setY(DoImin);
            }
            else{
                newLocalPos.setY(0);
            }
        }
        else{
            if(pulse->GetLocalPos().getY()<=0){
                newLocalPos.setY(0);
            }
            else{
                newLocalPos.setY(DoImax);
            }

        }

    }
    else{
        // G4cout<<"[GateDualLayerLaw]:  DoI model applied in Z direction."<<G4endl;
        pulse->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kZAxis, limits, at, DoImin, DoImax);
        //G4cout<<pulse.GetVolumeID().GetBottomCreator()->GetSolidName()<<"Xmin="<<DoImin<<"  Xmax"<<DoImax<<G4endl;

        if(axis.getZ()>0){
            if(pulse->GetLocalPos().getZ()<=0){
                newLocalPos.setZ(DoImin);
            }
            else{
                newLocalPos.setZ(0);
            }
        }
        else{
            if(pulse->GetLocalPos().getZ()<=0){
                newLocalPos.setZ(0);
            }
            else{
                newLocalPos.setZ(DoImax);
            }

        }
    }


    pulse->SetLocalPos(newLocalPos);
    pulse->SetGlobalPos(pulse->GetVolumeID().MoveToAncestorVolumeFrame(pulse->GetLocalPos()));





}

void GateDualLayerLaw::DescribeMyself (size_t indent) const {
    G4cout << GateTools::Indent(indent) << "DoI dualLayer model"<< Gateendl;
}
