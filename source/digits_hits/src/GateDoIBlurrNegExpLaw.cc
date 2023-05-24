/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  \class  GateDoIBlurrNegExpLaw

  The user can choose the direction in which he wants to applied the DoI model.

  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/

#include "G4VoxelLimits.hh"
#include "G4Transform3D.hh"
#include "G4VSolid.hh"

#include "GateDoIBlurrNegExpLaw.hh"

#include "GateTools.hh"
#include "GateVolumeID.hh"

#include "Randomize.hh"

#include "G4UnitsTable.hh"

#include "GateObjectStore.hh"
#include "GateConstants.hh"



GateDoIBlurrNegExpLaw::GateDoIBlurrNegExpLaw(const G4String& itsName) :
    GateVDoILaw(itsName)
{
    m_messenger = new GateDoIBlurrNegExpLawMessenger(this);
    xAxis.setX(1);
    yAxis.setY(1);
    zAxis.setZ(1);



}



void GateDoIBlurrNegExpLaw::ComputeDoI( GateDigi* digi, G4ThreeVector axis)  { //GateDoImodels* DoImodels

    //It is not efficient. Maybe create a map with the names of the volumes and  the limits in each direction ?? For local easy.
    //for global  how do I get the volume names. From sensitive detector. That information in the actor should be provided here ?

    G4ThreeVector newLocalPos;
    newLocalPos=digi->GetLocalPos();

    G4double doICrysEntSys=0;
    G4double newDoI;
    if(axis.isParallel(xAxis)){
         // G4cout<<"[GateDoIBlurrNegExpLaw]:  DoI model applied in X direction."<<G4endl;
        digi->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kXAxis, limits, at, DoImin, DoImax);

        if(axis.getX()>0){
              doICrysEntSys=(DoImax-DoImin)/2-digi->GetLocalPos().getX();

        }
        else{
              doICrysEntSys=(DoImax-DoImin)/2-digi->GetLocalPos().getX();
        }
        newDoI=G4RandGauss::shoot(newLocalPos.getX(),(GetEntranceFWHM()/GateConstants::fwhm_to_sigma)*exp(doICrysEntSys/GetExpInvDecayConst()));
        if(newDoI<DoImin) newDoI=DoImin;
        if(newDoI>DoImax) newDoI=DoImax;
        newLocalPos.setX(newDoI);



    }
    else if(axis.isParallel(yAxis)){
        // G4cout<<"[GateDoIBlurrNegExpLaw]:  DoI model applied in Y direction."<<G4endl;
        digi->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kYAxis, limits, at, DoImin, DoImax);
        if(axis.getY()>0){
            doICrysEntSys=digi->GetLocalPos().getY()+(DoImax-DoImin)/2;
        }
        else{
            doICrysEntSys=(DoImax-DoImin)/2-digi->GetLocalPos().getY();
        }
        newDoI=G4RandGauss::shoot(newLocalPos.getY(),(GetEntranceFWHM()/GateConstants::fwhm_to_sigma)*exp(doICrysEntSys/GetExpInvDecayConst()));
        if(newDoI<DoImin) newDoI=DoImin;
        if(newDoI>DoImax) newDoI=DoImax;
        newLocalPos.setY(newDoI);

    }
    else{
        // G4cout<<"[GateDoIBlurrNegExpLaw]:  DoI model applied in Z direction."<<G4endl;
        //Limits in local system. DoI min and masx equals differnet signe
        digi->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kZAxis, limits, at, DoImin, DoImax);
        //G4cout<<digi.GetVolumeID().GetBottomCreator()->GetSolidName()<<"Xmin="<<DoImin<<"  Xmax"<<DoImax<<G4endl;

        if(axis.getZ()>0){
            doICrysEntSys=newLocalPos.getZ()+(DoImax-DoImin)/2;
        }
        else{
            doICrysEntSys=(DoImax-DoImin)/2-newLocalPos.getZ();
        }
        //G4cout<<" localDoI= "<<newLocalPos.getZ()<<" SRDoI="<<doICrysEntSys<<"  sigma="<<(GetEntranceFWHM()/GateConstants::fwhm_to_sigma)*exp(doICrysEntSys/GetExpInvDecayConst())<<G4endl;
       newDoI=G4RandGauss::shoot(newLocalPos.getZ(),(GetEntranceFWHM()/GateConstants::fwhm_to_sigma)*exp(doICrysEntSys/GetExpInvDecayConst()));
       if(newDoI<DoImin) newDoI=DoImin;
       if(newDoI>DoImax) newDoI=DoImax;
       newLocalPos.setZ(newDoI);
    }


    digi->SetLocalPos(newLocalPos);
    digi->SetGlobalPos(digi->GetVolumeID().MoveToAncestorVolumeFrame(digi->GetLocalPos()));





}

void GateDoIBlurrNegExpLaw::DescribeMyself (size_t indent) const {
    G4cout << "DoI DoIBlurrNegExp model\n";
    G4cout << GateTools::Indent(indent) << "Exponential Inv decay const:\t" << G4BestUnit(GetExpInvDecayConst(),"Length") << Gateendl;
    G4cout << GateTools::Indent(indent) << "DoI FWHM at detector entrance:\t" << G4BestUnit(GetEntranceFWHM(),"Length")<< Gateendl;

}
