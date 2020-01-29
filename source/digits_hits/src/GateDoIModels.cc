/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateDoIModels.hh"

#include "G4UnitsTable.hh"
#include"GateDualLayerLaw.hh"
#include "GateDoIModelsMessenger.hh"
#include "GateTools.hh"


GateDoIModels::GateDoIModels(GatePulseProcessorChain* itsChain,const G4String& itsName)
  //: GateVPulseProcessor(itsChain,itsName),m_DoIaxis(itsDoIAxis){
  : GateVPulseProcessor(itsChain,itsName){

  m_messenger = new GateDoIModelsMessenger(this);
  //m_DoILaw= new GateDualLayerLaw(GetObjectName());


  flgCorrectAxis=0;

//I DO NOT UNDERTAND WHY HERE AXIS VALUES FROM MESSENGER ARE STILL  NOT LOADED ? Where are loaded?
  //Are loaded before the procesing of pulses. But do notknow where


}

void GateDoIModels::SetDoIAxis( G4ThreeVector val) {
   // G4cout<<"setting DoI axis from messenger"<<G4endl;

    m_DoIaxis = val;
    //G4cout<<"axis=  "<<m_DoIaxis.getX()<<"  "<<m_DoIaxis.getY()<<"  "<<m_DoIaxis.getZ()<<G4endl;

    G4ThreeVector xAxis(1.0,0,0.0);
    G4ThreeVector yAxis(0.0,1.0,0.0);
    G4ThreeVector zAxis(0.0,0.0,1.0);

    // G4cout<<" is parallell to z "<<m_DoIaxis.isParallel(zAxis)<<G4endl;
    //G4cout<<"axis=  "<<m_DoIaxis.getX()<<"  "<<m_DoIaxis.getY()<<"  "<<m_DoIaxis.getZ()<<G4endl;
    if(m_DoIaxis.isParallel(xAxis)||m_DoIaxis.isParallel(yAxis)||m_DoIaxis.isParallel(zAxis)){
        flgCorrectAxis=1;

    }
    else{

        G4cout<<"[GateDoIModels::GateDoIModels]:  one of the three axis must be selected  for DoI:X, Y or Z."<<G4endl;
        G4cout<<"[GateDoIModels::GateDoIModels]:  DoI model has not been applied."<<G4endl;
    }
}






GateDoIModels::~GateDoIModels()
{
  delete m_messenger;
  delete m_DoILaw;
}







void GateDoIModels::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  if (!inputPulse) {
    if (nVerboseLevel>1)
        G4cout << "[GateDoIModels::ProcessOnePulse]: input pulse was null -> nothing to do\n\n";
    return;
  }
  if (inputPulse->GetEnergy()==0) {
    if (nVerboseLevel>1)
        G4cout << "[GateDoIModels::ProcessOneHit]: energy is null for " << inputPulse << " -> pulse ignored\n\n";
    return;
  }




  GatePulse* outputPulse = new GatePulse(*inputPulse);
  //G4cout << "eventID"<<inputPulse->GetEventID()<<"  effectEnergy="<<m_effectiveEnergyLaw->ComputeEffectiveEnergy(*outputPulse)<<G4endl;
  if(flgCorrectAxis==1){
    m_DoILaw->ComputeDoI(outputPulse, m_DoIaxis);
  }

      outputPulseList.push_back(outputPulse);
      if (nVerboseLevel>1)
          G4cout << "Copied pulse to output:\n"
                 << *outputPulse << Gateendl << Gateendl ;

}



void GateDoIModels::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "axis: (" << m_DoIaxis.getX()<<","<<m_DoIaxis.getY()<<","<<m_DoIaxis.getZ()<<Gateendl;
  G4cout << GateTools::Indent(indent) << "law: " << m_DoILaw<<Gateendl;
}
