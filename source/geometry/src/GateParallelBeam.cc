/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateParallelBeam.hh"

#include "G4VisAttributes.hh"

#include "GateBox.hh"
#include "GateHexagone.hh"
#include "GateTrap.hh"
#include "GateObjectChildList.hh"
#include "GateObjectRepeaterList.hh"
#include "GateVolumePlacement.hh"
#include "GateParallelBeamMessenger.hh"
#include "GateMaterialDatabase.hh"
#include "G4UnitsTable.hh"

//----------------------------------------------------------------------------------------------------------
GateParallelBeam::GateParallelBeam(const G4String& itsName, G4bool acceptsChildren, G4int depth)
   : GateBox(itsName,"Vacuum",1,1,1,acceptsChildren,depth),
     m_SeptalThickness(1.),m_InnerRadius(1.),m_Height(1.),
     m_DimensionX(1.),m_DimensionY(1.),mMaterialName("Vacuum")
{
   PreComputeConstants ();

   m_lineInserter = new GateBoxReplicaPlacement(this,"lines",mMaterialName,m_DimensionX,l_Width,m_Height,l_Delta,kYAxis,l_N);
   GetCreator()->GetTheChildList()->AddChild(m_lineInserter);
   
   m_boxInserter = new GateBoxReplicaPlacement(this,"boxes",mMaterialName,b_Length,b_Width,m_Height,b_Delta,kXAxis,b_N);
   m_lineInserter->GetCreator()->GetTheChildList()->AddChild(m_boxInserter);
   
   m_CentralHole = new GateHexagone ("central hole","Air",m_InnerRadius,m_Height);
   m_boxInserter->GetCreator()->GetTheChildList()->AddChild(m_CentralHole);


   m_SideTrap1 = new GateTrap ("side hole 1","Air",0.5*m_Height,0.0,0.0,0.5*m_InnerRadius,0.25*m_TrapSideSmall,
   	0.5*m_TrapSideSmall,-m_TrapAngle,0.5*m_InnerRadius,0.25*m_TrapSideSmall,0.5*m_TrapSideSmall,-m_TrapAngle);   
   m_SideTrap1->GetVolumePlacement()->SetTranslation(m_Translation1);
   
   
   
   m_SideTrap2 = new GateTrap ("side hole 2","Air",0.5*m_Height,0.0,0.0,0.5*m_InnerRadius,0.25*m_TrapSideSmall,
   	0.5*m_TrapSideSmall,m_TrapAngle,0.5*m_InnerRadius,0.25*m_TrapSideSmall,0.5*m_TrapSideSmall,m_TrapAngle);
   m_SideTrap2->GetVolumePlacement()->SetTranslation(m_Translation2);
   
   
   m_SideTrap3 = new GateTrap ("side hole 3","Air",0.5*m_Height,0.0,0.0,0.5*m_InnerRadius,0.5*m_TrapSideSmall,
   	0.25*m_TrapSideSmall,m_TrapAngle,0.5*m_InnerRadius,0.5*m_TrapSideSmall,0.25*m_TrapSideSmall,m_TrapAngle);
   m_SideTrap3->GetVolumePlacement()->SetTranslation(m_Translation3);
   
   
   	
   m_SideTrap4 = new GateTrap ("side hole 4","Air",0.5*m_Height,0.0,0.0,0.5*m_InnerRadius,0.5*m_TrapSideSmall,
   	0.25*m_TrapSideSmall,-m_TrapAngle,0.5*m_InnerRadius,0.5*m_TrapSideSmall,0.25*m_TrapSideSmall,-m_TrapAngle);
   m_SideTrap4->GetVolumePlacement()->SetTranslation(m_Translation4);
 
   m_boxInserter->GetCreator()->GetTheChildList()->AddChild(m_SideTrap1);
   m_boxInserter->GetCreator()->GetTheChildList()->AddChild(m_SideTrap2);
   m_boxInserter->GetCreator()->GetTheChildList()->AddChild(m_SideTrap3);
   m_boxInserter->GetCreator()->GetTheChildList()->AddChild(m_SideTrap4);

   
   m_messenger = new GateParallelBeamMessenger(this);
}
//--------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------
GateParallelBeam::GateParallelBeam(const G4String& itsName,const G4String& itsMaterialName,
    G4double itsSeptalThickness,G4double itsInnerRadius,G4double itsHeight, G4double itsDimensionX, G4double itsDimensionY)
   : GateBox(itsName,itsMaterialName,itsDimensionX,itsDimensionY,itsHeight,false,false),
     m_SeptalThickness(itsSeptalThickness),m_InnerRadius(itsInnerRadius),m_Height(itsHeight),
     m_DimensionX(itsDimensionX),m_DimensionY(itsDimensionY),mMaterialName(itsMaterialName)
{
  PreComputeConstants ();

  m_lineInserter = new GateBoxReplicaPlacement(this,"lines",mMaterialName,m_DimensionX,l_Width,m_Height,l_Delta,kYAxis,l_N);
  GetCreator()->GetTheChildList()->AddChild(m_lineInserter);
   
   m_boxInserter = new GateBoxReplicaPlacement(this,"boxes",mMaterialName,b_Length,b_Width,m_Height,b_Delta,kXAxis,b_N);
   m_lineInserter->GetCreator()->GetTheChildList()->AddChild(m_boxInserter);
   
   m_CentralHole = new GateHexagone ("central hole","Air",m_InnerRadius,m_Height);
   m_boxInserter->GetCreator()->GetTheChildList()->AddChild(m_CentralHole);


   m_SideTrap1 = new GateTrap ("side hole 1","Air",0.5*m_Height,0.0,0.0,0.5*m_InnerRadius,0.25*m_TrapSideSmall,
   	0.5*m_TrapSideSmall,-m_TrapAngle,0.5*m_InnerRadius,0.25*m_TrapSideSmall,0.5*m_TrapSideSmall,-m_TrapAngle);   
   m_SideTrap1->GetVolumePlacement()->SetTranslation(m_Translation1);
   
   
   
  m_SideTrap2 = new GateTrap ("side hole 2","Air",0.5*m_Height,0.0,0.0,0.5*m_InnerRadius,0.25*m_TrapSideSmall,
			      0.5*m_TrapSideSmall,m_TrapAngle,0.5*m_InnerRadius,0.25*m_TrapSideSmall,0.5*m_TrapSideSmall,m_TrapAngle);
  m_SideTrap2->GetVolumePlacement()->SetTranslation(m_Translation2);
   
   
  m_SideTrap3 = new GateTrap ("side hole 3","Air",0.5*m_Height,0.0,0.0,0.5*m_InnerRadius,0.5*m_TrapSideSmall,
			      0.25*m_TrapSideSmall,m_TrapAngle,0.5*m_InnerRadius,0.5*m_TrapSideSmall,0.25*m_TrapSideSmall,m_TrapAngle);
  m_SideTrap3->GetVolumePlacement()->SetTranslation(m_Translation3);
   
   
   	
  m_SideTrap4 = new GateTrap ("side hole 4","Air",0.5*m_Height,0.0,0.0,0.5*m_InnerRadius,0.5*m_TrapSideSmall,
			      0.25*m_TrapSideSmall,-m_TrapAngle,0.5*m_InnerRadius,0.5*m_TrapSideSmall,0.25*m_TrapSideSmall,-m_TrapAngle);
  m_SideTrap4->GetVolumePlacement()->SetTranslation(m_Translation4);
 
  m_boxInserter->GetCreator()->GetTheChildList()->AddChild(m_SideTrap1);
  m_boxInserter->GetCreator()->GetTheChildList()->AddChild(m_SideTrap2);
  m_boxInserter->GetCreator()->GetTheChildList()->AddChild(m_SideTrap3);
  m_boxInserter->GetCreator()->GetTheChildList()->AddChild(m_SideTrap4);


   m_messenger = new GateParallelBeamMessenger(this);
}
//--------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------
GateParallelBeam::~GateParallelBeam()
{
  delete m_messenger;
}
//--------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------
void GateParallelBeam::PreComputeConstants()
{

  m_TrapSideSmall = 2.0*m_InnerRadius/sqrt(3.0);
  m_TrapSideBig = 2.0*(m_InnerRadius+m_SeptalThickness)/sqrt(3.0);
  m_TrapAngle = atan (0.5/sqrt(3.0));

  //   m_TrapAngle = 0.0;

  l_Width  = 2. * (m_InnerRadius + m_SeptalThickness);
  l_Delta  = 2. * (m_InnerRadius + m_SeptalThickness);
  l_N      = int ( m_DimensionY / l_Delta );
  m_DimensionYn = l_Delta * l_N;

  b_Length = 2.0 * m_TrapSideSmall + m_TrapSideBig;
  b_Delta  = 2.0 * m_TrapSideSmall + m_TrapSideBig;
  b_Width  = 2. * (m_InnerRadius + m_SeptalThickness);
  b_N      = int ( m_DimensionX / b_Delta );
  m_DimensionXn = b_Delta * b_N;

  m_Translation1 = G4ThreeVector (0.625*m_TrapSideSmall+0.5*m_TrapSideBig,0.5*m_InnerRadius+m_SeptalThickness,0);
  m_Translation2 = G4ThreeVector (-0.625*m_TrapSideSmall-0.5*m_TrapSideBig,0.5*m_InnerRadius+m_SeptalThickness,0);
  m_Translation3 = G4ThreeVector (0.625*m_TrapSideSmall+0.5*m_TrapSideBig,-0.5*m_InnerRadius-m_SeptalThickness,0);
  m_Translation4 = G4ThreeVector (-0.625*m_TrapSideSmall-0.5*m_TrapSideBig,-0.5*m_InnerRadius-m_SeptalThickness,0);
}
//--------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------
void GateParallelBeam::ResizeParallelBeam()
{
  PreComputeConstants();

   GetParallelBeamCreator()->SetBoxXLength(m_DimensionXn);
   GetParallelBeamCreator()->SetBoxYLength(m_DimensionYn);
   GetParallelBeamCreator()->SetBoxZLength(m_Height);
   GetParallelBeamCreator()->SetMaterialName(mMaterialName);

   // Update the Hexagone-inserter
   m_lineInserter->Update(mMaterialName,m_DimensionXn,l_Width,m_Height,l_Delta,l_N);
   m_boxInserter->Update(mMaterialName,b_Length,b_Width,m_Height,b_Delta,b_N);
   m_CentralHole->SetHexagoneRadius(m_InnerRadius);
   m_CentralHole->SetHexagoneHeight(m_Height);

  m_SideTrap1->SetTrapDz(0.5*m_Height);
  m_SideTrap1->SetTrapDy1(0.5*m_InnerRadius);
  m_SideTrap1->SetTrapDx1(0.25*m_TrapSideSmall);
  m_SideTrap1->SetTrapDx2(0.5*m_TrapSideSmall);
  m_SideTrap1->SetTrapAlp1(-m_TrapAngle);
  m_SideTrap1->SetTrapDy2(0.5*m_InnerRadius);
  m_SideTrap1->SetTrapDx3(0.25*m_TrapSideSmall);
  m_SideTrap1->SetTrapDx4(0.5*m_TrapSideSmall);
  m_SideTrap1->SetTrapAlp2(-m_TrapAngle);

  m_SideTrap2->SetTrapDz(0.5*m_Height);
  m_SideTrap2->SetTrapDy1(0.5*m_InnerRadius);
  m_SideTrap2->SetTrapDx1(0.25*m_TrapSideSmall);
  m_SideTrap2->SetTrapDx2(0.5*m_TrapSideSmall);
  m_SideTrap2->SetTrapAlp1(m_TrapAngle);
  m_SideTrap2->SetTrapDy2(0.5*m_InnerRadius);
  m_SideTrap2->SetTrapDx3(0.25*m_TrapSideSmall);
  m_SideTrap2->SetTrapDx4(0.5*m_TrapSideSmall);
  m_SideTrap2->SetTrapAlp2(m_TrapAngle);

  m_SideTrap3->SetTrapDz(0.5*m_Height);
  m_SideTrap3->SetTrapDy1(0.5*m_InnerRadius);
  m_SideTrap3->SetTrapDx1(0.5*m_TrapSideSmall);
  m_SideTrap3->SetTrapDx2(0.25*m_TrapSideSmall);
  m_SideTrap3->SetTrapAlp1(m_TrapAngle);
  m_SideTrap3->SetTrapDy2(0.5*m_InnerRadius);
  m_SideTrap3->SetTrapDx3(0.5*m_TrapSideSmall);
  m_SideTrap3->SetTrapDx4(0.25*m_TrapSideSmall);
  m_SideTrap3->SetTrapAlp2(m_TrapAngle);

  m_SideTrap4->SetTrapDz(0.5*m_Height);
  m_SideTrap4->SetTrapDy1(0.5*m_InnerRadius);
  m_SideTrap4->SetTrapDx1(0.5*m_TrapSideSmall);
  m_SideTrap4->SetTrapDx2(0.25*m_TrapSideSmall);
  m_SideTrap4->SetTrapAlp1(-m_TrapAngle);
  m_SideTrap4->SetTrapDy2(0.5*m_InnerRadius);
  m_SideTrap4->SetTrapDx3(0.5*m_TrapSideSmall);
  m_SideTrap4->SetTrapDx4(0.25*m_TrapSideSmall);
  m_SideTrap4->SetTrapAlp2(-m_TrapAngle);

  m_SideTrap1->GetVolumePlacement()->SetTranslation(m_Translation1);
  m_SideTrap2->GetVolumePlacement()->SetTranslation(m_Translation2);
  m_SideTrap3->GetVolumePlacement()->SetTranslation(m_Translation3);
  m_SideTrap4->GetVolumePlacement()->SetTranslation(m_Translation4);
}
//--------------------------------------------------------------------------------------------------------------


