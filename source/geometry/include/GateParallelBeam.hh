/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateParallelBeam_h
#define GateParallelBeam_h 1

#include "vector"

#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"

#include "GateBox.hh"
#include "GateBoxReplicaPlacement.hh"

class GateHexagone;
class GateTrap;
class GateBoxReplicaPlacement;
class G4Material;

class GateParallelBeamMessenger;

class GateParallelBeam : public GateBox
{
   public:
     // Constructor1
     GateParallelBeam(const G4String& itsName, G4bool acceptsChildren=true, G4int depth=0);
     
     // Constructor2
     GateParallelBeam(const G4String& itsName,const G4String& itsMaterialName,
     			      G4double itsSeptalThickness,G4double itsInnerRadius,G4double itsHeight,
			      G4double itsDimensionX, G4double itsDimensionY);

     virtual ~GateParallelBeam();

     FCT_FOR_AUTO_CREATOR_VOLUME(GateParallelBeam)

     virtual void ResizeParallelBeam();

     void PreComputeConstants();
     
     GateBox* GetParallelBeamCreator() const
      { return (GateBox*) GetCreator(); }

     inline G4double GetParallelBeamSeptalThickness() const
	{ return m_SeptalThickness; }
     inline G4double GetParallelBeamInnerRadius() const
	{ return m_InnerRadius; }
     inline G4double GetParallelBeamHeight() const
	{ return m_Height; }
     inline G4double GetParallelBeamDimensionX() const
	{ return m_DimensionX; }
     inline G4double GetParallelBeamDimensionY() const
	{ return m_DimensionY; }
     inline const G4String& GetParallelBeamMaterial() const
	{ return mMaterialName; }

     inline void SetParallelBeamSeptalThickness(G4double val)
	{ m_SeptalThickness = val; ResizeParallelBeam();}
     inline void SetParallelBeamInnerRadius(G4double val)
      	{ m_InnerRadius = val; ResizeParallelBeam(); }
     inline void SetParallelBeamHeight(G4double val)
	{ m_Height = val; ResizeParallelBeam(); }
     inline void SetParallelBeamDimensionX(G4double val)
	{ m_DimensionX = val; ResizeParallelBeam(); }
     inline void SetParallelBeamDimensionY(G4double val)
	{ m_DimensionY = val; ResizeParallelBeam();}
     inline void SetParallelBeamMaterial(const G4String& val)
	{ mMaterialName = val; ResizeParallelBeam();}


   protected:
     G4double m_SeptalThickness,m_InnerRadius,m_Height,m_DimensionX,m_DimensionY;
     G4double m_DimensionXn,m_DimensionYn;
     G4String mMaterialName;
     G4double b_Delta,b_Width,b_Length;
     G4double l_Delta,l_Width;
     G4double m_TrapSideSmall,m_TrapSideBig,m_TrapAngle;
     G4int    b_N;
     G4int    l_N;

     G4ThreeVector  m_Translation1;
     G4ThreeVector  m_Translation2;
     G4ThreeVector  m_Translation3;
     G4ThreeVector  m_Translation4;

     GateBoxReplicaPlacement* m_lineInserter;
     GateBoxReplicaPlacement* m_boxInserter;

     GateHexagone*         m_CentralHole;
     GateTrap*             m_SideTrap1;
     GateTrap*             m_SideTrap2;
     GateTrap*             m_SideTrap3;
     GateTrap*             m_SideTrap4;


     GateParallelBeamMessenger* m_messenger;
};

MAKE_AUTO_CREATOR_VOLUME(parallelbeam,GateParallelBeam)

#endif
