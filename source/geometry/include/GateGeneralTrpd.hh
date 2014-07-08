/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEGENERALTRPD_H
#define GATEGENERALTRPD_H 1

#include "globals.hh"
#include "GateVVolume.hh"
#include "GateVolumeManager.hh"
#include "GateObjectChildList.hh"
#include "G4Trap.hh"

class G4LogicalVolume;
class G4VPhysicalVolume;
class G4Material;
class GateGeneralTrpdMessenger;

//---------------------------------------------------------------------
class GateGeneralTrpd  : public GateVVolume
{
public:
  GateGeneralTrpd(const G4String& itsName, G4bool acceptsChildren=true, G4int depth=0); 
  GateGeneralTrpd(const G4String& itsName,const G4String& itsMaterialName,
                  G4double itsX1Length, G4double itsY1Length,
                  G4double itsX2Length, G4double itsY2Length,
                  G4double itsX3Length, G4double itsX4Length,
                  G4double itsZLength,
                  G4double itsTheta,    G4double itsPhi,
                  G4double itsAlp1,     G4double itsAlp2,
                  G4bool itsFlagAcceptChildren=true,
                  G4int depth=0);
  virtual ~GateGeneralTrpd();
   
  FCT_FOR_AUTO_CREATOR_VOLUME(GateGeneralTrpd)

  virtual void DestroyOwnVolume();
  void DestroyOwnSolidAndLogicalVolume();

  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly); 
  virtual void DescribeMyself(size_t indent);
  inline G4double GetHalfDimension(size_t /*axis*/)	  {return 0.; }

  void SetTrpdX1Length(G4double val)	{m_X1=val;}
  void SetTrpdX2Length(G4double val)	{m_X2=val;}
  void SetTrpdX3Length(G4double val)	{m_X3=val;}
  void SetTrpdX4Length(G4double val)	{m_X4=val;}
  void SetTrpdY1Length(G4double val)	{m_Y1=val;}
  void SetTrpdY2Length(G4double val)	{m_Y2=val;}
  void SetTrpdZLength(G4double val)	{m_Z=val;}
  void SetTrpdTheta(G4double val)	{m_Theta=val;}
  void SetTrpdPhi(G4double val)		{m_Phi=val;}
  void SetTrpdAlp1(G4double val)	{m_Alp1=val;}
  void SetTrpdAlp2(G4double val)	{m_Alp2=val;}

  inline G4double GetTrpdX1Length()	{return m_X1;}
  inline G4double GetTrpdX2Length()	{return m_X2;}
  inline G4double GetTrpdX3Length()	{return m_X3;}
  inline G4double GetTrpdX4Length()	{return m_X4;}
  inline G4double GetTrpdY1Length()	{return m_Y1;}
  inline G4double GetTrpdY2Length()	{return m_Y2;}
  inline G4double GetTrpdZLength()	{return m_Z;}
  inline G4double GetTrpdTheta()	{return m_Theta;}
  inline G4double GetTrpdPhi()		{return m_Phi;}
  inline G4double GetTrpdAlp1()		{return m_Alp1;}
  inline G4double GetTrpdAlp2()		{return m_Alp2;}

protected:
  G4Trap*             m_general_trpd_solid;  //!< Solid pointer
  G4LogicalVolume*    m_general_trpd_log;    //!< logical volume pointer for extruded trapezoid
  G4VPhysicalVolume*  pGeneralTrpdPhys;
  G4double m_X1,m_X2,m_X3,m_X4,m_Y1,m_Y2,m_Z,m_Theta,m_Phi,m_Alp1,m_Alp2;
  //! Messenger
  GateGeneralTrpdMessenger* m_Messenger; 

};
//---------------------------------------------------------------------

MAKE_AUTO_CREATOR_VOLUME(general_trpd, GateGeneralTrpd)

#endif
