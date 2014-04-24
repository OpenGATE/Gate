/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateTrpd_h
#define GateTrpd_h 1

#include "globals.hh"

#include "GateVVolume.hh"
#include "GateVolumeManager.hh"

#include "G4RunManager.hh"
#include "G4VVisManager.hh"
#include "G4Trd.hh"
#include "G4Box.hh"
#include "G4SubtractionSolid.hh"
#include "G4ThreeVector.hh"

class GateTrpdMessenger;

/*! \class  GateTrpd
  \brief  GateTrpd creates a logical volume with a regular extruded Trapezoid (Trpd acronym) shape
  based on the GEANT4 boolean "subtraction" solid an with a box shaped extrusion. See Geant4 doc for further
  details.
  \brief The definition of the Trd solidis based on 5 lengths : dx1,dx2,dz1,dz2,dz defining where :
  \brief dx1 & dy1 give the length of the first X-Y rectangle crossing Z axe at -dz position
  \brief dy2 & dy2 give the length of the first X-Y rectangle crossing Z axe at +dz position

  \brief The definition of the box shaped extruded volume is base on the 3 lengths/ the 3 coordinates
  of the box length/ center position.
  By extension, this boolean volume can also be used to create any shapes defined as a instersection with a
  "Trd" and a rectangle, like for example a simple prism.

  \brief  - GateTrpd - by jean-marc.vieira@iphe.unil.ch
*/

/* 22/10/2007
   GateTrpd create also the physical volume
   virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material*);
   virtual G4VPhysicalVolume* ConstructOwnPhysicalVolume();
   virtual void DestroyOwnSolidAndLogicalVolume();
   changes by emilia.becheva@cea.fr
*/


class GateTrpd  : public GateVVolume
{
public:


  //! Constructor, by defaut extruded box placed at origin

  GateTrpd(const G4String& itsName,
           G4bool acceptsChildren=true,
           G4int depth=0);

  GateTrpd(const G4String& itsName,const G4String& itsMaterialName,
           G4double itsX1Length, G4double itsY1Length,
           G4double itsX2Length, G4double itsY2Length,
           G4double itsZLength,
           G4double itsXBxLength = 1.,
           G4double itsYBxLength = 1.,
           G4double itsZBxLength = 1.,
           G4double itsXBxPos = 0.,
           G4double itsYBxPos = 0.,
           G4double itsZBxPos = 0.,
           G4bool acceptsChildren=true,
           G4int depth=0);
  //! Destructor
  virtual ~GateTrpd();

  FCT_FOR_AUTO_CREATOR_VOLUME(GateTrpd)

  //! \name Implementations of pure virtual methods declared by the base-class
  //@{

  //! Implementation of the pure virtual method ConstructOwnSolidAndLogical() declared by the base-class.
  //! Construct a new caps shape solid ("trapezoid minus placed box") and its logical volume.
  //! If flagUpdateOnly is set to 1, the Trd is updated rather than rebuilt.
  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material*, G4bool);
  //  virtual G4VPhysicalVolume* ConstructOwnPhysicalVolume();

  //! Implementation of the pure virtual method DestroyOwnSolidAndLogicalVolume() declared by the base-class.
  //! Destroy the solid and logical volume created by ConstructOwnLogicalVolume()
  virtual void DestroyOwnSolidAndLogicalVolume();

  /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
    \brief a description of the creator

    \param indent: the print-out indentation (cosmetic parameter)
  */
  virtual void DescribeMyself(size_t indent);

  //! Implementation of the pure virtual method GetHalfDimension() declared by the base-class
  //! Returns the half-size of the Trd along an axis (X1=0, Y1=1, X2=2, Y2=3, Z=4 )
  //! For Trapezoid, we give the mean cross section size.
  inline G4double GetHalfDimension(size_t axis)	  {return GetTrpdHalfLength(axis); }

  //! Overload of the dummy virtual method ComputeMyOwnVolume() defined by the base-class
  //! Returns the volume of the solid
  //! -WARNING- : This works only if the box is totally included in the trapezoid shape !!!
  inline G4double ComputeMyOwnVolume()  const
  {return ( (m_trpdLength[4]/6.) *
            (m_trpdLength[0]*m_trpdLength[1]
             + (m_trpdLength[0]+m_trpdLength[2])*(m_trpdLength[1]+m_trpdLength[3])
             + m_trpdLength[2]*m_trpdLength[3] ) // trapezoid volume
            - m_trpdLength[5]*m_trpdLength[6]*m_trpdLength[7]); } //extruded box volume

  //@}

  //! \name getters and setters
  //@{


  //! S E T T E R S

  //! Set the Trd length along X
  void SetTrpdX1Length (G4double val)
  {  m_trpdLength[0] = val; /*ComputeParameters();*/ }
  //! Set the Trd length along Y
  void SetTrpdY1Length (G4double val)
  {  m_trpdLength[1] = val; /*ComputeParameters();*/ }
  //! Set the Trd length along X
  void SetTrpdX2Length (G4double val)
  {  m_trpdLength[2] = val; /*ComputeParameters();*/ }
  //! Set the Trd length along Y
  void SetTrpdY2Length (G4double val)
  {  m_trpdLength[3] = val; /*ComputeParameters();*/ }
  //! Set the Trd length along Z
  void SetTrpdZLength (G4double val)
  {  m_trpdLength[4] = val; /*ComputeParameters();*/ }
  //! Set the extruded box X lentgh
  void SetTrpdTrudXLength (G4double val)
  {  m_trpdLength[5] = val; /*ComputeParameters();*/ }
  //! Set the extruded box Y lentgh
  void SetTrpdTrudYLength (G4double val)
  {  m_trpdLength[6] = val; /*ComputeParameters();*/ }
  //! Set the extruded box Z lentgh
  void SetTrpdTrudZLength (G4double val)
  {  m_trpdLength[7] = val; /*ComputeParameters();*/ }
  //! Set the extruded box X position
  void SetTrpdTrudXPos (G4double val)
  {  m_trpdLength[8] = val; /*ComputeParameters();*/ }
  //! Set the extruded box Y position
  void SetTrpdTrudYPos (G4double val)
  {  m_trpdLength[9] = val; /*ComputeParameters();*/ }
  //! Set the extruded box Z position
  void SetTrpdTrudZPos (G4double val)
  {  m_trpdLength[10]= val; /*ComputeParameters();*/ }

  // ! G E T T E R S

  //! Get the Trd length along an axis  (X1=0, Y1=1, X2=2, Y2=3, Z=4)
  inline G4double GetTrpdLength(size_t axis)      {return m_trpdLength[axis];}
  //! Get the Trd length along X1-Y1
  inline G4double GetTrpdX1Length()               {return GetTrpdLength(0);}
  inline G4double GetTrpdY1Length()               {return GetTrpdLength(1);}
  //! Get the Trd length along X2-Y2
  inline G4double GetTrpdX2Length()               {return GetTrpdLength(2);}
  inline G4double GetTrpdY2Length()               {return GetTrpdLength(3);}
  //! Get the Trd length along Z
  inline G4double GetTrpdZLength()          	  {return GetTrpdLength(4);}

  //! Get the extruded box length along X-Y-Z
  inline G4double GetTrpdTrudXLength()           {return GetTrpdLength(5);}
  inline G4double GetTrpdTrudYLength()           {return GetTrpdLength(6);}
  inline G4double GetTrpdTrudZLength()           {return GetTrpdLength(7);}

  //! Get the extruded box center position
  inline G4double GetTrpdTrudXPos()              {return GetTrpdLength(8);}
  inline G4double GetTrpdTrudYPos()              {return GetTrpdLength(9);}
  inline G4double GetTrpdTrudZPos()              {return GetTrpdLength(10);}

  //! Get the half-Trd length along an axis   (Trapezoid: X1=0, Y1=1, X2=2, Y2=3, Z=4, Extruded Box: X=5, Y=6, Z=7)
  inline G4double GetTrpdHalfLength(size_t axis)  {return GetTrpdLength(axis)/2.;}

  //! Get the Trd half-length along X1-Y1
  inline G4double GetTrpdX1HalfLength()           {return GetTrpdHalfLength(0);}
  inline G4double GetTrpdY1HalfLength()           {return GetTrpdHalfLength(1);}
  //! Get the Trd half-length along X2-Y2
  inline G4double GetTrpdX2HalfLength()           {return GetTrpdHalfLength(2);}
  inline G4double GetTrpdY2HalfLength()           {return GetTrpdHalfLength(3);}
  //! Get the Trd half-length along Z
  inline G4double GetTrpdZHalfLength()    	  {return GetTrpdHalfLength(4);}

  //! Get the extruded box half-length along X-Y-Z:
  inline G4double GetTrpdTrudXHalfLength()    	  {return GetTrpdHalfLength(5);}
  inline G4double GetTrpdTrudYHalfLength()    	  {return GetTrpdHalfLength(6);}
  inline G4double GetTrpdTrudZHalfLength()    	  {return GetTrpdHalfLength(7);}

  //@}

private:
  //! \name own geometry
  //@{
  G4Trd*               m_trd_solid;                 //!< Solid pointer
  G4Box*               m_box_solid;
  G4SubtractionSolid*  m_trpd_solid;

  G4LogicalVolume*     m_trpd_log; 	      	    //!< logical volume pointer for extruded trapezoid

  //@}

  //! \name parameters
  //@{
  G4double m_trpdLength[11];//!< Trpd lengths 0:dx1 1:dy1 2:dx2 3:dy1 4:dz, 5,6,7:3 box_lengths 8,9,10:3 box_positions

  //@}

  //! Messenger
  GateTrpdMessenger* m_Messenger;

};

MAKE_AUTO_CREATOR_VOLUME(trpd,GateTrpd)

#endif
