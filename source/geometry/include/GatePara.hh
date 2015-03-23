/*
 * GatePara.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATEPARA_H
#define GATEPARA_H

#include "globals.hh"
#include "GateVVolume.hh"
#include "GateVolumeManager.hh"
#include "G4ThreeVector.hh"

class G4Para;
class GateParaMessenger;

class GatePara : public GateVVolume
{
public:
  //! Constructor
  GatePara(const G4String& itsName,
                 G4bool acceptsChildren=true,
                 G4int depth=0);

  GatePara(const G4String& itsName,
                 const G4String& itsMaterialName,
                 G4double itsXLength,
                 G4double itsYLength,
                 G4double itsZLength,
                 G4double itsAlpha,
                 G4double itsTheta,
                 G4double itsPhi,
                 G4bool acceptsChildren=true,
                 G4int depth=0);
  //! Destructor
  virtual ~GatePara();

  FCT_FOR_AUTO_CREATOR_VOLUME(GatePara)

  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly);

  virtual void DestroyOwnSolidAndLogicalVolume();

  /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
    \brief a description of the creator

    \param level: the print-out indentation (cosmetic parameter)
  */
  virtual void DescribeMyself(size_t level);

  inline G4double GetHalfDimension(size_t axis) {
    return m_length[axis]/2.0;
  }

  //! Overload of the dummy virtual method ComputeMyOwnVolume() defined by the base-class
  //! Returns the volume of the solid
  G4double ComputeMyOwnVolume()  const;

  //! \name getters and setters
  //@{
  inline G4double GetParaXLength() 	{ return m_length.x(); }
  inline G4double GetParaYLength() 	{ return m_length.y(); }
  inline G4double GetParaZLength() 	{ return m_length.z(); }
  inline G4double GetParaAlpha() 	{ return m_alpha; }
  inline G4double GetParaTheta() 	{ return m_theta; }
  inline G4double GetParaPhi() 		{ return m_phi; }

  void SetParaXLength(G4double val)	{ m_length.setX(val); }
  void SetParaYLength(G4double val)	{ m_length.setY(val); }
  void SetParaZLength(G4double val)	{ m_length.setZ(val); }
  void SetParaAlpha(G4double val)	{ m_alpha = val; }
  void SetParaTheta(G4double val)	{ m_theta = val; }
  void SetParaPhi(G4double val)		{ m_phi = val; }
  //@}


private:
  //! \name own geometry
  //@{
  G4Para*          m_para_solid;       	    //!< Solid pointer
  G4LogicalVolume*   m_para_log; 	      	    //!< logical volume pointer
  //@}

  //! \name parameters
  //@{
  G4ThreeVector m_length;	//!< Size
  G4double m_alpha;		//!< Angle formed by the y axis and by the plane joining the centre of the faces parallel to the z-x plane at -dy/2 and +dy/2
  G4double m_theta;		//!< Polar angle of the line joining the centres of the faces at -dz/2 and +dz/2 in z
  G4double m_phi;		//!< Azimuthal angle of the line joining the centres of the faces at -dz/2 and +dz/2 in z
  //@}

  //! Messenger
  GateParaMessenger* m_Messenger;
};

MAKE_AUTO_CREATOR_VOLUME(para,GatePara)


#endif /* GATEPARA_H */
