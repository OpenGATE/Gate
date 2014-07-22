/*
 * GateGenericTrap.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATEGENERICTRAP_H
#define GATEGENERICTRAP_H

#include "globals.hh"
#include "GateVVolume.hh"
#include "GateVolumeManager.hh"
#include "G4TwoVector.hh"
#include "G4ThreeVector.hh"
#include "G4VisExtent.hh"
#include "G4GenericTrap.hh"

class G4GenericTrap;
class GateGenericTrapMessenger;

class GateGenericTrap : public GateVVolume
{
public:
  //! Constructor
  GateGenericTrap(const G4String& itsName,
                 G4bool acceptsChildren=true,
                 G4int depth=0);

  GateGenericTrap(const G4String& itsName,
                 const G4String& itsMaterialName,
                 std::vector<G4TwoVector> itsVertices,
                 G4double itsZLength,
                 G4bool acceptsChildren=true,
                 G4int depth=0);
  //! Destructor
  virtual ~GateGenericTrap();

  FCT_FOR_AUTO_CREATOR_VOLUME(GateGenericTrap)

  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly);

  virtual void DestroyOwnSolidAndLogicalVolume();

  /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
    \brief a description of the creator

    \param level: the print-out indentation (cosmetic parameter)
  */
  virtual void DescribeMyself(size_t level);
  inline G4double GetHalfDimension(size_t axis) {
    G4VisExtent extent = m_generictrap_solid->GetExtent();
    if (axis==0) {
	return (extent.GetXmax() - extent.GetXmin())/2.0;
    } else if (axis==1) {
	return (extent.GetYmax() - extent.GetYmin())/2.0;

    } else if (axis==2) {
	return m_zLength/2.0;
    }
    return 0.0;
  }

  //! Overload of the dummy virtual method ComputeMyOwnVolume() defined by the base-class
  //! Returns the volume of the solid
  G4double ComputeMyOwnVolume()  const;

  //! \name getters and setters
  //@{
  inline std::vector<G4TwoVector> GetGenericTrapVertices() 	{ return m_vertices; }
  inline G4double GetGenericTrapZLength() 	{ return m_zLength; }

  void SetGenericTrapVertices(std::vector<G4TwoVector> val)	{ m_vertices = val; }
  void SetGenericTrapVertex(G4int index, G4TwoVector val)	{ m_vertices[index] = val; }
  void SetGenericTrapVertex(G4int index, G4ThreeVector val)	{ m_vertices[index].set( val.x(), val.y() ); }
  void SetGenericTrapVertex(G4int index, G4double x, G4double y)	{ m_vertices[index].set(x, y); }
  void SetGenericTrapZLength(G4double val)	{ m_zLength = val; }
  //@}


private:
  //! \name own geometry
  //@{
  G4GenericTrap*          m_generictrap_solid;       	    //!< Solid pointer
  G4LogicalVolume*   m_generictrap_log; 	      	    //!< logical volume pointer
  //@}

  //! \name parameters
  //@{
  G4double m_zLength;		//!< z length
  std::vector<G4TwoVector> m_vertices; //!< vertices
  //@}

  //! Messenger
  GateGenericTrapMessenger* m_Messenger;
};

MAKE_AUTO_CREATOR_VOLUME(generictrapezoid,GateGenericTrap)


#endif /* GATEGENERICTRAP_H */
