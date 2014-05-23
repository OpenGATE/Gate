/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEVGLOBALPLACEMENT_H
#define GATEVGLOBALPLACEMENT_H 1

#include "globals.hh"
#include <vector>
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include "GateClockDependent.hh"
#include "GatePlacementQueue.hh"

class GateVVolume;

//-------------------------------------------------------------------------------------------------
/*! \class  GateVGlobalPlacement
  \brief  Framework base-class for all repeater classes
    
  - GateVGlobalPlacement - by Daniel.Strul@iphe.unil.ch 
    
  - The GateVObjectMove is derived from GateClockDependent. It is responsible for
  computing the orientation and position of a series of copies of a volume, according
  to a repetition algorithm equation programmed by the user.
      
  - GateVGlobalPlacement is an abstract class: specific repeaters can be modelled by 
  deriving application classes from it. In each application class,
  it is the responsability of the developper to implement a model of the repetition
  in the pure virtual method PushMyPlacements(). 
      
  - It is also the responsability of the developper to implement the pure virtual method DescribeMyself().
  This method should print-out a description of the repeater (i.e. its parameters). 
      
*/      
class GateVGlobalPlacement : public GateClockDependent
{
public:
  /*! \brief Constructor

    \param itsObjectInserter: the inserter to which the repeater is appended
    \param itsName:       	  the name chosen for this repeater
  */    
  GateVGlobalPlacement(GateVVolume* itsObjectInserter, const G4String& itsName);
  //! Destructor
  virtual ~GateVGlobalPlacement();

  /*! \brief Implementation of the pure virtual method defined by the base-class
    \brief Calls DescribeMyself() to print-out a description of the repeater

    \param indent: the print-out indentation (cosmetic parameter)
  */    
  virtual void Describe(size_t indent=0);

  /*! Public method to be called to repeat a series of placements placed into an input queue
    
    \param  motherQueue:  the input queue to repeat
      
    If the repeater is enabled, it repeats each placement in the input queue using the user-defined
    virtual method PushMyPlacements. The series thus obtained are all placed into the repeater placement 
    queue, which is returned to the caller. 
    If the repeater is disabled, it directly places a copy of each input placement into the
    output queue, so that the output queue is a direct copy of the input queue.
    In both cases (enabled or disabled), the input queue is emptied as it is processed.
  */
  virtual GatePlacementQueue* ComputePlacements(GatePlacementQueue* motherQueue);

protected:
  /*! \brief Pure virtual method (to be implemented in sub-classes)
    \brief Must compute a series of placements (position+orientation) for a series of copies of the
    \brief orignal volume. These placements must be stored in the placement queue

    \param currentRotationMatrix: the rotation matrix that defines the current orientation of the volume
    \param currentPosition:       the vector that defines the current position of the volume
    \param aTime:                 the current time
  */    
  virtual void PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
                                const G4ThreeVector& currentPosition,
                                G4double aTime) = 0;
  /*! \brief Pure virtual method (to be implemented in sub-classes)
    \brief Must print-out a short description of the repeater

    \param indent: the print-out indentation (cosmetic parameter)
  */    
  virtual void DescribeMyself(size_t indent)=0;

public:
  //! \name management of the placement queue
  //@{
  //! Places a new placement into the queue
  virtual inline void PushBackPlacement(GatePlacement placement)
  { 
    m_placementQueue.push_back(placement) ; }
  //! Makes a new placement from a rotation matrix and a translation, and pushes this placement
  //! into the placement queue
  virtual inline void PushBackPlacement(const G4RotationMatrix& rotationMatrix,const G4ThreeVector& position)
  { m_placementQueue.push_back(rotationMatrix,position) ; }
  //! Extract and return the placement in the front of the queue
  virtual inline GatePlacement PopFrontPlacement()
  { return m_placementQueue.pop_front() ; }
 
  virtual inline const GatePlacement & FrontPlacement()
  { return m_placementQueue.front() ; }
 
  virtual int size() { return m_placementQueue.size() ; }  
  //@}      
      
  //! \name getters and setters
  //@{
  //! Return the object to which the repeater is attached
  virtual inline GateVVolume* GetVolumeCreator() const 
  { return m_objectInserter;}     
  //@}

  //inline virtual void ComputeParameters(G4double aTime) {}
  inline virtual void ComputeParameters(G4double ) {}
 
protected:
  GateVVolume* m_objectInserter;    //!< object to which the repeater is attached
  GatePlacementQueue m_placementQueue;      //!< Repeater private placement queue
};

#endif

