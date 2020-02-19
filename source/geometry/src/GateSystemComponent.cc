/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#include "GateSystemComponent.hh"

#include "GateTools.hh"
#include "GateVSystem.hh"
#include "GateSystemComponentMessenger.hh"
#include "GateSystemListManager.hh"
#include "GateTranslationMove.hh"
#include "GateRotationMove.hh"
#include "GateVolumePlacement.hh"
#include "GateOrbitingMove.hh"
#include "GateEccentRotMove.hh"
#include "GateVVolume.hh"
#include "GateObjectChildList.hh"
#include "GateLinearRepeater.hh"
#include "GateAngularRepeater.hh"
#include "GateGenericRepeater.hh"
#include "GateSphereRepeater.hh"
#include "GateSystemComponentList.hh"
#include "GateObjectRepeaterList.hh"


//-------------------------------------------------------------------------------------------
/* Constructor

   itsName:       	  the name chosen for this system-component
   itsMotherComponent:   the mother of the component (0 if top of a tree)
   itsSystem:            the system to which the component belongs
*/
GateSystemComponent::GateSystemComponent(const G4String& itsName,
      	      	      	      	      	 GateSystemComponent* itsMotherComponent,
                                         GateVSystem* itsSystem)
  : GateClockDependent( itsSystem->GetObjectName() + "/" + itsName,false),
    m_system(itsSystem),
    m_creator(0),
    m_messenger(0),
    m_motherComponent(itsMotherComponent)
{

  //  G4cout << " DEBUT Constructeur GateSystemComponent\n";

  //  G4cout << " mcreator(0)" << (m_creator) << Gateendl;

  // If we have a mother, register ourself to it
  if (m_motherComponent)
    m_motherComponent->InsertChildComponent(this);

  // Create a new messenger for this component
  m_messenger = new GateSystemComponentMessenger(this);

  // Create a new child list
  m_childComponentList = new GateSystemComponentList(this,GetObjectName()+"/daughters");

  //  G4cout << " FIN Constructeur GateSystemComponent\n";
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Destructor
GateSystemComponent::~GateSystemComponent()
{
  // Delete the child list
  delete m_childComponentList;

  // Delete the messenger
  delete m_messenger;

}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
/* Method overloading the base-class virtual method Describe().
   This methods prints-out a description of the component

   indent: the print-out indentation (cosmetic parameter)
*/
void GateSystemComponent::Describe(size_t indent)
{
  // Call the base-class class method
  GateClockDependent::Describe(indent);

  // List the creators attached to the system-component
  G4cout << GateTools::Indent(indent) << "Attached to volume: " << ( m_creator ? m_creator->GetObjectName() : G4String("---") ) << Gateendl;

  // Describe the tree of child-components
  m_childComponentList->DescribeChildComponents(indent,true);
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
/* Check whether an creator is connected to the component tree

   anCreator: the creator we want to check

   returns true if the creator is attached to one of the components of the component-tree
*/
G4bool GateSystemComponent::CheckConnectionToCreator(GateVVolume* anCreator)
{

  //  G4cout << " DEBUT GateSystemComponent::CheckConnectionToCreator anCreator =  "  << anCreator->GetObjectName() << Gateendl;
  //  G4cout << " m_creator name = " << m_creator << Gateendl;


  // Return true is we're directly connected to the creator
  if ( anCreator == m_creator )
    return true;

  // Return true if one of our children is connected (directly or indirectly) to the ancestor
  if ( m_childComponentList->CheckConnectionToCreator(anCreator) )
    return true;

  // We're not connected to the creator, and none of our descendants is: return false
  return false;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Attach am creator to the component, provided that the attachment request is valid
// (calls IsValidAttachmentRequest() to check the request validity)
void GateSystemComponent::SetCreator(GateVVolume* anCreator)
{
  //G4cout << " DEBUT GateSystemComponent::SetCreator\n";

  //G4cout << " 2 mcreator(0)" << (m_creator) << Gateendl;

  // Verbose output
  if (nVerboseLevel){
    G4cout   << "[" << GetObjectName() << "::SetInserter]:\n"
             << "\tReceived request for attachment of volume creator '" << anCreator->GetObjectName() << "' to this system-component\n";
  }

  // Check whether the creator is what we really need, i.e. an autoplaced creator
  //ancien  GateVVolume* creatorInserter = dynamic_cast<GateVVolume*>(anCreator);

  GateVVolume* creatorInserter = anCreator;

  //G4cout << " =====> creatorInserter = " << creatorInserter->GetObjectName() << Gateendl;

  if (!creatorInserter) {
    G4cerr   << "[" << GetObjectName() << "::IsValidAttachmentRequest]:\n"
             << "\tThe creator is not a valid creator creator!\n";
    return;
  }


  // We are now sure that we deal with an autoplaced creator
  // Check whether it's a valid one

  //G4cout << " =====> Check !IsValidAttachmentRequest(creatorInserter) = " << (!IsValidAttachmentRequest(creatorInserter)) << Gateendl;



  if (!IsValidAttachmentRequest(creatorInserter))  {
    G4cerr   << "[" << GetObjectName() << "::SetInserter]:\n"
             << "\tIgnoring attachment request\n";
    return;
  }


  // Everything's fine: set the creator pointer
  if (nVerboseLevel){
    G4cout << "[" << GetObjectName() << "::SetInserter]:\n"
           << "\tAttaching volume creator '" << anCreator->GetObjectName() << "' to this component\n";}

  //G4cout << " =====> m_creator = " << m_creator << Gateendl;

  m_creator = creatorInserter;
  //G4cout << " FIN GateSystemComponent::SetCreator\n";

}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Tells whether an creator may be attached to this component
// This virtual method makes a number of tests: is the creator pointer valid,
// does the creator owns a movement-list, does it own a repeater-list...
// It can be and should be overloaded when we want to do specific tests (for specific components)
G4bool GateSystemComponent::IsValidAttachmentRequest(GateVVolume* anCreator) const
{

  // G4cout << " DEBUT GateSystemComponent::IsValidAttachmentRequest\n";


  // G4cout << " 3 mcreator(0)" << (m_creator) << Gateendl;

  // Check that the creator pointer is valid
  if (!anCreator) {
    G4cerr   << "[" << GetObjectName() << "::IsValidAttachmentRequest]:\n"
             << "\tThe creator is null!\n";
    return false;
  }

  //G4cout << " Test1\n";

  // Disrecard the request if an creator is already attached to us
  if (m_creator ) {
    G4cerr << "[" << GetObjectName() << "::IsValidAttachmentRequest]:\n"
           << "\tA volume creator ('" << m_creator->GetObjectName() << "') is already attached to this system component\n";
    return false;
  }

  //G4cout << " Test2\n";
  // Check that there is no inter-system conflict
  // (i.e. that the creator is not already attached to another system)
  GateVSystem* creatorSystem = GateSystemListManager::GetInstance()->FindSystemOfCreator(anCreator);

  //G4cout << " Test22\n";

  if (creatorSystem)
    if ( creatorSystem != GetSystem() ) {
      G4cerr  << "[" << GetObjectName() << "::IsValidAttachmentRequest]:\n"
      	      << "\tThe volume creator '" << anCreator->GetObjectName() << "' or one of its ancestors is already attached to another system ('"
              << creatorSystem->GetObjectName() << "')\n";
      return false;
    }
  //G4cout << " Test3\n";
  // Check that the creator owns a movement list
  GateObjectRepeaterList* moveList = anCreator->GetMoveList();
  if (!moveList) {
    G4cerr   << "[" << GetObjectName() << "::IsValidAttachmentRequest]:\n"
             << "\tThe creator '" << anCreator->GetObjectName() << "' can not be displaced!\n";
    return false;
  }
  //G4cout << " Test4\n";
  // Check that the creator owns a repeater list
  GateObjectRepeaterList* repeaterList = anCreator->GetRepeaterList();
  if (!repeaterList) {
    G4cerr   << "[" << GetObjectName() << "::IsValidAttachmentRequest]:\n"
             << "\tThe creator '" << anCreator->GetObjectName() << "' can not be repeated!\n";
    return false;
  }
  //G4cout << " FIN GateSystemComponent::IsValidAttachmentRequest\n";
  // OK, everything's fine
  return true;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Returns the number of physical volumes created by the creator
size_t GateSystemComponent::GetVolumeNumber() const
{
  return m_creator ? m_creator->GetVolumeNumber() : 0;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Returns one of the physical volumes created by the creator
G4VPhysicalVolume* GateSystemComponent::GetPhysicalVolume(size_t copyNumber) const
{
  return m_creator ? m_creator->GetPhysicalVolume(copyNumber) : 0;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Returns the translation vector for one of the physical volumes created by the creator
G4ThreeVector GateSystemComponent::GetCurrentTranslation(size_t copyNumber) const
{
  static G4ThreeVector defaultPosition;

  G4VPhysicalVolume* aVolume = GetPhysicalVolume(copyNumber);
  return aVolume ? aVolume->GetTranslation() : defaultPosition;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Returns the rotation matrix for one of the physical volumes created by the creator
G4RotationMatrix* GateSystemComponent::GetCurrentRotation(size_t copyNumber) const
{
  G4VPhysicalVolume* aVolume = GetPhysicalVolume(copyNumber);
  return aVolume ? aVolume->GetRotation() : 0 ;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// The function returns the creator's placement move, if one can be find was found in the creator's move list
GateVolumePlacement* GateSystemComponent::FindPlacementMove() const
{
  return FindMove<GateVolumePlacement>();
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// The function returns the creator's translation move, if one can be find was found in the creator's move list
GateTranslationMove* GateSystemComponent::FindTranslationMove() const
{
  return FindMove<GateTranslationMove>() ;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// The function returns the creator's orbiting move, if one can be find was found in the creator's move list
GateOrbitingMove* GateSystemComponent::FindOrbitingMove() const
{
  return FindMove<GateOrbitingMove>();
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// The function returns the creator's rotation move, if one can be find was found in the creator's move list
GateRotationMove* GateSystemComponent::FindRotationMove() const
{
  return FindMove<GateRotationMove>();
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// The function returns the creator's EccentRot move, if a translation can be find was found in the creator's move list
GateEccentRotMove* GateSystemComponent::FindEccentRotMove() const
{
  return FindMove<GateEccentRotMove>();
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// The function returns the creator's translation velocity, if a translation can be find was found in the creator's move list
G4ThreeVector GateSystemComponent::GetTranslationVelocity() const
{
  static const G4ThreeVector defaultVelocity;

  GateTranslationMove* aMove = FindTranslationMove();
  return aMove ? aMove->GetVelocity() : defaultVelocity;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// The function returns the creator's rotation velocity, if a translation can be find was found in the creator's move list
G4double GateSystemComponent::GetRotationVelocity() const
{
  GateRotationMove* aMove = FindRotationMove();
  return aMove ? aMove->GetVelocity() : 0.;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// The function returns the creator's orbiting velocity, if an orbiting can be find was found in the creator's move list
G4double GateSystemComponent::GetOrbitingVelocity() const
{
  GateOrbitingMove* aMove = FindOrbitingMove();
  return aMove ? aMove->GetVelocity() : 0.;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// The function returns the creator's EccentRot velocity, if an eccentrot can be find was found in the creator's move list
G4double GateSystemComponent::GetEccentRotVelocity() const
{
  GateEccentRotMove* aMove = FindEccentRotMove();
  return aMove ? aMove->GetVelocity() : 0.;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// The function returns the creator's EccentRot shift, if an eccentrot can be find was found in the creator's move list
const G4ThreeVector& GateSystemComponent::GetEccentRotShift() const
{
  static const G4ThreeVector defaultShift(0.,0.,0.);
  GateEccentRotMove* aMove = FindEccentRotMove();
  return aMove ? aMove->GetShift() : defaultShift; //0.;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
/* Compute the offset (displacement) between a feature of the creator and a feature of its mother creator

   By default, all alignments are set to align_center, so that we compute the offset between the creator's center
   and the center of its mother's reference frame
   We could select align_left for both alignments: in that case, we would compute the offset between the left edge
   of the creator and the left edge of its mother
   To compute an internal ring diameter from a block position, we actually select align_left for the block and
   align_center for its mother: thus, we compute the distance between the block's left edge and its mother's center

   axis: 	      	      the axis along which we want to compute the offset
   alignment:    	      the feature of the creator for which we want to compute the offset
   it can be its center (align_center), its left border (align_left) or its right border (align_right)
   referenceAlignment:       the feature of the mother volume, with regards to which we compute the offset
   it can be its center (align_center), its left border (align_left) or its right border (align_right)

   Returns the offset along the requested axis between the creator's feature considered and the mother's feature considered
*/
G4double GateSystemComponent::ComputeOffset(size_t axis,Alignment1D alignment,Alignment1D referenceAlignment) const
{
  // No creator
  if (!m_creator)
    return 0.;

  // Get the position of the center of the first creator's volume
  G4double position = GetCurrentTranslation()[axis];

  // Get the volume half-size
  G4double halfLength = m_creator->GetCreator()->GetHalfDimension(axis);

  // Get the volume's mother half-dimensize
  GateVVolume* motherCreator = m_creator->GetMotherCreator();
  if (!motherCreator)
    return 0;
  G4double motherHalfLength = motherCreator->GetHalfDimension(axis);

  // Compute the alignment offset, based on the volume half sizes and on the requested alignments
  G4double alignmentOffset = alignment * halfLength - referenceAlignment * motherHalfLength ;

  // Compute and return the global offset
  return position - alignmentOffset;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Compute the total number of volumes handled by the 'younger' sister-components
G4int GateSystemComponent::ComputeOutputOffset()
{
  if ( !m_motherComponent )
    return 0;

  G4int result =0;

  // Loop in the mother's daughter list until we find the requested daughter
  // dangerous!! can lead to an infinite loop
  size_t i = 0;
  while ( m_motherComponent->GetChildComponent(i) != this )
    {
      result += m_motherComponent->GetChildComponent(i)->GetVolumeNumber();
      ++i;
    }
  //variant, Is this correct?? Is there more than one instance of children per list component??
  //for (size_t i = 0; i<m_motherComponent->GetChildNumber(); i++)
  //if ( m_motherComponent->GetChildComponent(i) == this )
	//return m_motherComponent->GetChildComponent(i)->GetVolumeNumber();

  return result;
  //return 0;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Finds a component in the component tree from its name
GateSystemComponent* GateSystemComponent::FindSystemComponent(const G4String& componentName)
{
  if (componentName==GetObjectName())
    return this;

  return m_childComponentList->FindSystemComponent(componentName);
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Finds the first linear-repeater in the creator's repeater list
GateLinearRepeater* GateSystemComponent::FindLinearRepeater()
{
  return FindRepeater<GateLinearRepeater>();
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Finds the first linear-repeater's repeat number
G4int GateSystemComponent::GetLinearRepeatNumber()
{
  GateLinearRepeater* repeater = FindLinearRepeater();
  return repeater ? repeater->GetRepeatNumber() : 1;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Finds the first linear-repeater's repeat vector
const G4ThreeVector& GateSystemComponent::GetLinearRepeatVector()
{
  static const G4ThreeVector theDefaultRepeatVector;

  GateLinearRepeater* repeater = FindLinearRepeater();
  return repeater ? repeater->GetRepeatVector() : theDefaultRepeatVector;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Finds the first angular-repeater in the creator's repeater list
GateAngularRepeater* GateSystemComponent::FindAngularRepeater()
{
  return FindRepeater<GateAngularRepeater>();
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Finds the first angular-repeater's repeat number
G4int GateSystemComponent::GetAngularRepeatNumber()
{
  GateAngularRepeater* repeater = FindAngularRepeater();
  return repeater ? repeater->GetRepeatNumber() : 1;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Finds the first angular-repeater's repeat angular pitch
G4double GateSystemComponent::GetAngularRepeatPitch()
{
  GateAngularRepeater* repeater = FindAngularRepeater();
  return repeater ? ((repeater->GetAngularSpan() == 360. * degree) ?
                     repeater->GetAngularPitch_1() : repeater->GetAngularPitch_2())
    : 360. * degree;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Finds the first angular-repeater's Modulo number
G4int GateSystemComponent::GetAngularModuloNumber()
{
  GateAngularRepeater* repeater = FindAngularRepeater();
  return repeater ? repeater->GetAngularModuloNumber() : 1;
}
//-------------------------------------------------------------------------------------------




// Finds the first angular-repeater's Z shift 1
G4double GateSystemComponent::GetAngularRepeatZShift1()
{GateAngularRepeater* repeater = FindAngularRepeater(); return repeater ? repeater->GetZShift1() : 0. ; }

G4double GateSystemComponent::GetAngularRepeatZShift2()
{GateAngularRepeater* repeater = FindAngularRepeater(); return repeater ? repeater->GetZShift2() : 0.; }

G4double GateSystemComponent::GetAngularRepeatZShift3()
{GateAngularRepeater* repeater = FindAngularRepeater(); return repeater ? repeater->GetZShift3() : 0.; }

G4double GateSystemComponent::GetAngularRepeatZShift4()
{GateAngularRepeater* repeater = FindAngularRepeater(); return repeater ? repeater->GetZShift4() : 0.; }

G4double GateSystemComponent::GetAngularRepeatZShift5()
{GateAngularRepeater* repeater = FindAngularRepeater(); return repeater ? repeater->GetZShift5() : 0.; }

G4double GateSystemComponent::GetAngularRepeatZShift6()
{GateAngularRepeater* repeater = FindAngularRepeater(); return repeater ? repeater->GetZShift6() : 0.; }

G4double GateSystemComponent::GetAngularRepeatZShift7()
{GateAngularRepeater* repeater = FindAngularRepeater(); return repeater ? repeater->GetZShift7() : 0.; }

G4double GateSystemComponent::GetAngularRepeatZShift8()
{GateAngularRepeater* repeater = FindAngularRepeater(); return repeater ? repeater->GetZShift8() : 0.; }

// Finds the first sphere-repeater in the creator's repeater list
GateSphereRepeater* GateSystemComponent::FindSphereRepeater()
{
  return FindRepeater<GateSphereRepeater>();
}



//-------------------------------------------------------------------------------------------
// Finds the first sphere-repeater's repeat axial pitch
G4double GateSystemComponent::GetSphereAxialRepeatPitch()
{
  GateSphereRepeater* repeater = FindSphereRepeater();
  return repeater ? repeater->GetAxialPitch() : 1;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Finds the first sphere-repeater's repeat azimuthal pitch
G4double GateSystemComponent::GetSphereAzimuthalRepeatPitch()
{
  GateSphereRepeater* repeater = FindSphereRepeater();
  return repeater ? repeater->GetThetaAngle() : 360. * degree;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Finds the first sphere-repeater's axial repeat number
G4int GateSystemComponent::GetSphereAxialRepeatNumber()
{
  GateSphereRepeater* repeater = FindSphereRepeater();
  return repeater ? repeater->GetRepeatNumberWithPhi() : 1;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Finds the first sphere-repeater's azimuthal repeat number
G4int GateSystemComponent::GetSphereAzimuthalRepeatNumber()
{
  GateSphereRepeater* repeater = FindSphereRepeater();
  return repeater ? repeater->GetRepeatNumberWithTheta() : 1;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Finds the first sphere-repeater's radius of replication
G4double GateSystemComponent::GetSphereRadius()
{
  GateSphereRepeater* repeater = FindSphereRepeater();
  return repeater ? repeater->GetRadius() : 0.;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Finds the first generic repeater in the creator's repeater list
GateGenericRepeater* GateSystemComponent::FindGenericRepeater()
{
  return FindRepeater<GateGenericRepeater>();
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Finds the first linear-repeater's repeat number
G4int GateSystemComponent::GetGenericRepeatNumber()
{
  GateGenericRepeater* repeater = FindGenericRepeater();
  return repeater ? repeater->GetRepeatNumber() : 1;
}
//-------------------------------------------------------------------------------------------



void GateSystemComponent::setInCoincidenceWith(G4String aRsectorName )
{
  size_t pos = GetObjectName().rfind( "/");
  G4String thename = GetObjectName().substr( pos + 1);
  G4cout << " my name is  " << thename<< Gateendl;
  if ( thename ==  aRsectorName ) return;


  G4cout << " GateSystemComponent::setInCoincidenceWith  entered  for " <<GetObjectName()<< Gateendl;
  G4cout << " rsector name parameter " << aRsectorName<< Gateendl;

  if ( m_coincidence_rsector.empty() != 1 )
    {
      std::vector<G4String>::iterator it;
      G4cout << " vector is not empty  looking for " << aRsectorName<< Gateendl;
      G4cout << " it contains \n";
      for (size_t i = 0; i < m_coincidence_rsector.size(); i++ )
        G4cout << m_coincidence_rsector[i]<<"  ";
      G4cout<< Gateendl;
      it = std::find( m_coincidence_rsector.begin() , m_coincidence_rsector.end
                      () , aRsectorName );
      if ( it == m_coincidence_rsector.end() )        {
        GateSystemComponent* theComponent = m_system->FindBoxCreatorComponent(aRsectorName );
        if ( theComponent != 0 )
          { size_t pos = GetObjectName().rfind( "/");           G4String thename = GetObjectName().substr( pos + 1);
            m_coincidence_rsector.push_back(aRsectorName);           theComponent->setInCoincidenceWith( thename );
            for (size_t i = 0; i < m_coincidence_rsector.size(); i++ )theComponent
                                                                        ->setInCoincidenceWith( m_coincidence_rsector[i] );
            G4cout<<"GateSystemComponent::setInCoincidenceWith() :: setting " <<
              thename<< " in coincidence with " << aRsectorName << Gateendl;
          }         else G4cout<<"GateSystemComponent::setInCoincidenceWith() :: WARNING Component named " <<aRsectorName<<" was not found. Ignored.";
      } else { G4cout << "already found  exiting \n";return; }
    }     else
    { G4cout << " vector is empty  looking for " << aRsectorName<< Gateendl
        ;
      GateSystemComponent* theComponent = m_system->FindBoxCreatorComponent( aRsectorName );
      if ( theComponent != 0 )
        {size_t pos = GetObjectName().rfind( "/");
          G4String thename = GetObjectName().substr( pos + 1 );
          m_coincidence_rsector.push_back( aRsectorName );
          G4cout<<"GateSystemComponent::setInCoincidenceWith() :: setting " <<thename << " in coincidence with " << aRsectorName << Gateendl;             theComponent->setInCoincidenceWith( thename );
        }
      else G4cout<<"GateSystemComponent::setInCoincidenceWith() :: WARNING Component named " <<aRsectorName<<" was not found. Ignored.\n";
    }
}


G4int GateSystemComponent::IsInCoincidenceWith(G4String aRsectorName )
{
  if ( m_coincidence_rsector.empty() != 1 )
    {
      std::vector<G4String>::iterator it;
      it = std::find( m_coincidence_rsector.begin() , m_coincidence_rsector.end() , aRsectorName );
      if ( it != m_coincidence_rsector.end() ) return 1;
    }
  return 0;
}
