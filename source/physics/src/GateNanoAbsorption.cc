#include "GateNanoAbsorption.hh"
#include "G4ios.hh"
#include "G4OpProcessSubType.hh"

G4double GateNanoAbsorption::GetMeanFreePath(const G4Track& aTrack,
                 G4double ,
                G4ForceCondition* )
 {
   const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
         const G4Material* aMaterial = aTrack.GetMaterial();
 
   G4double thePhotonMomentum = aParticle->GetTotalMomentum();
 
   G4MaterialPropertiesTable* aMaterialPropertyTable;
   G4MaterialPropertyVector* AttenuationLengthVector;
   
         G4double AttenuationLength = DBL_MAX;
 
   aMaterialPropertyTable = aMaterial->GetMaterialPropertiesTable();
 
   if ( aMaterialPropertyTable ) {
      AttenuationLengthVector = aMaterialPropertyTable->
                                                 GetProperty("NANOABSLENGTH");
            if ( AttenuationLengthVector ){
              AttenuationLength = AttenuationLengthVector->
                                          Value(thePhotonMomentum);
            }
            else {
 //             G4cout << "No Absorption length specified" << G4endl;
            }
         } 
         else {
 //           G4cout << "No Absorption length specified" << G4endl;
         }
 
         return AttenuationLength;
 }
 
