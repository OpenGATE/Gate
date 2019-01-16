

#include "GatePrimTrackInformation.hh"
#include "G4ios.hh"
#include "G4SystemOfUnits.hh"    

G4ThreadLocal G4Allocator<GatePrimTrackInformation> *
                                   aTrackInformationAllocator = 0;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
GatePrimTrackInformation::GatePrimTrackInformation()
  : G4VUserTrackInformation()
{
    fOriginalTrackID = 0;
    fParticleDefinition = 0;
    fOriginalPosition = G4ThreeVector(0.,0.,0.);
    fOriginalMomentum = G4ThreeVector(0.,0.,0.);
    fOriginalEnergy = 0.;
    fOriginalTime = 0.;

    m_energyPrimaryTrack=-1;
    m_PDGPrimaryTrack=0;




}


GatePrimTrackInformation::GatePrimTrackInformation(const G4Track* aTrack)
  : G4VUserTrackInformation()
{
    fOriginalTrackID = aTrack->GetTrackID();
    fParticleDefinition = aTrack->GetDefinition();
    fOriginalPosition = aTrack->GetPosition();
    fOriginalMomentum = aTrack->GetMomentum();
    fOriginalEnergy = aTrack->GetTotalEnergy();
    fOriginalTime = aTrack->GetGlobalTime();


    //MAybe include some other parameter to set my an specific track
    m_energyPrimaryTrack=-1;
     m_PDGPrimaryTrack=0;
}


GatePrimTrackInformation
::GatePrimTrackInformation(const GatePrimTrackInformation* aTrackInfo)
  : G4VUserTrackInformation()
{
    fOriginalTrackID = aTrackInfo->fOriginalTrackID;
    fParticleDefinition = aTrackInfo->fParticleDefinition;
    fOriginalPosition = aTrackInfo->fOriginalPosition;
    fOriginalMomentum = aTrackInfo->fOriginalMomentum;
    fOriginalEnergy = aTrackInfo->fOriginalEnergy;
    fOriginalTime = aTrackInfo->fOriginalTime;


    m_energyPrimaryTrack=aTrackInfo->m_energyPrimaryTrack;
    m_PDGPrimaryTrack=aTrackInfo->m_PDGPrimaryTrack;



}


GatePrimTrackInformation::~GatePrimTrackInformation()
{;}




GatePrimTrackInformation& GatePrimTrackInformation
::operator =(const GatePrimTrackInformation& aTrackInfo)
{
    fOriginalTrackID = aTrackInfo.fOriginalTrackID;
    fParticleDefinition = aTrackInfo.fParticleDefinition;
    fOriginalPosition = aTrackInfo.fOriginalPosition;
    fOriginalMomentum = aTrackInfo.fOriginalMomentum;
    fOriginalEnergy = aTrackInfo.fOriginalEnergy;
    fOriginalTime = aTrackInfo.fOriginalTime;


    m_energyPrimaryTrack=aTrackInfo.m_energyPrimaryTrack;
    m_PDGPrimaryTrack=aTrackInfo.m_PDGPrimaryTrack;
    return *this;
}


void GatePrimTrackInformation::SetEPrimTrackInformation(const G4Track* aTrack)
{


    //Meter la track con la info de la source ? (creo que en este caso era la detectada)si uso la primaria
    //creo qeu eso me sobra
     m_energyPrimaryTrack= aTrack->GetTotalEnergy();
     m_PDGPrimaryTrack=aTrack->GetParticleDefinition()->GetPDGEncoding();

}




void GatePrimTrackInformation::Print() const
{

    G4cout
      << "Original primary track ID " << fOriginalTrackID << " (" 
      << fParticleDefinition->GetParticleName() << ","
     << fOriginalEnergy/GeV << "[GeV])" << G4endl;
}

