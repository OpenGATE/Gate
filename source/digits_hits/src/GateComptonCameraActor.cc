
#include "GateComptonCameraActor.hh"
#ifdef G4ANALYSIS_USE_ROOT

#include "GateComptonCameraActorMessenger.hh"
#include "GateMiscFunctions.hh"

// g4 // inserted 30 Jan 2016:
#include <G4EmCalculator.hh>
#include <G4VoxelLimits.hh>
#include <G4NistManager.hh>
#include <G4PhysicalConstants.hh>



//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateComptonCameraActor::GateComptonCameraActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateComptonamerActor() -- begin\n");

  G4cout<<"###########################CONSTRUCTOR of GATEComptonCameraActor#############################################"<<G4endl;



  mEmin = 0.;
  mEmax = 50.;
  mENBins = 100;


   mVolmin=0;
   mVolmax=6;
   mVolNBins=6;
  


  mEdepmin = 0.;
  mEdepmax = 50.;
  mEdepNBins = 100;

  Ei = 0.;
  Ef = 0.;
  newEvt = true;
  newTrack = true;
  sumNi=0.;
  nTrack=0;
  sumM1=0.;
  sumM2=0.;
  sumM3=0.;
  edep = 0.;
  //nDaughterBB=0;
  counterConstructF=0;

  mSaveAsTextFlag = true;
  
  emcalc = new G4EmCalculator;


  //This first line ok
  //GateVVolume * attachVolume =GetVolume();
  //This line makes segmentation fault
  //I wanted to know the number of childs
  //attachVolume->GetLogicalVolume();
  //This line makes segmentation fault
 // nDaughterBB=attachVolume->GetLogicalVolume()->GetNoDaughters();
  //std::cout<< attachVolume->GetLogicalVolume()->GetNoDaughters()<<std::endl;
  //nDaughterBB=mVolume->GetLogicalVolume()->GetNoDaughters();

  pMessenger = new GateComptonCameraActorMessenger(this);



  GateDebugMessageDec("Actor",4,"GateComptonCamera() -- end\n");


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateComptonCameraActor::~GateComptonCameraActor(){
  GateDebugMessageInc("Actor",4,"~GateComptonCameraActor() -- begin\n");
  GateDebugMessageDec("Actor",4,"~GateComptonCameraActor() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Construct
void GateComptonCameraActor::Construct()
{

 //???I have to call it? What for?
 GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);
  EnableEndOfEventAction(true); // for save every n



  nDaughterBB=mVolume->GetLogicalVolume()->GetNoDaughters();
 // unsigned int numDaug=GetVolume()->GetLogicalVolume()->GetNoDaughters();
  attachPhysVolumeName=mVolume->GetPhysicalVolumeName();


    edepInEachLayerEvt=new double [nDaughterBB];

       for ( unsigned int i=0; i < nDaughterBB; i++) {
           edepInEachLayerEvt[i]=0.0;
          layerNames.push_back( mVolume->GetLogicalVolume()->GetDaughter(i)->GetName());
          //G4cout<<"layers names="<<layerNames.back()<<G4endl;
       }





  //############################3
  //root file
 // !!right know I do not know from where it takes Filename b
  pTfile = new TFile(mSaveFilename,"RECREATE");

  //A three for each layer!!! First try saving for one layer

       for(unsigned int i=0; i<nDaughterBB;i++){
           pSingles2.emplace_back(new TTree(layerNames.at(i), "Singles tree"));
          pSingles2.at(i)->Branch("edepEvt",&edepInEachLayerEvt[i],"edepEvt/D");

       }



  //pSingles.at(0)->Branch("edepEvt",&edptempAb,"edepEvt/F");
  pEnergySpectrum = new TH1D("energySpectrum","Energy Spectrum",GetENBins(),GetEmin() ,GetEmax() );
  pEnergySpectrum->SetXTitle("Energy (MeV)");
  
  pEdep  = new TH1D("edepHisto","Energy deposited per event",GetEdepNBins(),GetEdepmin() ,GetEdepmax() );
  pEdep->SetXTitle("E_{dep} (MeV)");


  pEdepTrack  = new TH1D("edepTrackHisto","Energy deposited per track",GetEdepNBins(),GetEdepmin() ,GetEdepmax() );
  pEdepTrack->SetXTitle("E_{dep} (MeV)");



 pVolumeName  = new TH1D("volumeIDTrackHisto","volume ID per track",GetVolNBins(), GetVolIDmin(), GetVolIDmax()  );
   pVolumeName->SetCanExtend(TH1::kAllAxes);
  pVolumeName->SetXTitle("Volume Name");

 /* //TEST
  pEdepAbs  = new TH1D("edepAHisto","Energy absorber deposited per event",GetEdepNBins(),GetEdepmin() ,GetEdepmax() );
  pEdepAbs->SetXTitle("E_{dep} (MeV)");*/



  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateComptonCameraActor::SaveData()
{
  GateVActor::SaveData();
  pTfile->Write();
  //pTfile->Close();

  // Also output data as txt if enabled
  if (mSaveAsTextFlag) {
    SaveAsText(pEnergySpectrum, mSaveFilename);
    SaveAsText(pEdep, mSaveFilename);
    SaveAsText(pEdepTrack, mSaveFilename);
    SaveAsText(pVolumeName, mSaveFilename);
   
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::ResetData()
{
  pEnergySpectrum->Reset();
  pEdep->Reset();
  pEdepTrack->Reset();
  pVolumeName->Reset();
  nEvent = 0;
 //Ravisar si algo mas que reeset if(edeplayer) delete [] edepInEachLayerEvt
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::BeginOfRunAction(const G4Run *)
{
    G4cout<<"###########################BEGIN ACTION#############################################"<<G4endl;

  GateDebugMessage("Actor", 3, "GateComptonCameraActor -- Begin of Run\n");
  ResetData();


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::BeginOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateComptonCameraActor -- Begin of Event\n");
  newEvt = true;
  edep = 0.;
  tof  = 0;
  //edepInEachLayerEvt.assign(nDaughterBB,0.0);
  for(unsigned int i=0;i<nDaughterBB;i++){
      edepInEachLayerEvt[i]=0.0;

  }
  edptempAb=0;


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::EndOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- End of Event\n");
  if (edep > 0) {
    pEdep->Fill(edep/MeV);



  //pEdepAbs->Fill(edepInEachLayerEvt[0]);
  //G4cout<<"edepAbs="<< edptempAb<<"edepAbs layer="<< edepInEachLayerEvt[0]<<G4endl;

        for(unsigned int i=0;i<nDaughterBB;i++){
           pSingles2.at(i)->Fill();

        }
    }
  nEvent++;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::PreUserTrackingAction(const GateVVolume * , const G4Track* t)
{
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- Begin of Track\n");

  newTrack = true;
  //AE  no hace nada con esto
  if (t->GetParentID()==1) nTrack++;
  edepTrack = 0.;

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::PostUserTrackingAction(const GateVVolume *, const G4Track* t)
{
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- End of Track\n");
  //double eloss = Ei-Ef;
  if (edepTrack > 0)  pEdepTrack->Fill(edepTrack/MeV,t->GetWeight() );




}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::UserSteppingAction(const GateVVolume *  , const G4Step* step)
{
  assert(step->GetTrack()->GetWeight() == 1.); // edep doesnt handle weight


  G4TouchableHandle touchable=step->GetPostStepPoint()->GetTouchableHandle();
  VolNameStep=touchable->GetVolume(0)->GetName();
  pVolumeName->Fill(VolNameStep,1);

  for(unsigned int i=0; i<nDaughterBB;i++){
      //A este nivel las energias estan bien
      //G4cout<<"Volume name step= "<<VolNameStep<<"   layer= "<<layerNames.at(i)<<"  energ="<<step->GetTotalEnergyDeposit()<<"  energy (Mev)"<<step->GetTotalEnergyDeposit()/MeV<<G4endl;
     if(VolNameStep==layerNames.at(i)){

        edepInEachLayerEvt[i] += step->GetTotalEnergyDeposit()/MeV;
        if (i==0) edptempAb+=step->GetTotalEnergyDeposit()/MeV;
        break;
     }
  }

  //Esta linea que cogia el puntero al volumen del input de la funcion me crea segmentation fault
  //G4cout<< " number of daughter volumen ste"<<volStpA->GetLogicalVolume()->GetNoDaughters()<<"name ofmother log vol"<<volStpA->GetMotherLogicalVolume()<<G4endl;


  //ACUMULATE THE EVENT IN THE volume  in whit preStep


  edep += step->GetTotalEnergyDeposit();

 //AE Por que acumula tambien el track????????????????
  edepTrack += step->GetTotalEnergyDeposit();

  //cout << "--- " << step->GetTrack()->GetTrackID() << " " << step->GetTrack()->GetParentID() << endl;
  if (newEvt) {
    double pretof = step->GetPreStepPoint()->GetGlobalTime();
    double posttof = step->GetPostStepPoint()->GetGlobalTime();
    tof = pretof + posttof;
    tof /= 2;
    //cout << "****************** new event tof=" << pretof << "/" << posttof << "/" << tof << " edep=" << edep << endl;
    newEvt = false;
  } else {
    double pretof = step->GetPreStepPoint()->GetGlobalTime();
    double posttof = step->GetPostStepPoint()->GetGlobalTime();
    double ltof = pretof + posttof;
    ltof /= 2;
    //cout << "****************** diff tof=" << ltof << " edep=" << edep << endl;
// AE if the event is new save the value of the private varibale tof , but otherwise nothing happen?
  }

  
  Ef=step->GetPostStepPoint()->GetKineticEnergy();
  if(newTrack){
    Ei=step->GetPreStepPoint()->GetKineticEnergy();
    pEnergySpectrum->Fill(Ei/MeV,step->GetTrack()->GetWeight());
    //AE Initial energy of new tracks
    newTrack=false;
  }
  
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateComptonCameraActor::SaveAsText(TH1D * histo, G4String initial_filename)
{
  // Compute new filename: remove extension, add name of the histo, add txt extension
  std::string filename = removeExtension(initial_filename);
  filename = filename + "_"+histo->GetName()+".txt";

  // Create output file
  std::ofstream oss;
  OpenFileOutput(filename, oss);

  // FIXME
  if (mSaveAsDiscreteSpectrumTextFlag) {
    oss << "# First line is two numbers " << std::endl
        << "#     First value is '1', it means 'discrete energy mode'" << std::endl
        << "#     Second value is ignored" << std::endl
        << "# Other lines : 2 columns. 1) energy 2) probability (nb divided by NbEvent)" << std::endl
        << "# Number of bins = " << mDiscreteSpectrum.size() << std::endl
        << "# Number of events: " << nEvent << std::endl
        << "1 0" << std::endl;
    for(int i=0; i<mDiscreteSpectrum.size(); i++) {
      oss << mDiscreteSpectrum.GetEnergy(i) << " " << mDiscreteSpectrum.GetValue(i)/nEvent << std::endl;
    }
    oss.close();
  }
  else {
    // write as text file with header and 2 columns: 1) energy 2) probability
    // The header is two numbers:
    ///    1 because it is mode 1 (see gps UserSpectrum)
    //     Emin of the histo

    // Root convention
    // For all histogram types: nbins, xlow, xup
    //         bin = 0;       underflow bin
    //         bin = 1;       first bin with low-edge xlow INCLUDED
    //         bin = nbins;   last bin with upper-edge xup EXCLUDED
    //         bin = nbins+1; overflow bin
    oss << "# First line is two numbers " << std::endl
        << "#     First value is '2', it means 'histogram mode'" << std::endl
        << "#     Second value is 'Emin' of the histogram" << std::endl
        << "# Other lines : 2 columns. 1) energy 2) probability (nb divided by NbEvent)" << std::endl
        << "# Number of bins = " << histo->GetNbinsX() << std::endl
        << "# Content below the first bin: " << histo->GetBinContent(0) << std::endl
        << "# Content above the last  bin: " << histo->GetBinContent(histo->GetNbinsX()+2) << std::endl
        << "# Content above the last  bin: " << histo->GetBinContent(histo->GetNbinsX()+2) << std::endl
        << "# Number of events: " << nEvent << std::endl
        << "2 " << histo->GetBinLowEdge(1) << std::endl; // start at 1
    for(int i=1; i<histo->GetNbinsX()+1; i++) {
      oss << histo->GetBinLowEdge(i) + histo->GetBinWidth(i) << " " << histo->GetBinContent(i)/nEvent << std::endl;
    }
    oss.close();
  }
}
//-----------------------------------------------------------------------------

#endif
